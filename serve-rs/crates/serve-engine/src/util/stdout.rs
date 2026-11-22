//! The process's stdout and stderr as serve/server.py main() sets them:
//! `sys.stdout, sys.stderr = TeeStream(ColorStream.wrap(...), logs, "srv")`. Before install() the text goes to the
//! plain streams. Every call is thread-safe; out() and err() write a whole line under the stream's lock.
//! As Python's text streams on Windows do, each "\n" reaches the OS as "\r\n". Text is written as UTF-8 (Python off a
//! console writes the ANSI code page there; ASCII text is the same).
use super::logbuf::{ColorStream, LogBuffer, TeeStream, TextWrite};
use std::io::{IsTerminal, Write};
use std::sync::{Arc, Mutex, RwLock};

#[derive(Clone, Copy)]
enum Which {
    Out,
    Err,
}

/// The OS stream itself, with Python's newline translation.
struct Raw(Which);

impl TextWrite for Raw {
    fn write(&self, text: &str) {
        let text = if cfg!(windows) { text.replace('\n', "\r\n") } else { text.to_string() };
        let _ = match self.0 {
            Which::Out => {
                let mut o = std::io::stdout().lock();
                o.write_all(text.as_bytes()).and_then(|_| o.flush())
            }
            Which::Err => {
                let mut o = std::io::stderr().lock();
                o.write_all(text.as_bytes()).and_then(|_| o.flush())
            }
        };
    }

    fn flush(&self) {
        let _ = match self.0 {
            Which::Out => std::io::stdout().flush(),
            Which::Err => std::io::stderr().flush(),
        };
    }

    fn isatty(&self) -> bool {
        match self.0 {
            Which::Out => std::io::stdout().is_terminal(),
            Which::Err => std::io::stderr().is_terminal(),
        }
    }
}

struct Sinks {
    out: Box<dyn TextWrite>,
    err: Box<dyn TextWrite>,
}

static SINKS: RwLock<Option<Sinks>> = RwLock::new(None);
static OUT_LOCK: Mutex<()> = Mutex::new(());
static ERR_LOCK: Mutex<()> = Mutex::new(());

/// Tee both streams into `logs` (source "srv"), colored on a terminal.
pub fn install(logs: Arc<LogBuffer>) {
    let out = TeeStream::new(ColorStream::wrap(Box::new(Raw(Which::Out))), logs.clone(), "srv");
    let err = TeeStream::new(ColorStream::wrap(Box::new(Raw(Which::Err))), logs, "srv");
    *SINKS.write().unwrap() = Some(Sinks { out: Box::new(out), err: Box::new(err) });
}

fn with(which: Which, f: impl FnOnce(&dyn TextWrite)) {
    let g = SINKS.read().unwrap();
    match (&*g, which) {
        (Some(s), Which::Out) => f(s.out.as_ref()),
        (Some(s), Which::Err) => f(s.err.as_ref()),
        (None, w) => f(&Raw(w)),
    }
}

/// `print(text, flush=True)`.
pub fn out(text: &str) {
    let _l = OUT_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    with(Which::Out, |w| {
        w.write(text);
        w.write("\n");
        w.flush();
    });
}

/// `print(text, file=sys.stderr)`.
pub fn err(text: &str) {
    let _l = ERR_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    with(Which::Err, |w| {
        w.write(text);
        w.write("\n");
        w.flush();
    });
}

/// `sys.stdout.write(text)`: partial lines, escape codes and `\r` redraws.
pub fn write_raw_stdout(text: &str) {
    let _l = OUT_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    with(Which::Out, |w| w.write(text));
}

/// `sys.stderr.write(text)`.
pub fn write_raw_stderr(text: &str) {
    let _l = ERR_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    with(Which::Err, |w| w.write(text));
}

/// `sys.stdout.flush()` (a ColorStream writes the line it holds).
pub fn flush_stdout() {
    let _l = OUT_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    with(Which::Out, |w| w.flush());
}

/// `sys.stdout.isatty()`.
pub fn stdout_is_tty() -> bool {
    std::io::stdout().is_terminal()
}

/// `os.get_terminal_size(sys.stdout.fileno()).columns`; None when stdout is not a terminal.
pub fn stdout_columns() -> Option<i64> {
    #[cfg(windows)]
    {
        use windows_sys::Win32::System::Console::{GetConsoleScreenBufferInfo, GetStdHandle, CONSOLE_SCREEN_BUFFER_INFO,
                                                  STD_OUTPUT_HANDLE};
        // SAFETY: plain Win32 calls on the process's own stdout handle, into a zeroed out-struct.
        unsafe {
            let h = GetStdHandle(STD_OUTPUT_HANDLE);
            let mut info: CONSOLE_SCREEN_BUFFER_INFO = std::mem::zeroed();
            if GetConsoleScreenBufferInfo(h, &mut info) == 0 {
                return None;
            }
            Some((info.srWindow.Right - info.srWindow.Left + 1) as i64)
        }
    }
    #[cfg(unix)]
    {
        // SAFETY: TIOCGWINSZ fills the winsize struct for fd 1.
        unsafe {
            let mut ws: libc::winsize = std::mem::zeroed();
            if libc::ioctl(1, libc::TIOCGWINSZ, &mut ws) != 0 {
                return None;
            }
            Some(ws.ws_col as i64)
        }
    }
    #[cfg(not(any(windows, unix)))]
    {
        None
    }
}
