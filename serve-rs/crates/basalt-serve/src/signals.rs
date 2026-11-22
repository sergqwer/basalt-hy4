//! Ctrl+C (and SIGTERM, which server.py routes to Ctrl+C's path): a counter the main loop polls.
use std::sync::atomic::{AtomicU32, Ordering};

static COUNT: AtomicU32 = AtomicU32::new(0);

pub fn count() -> u32 {
    COUNT.load(Ordering::SeqCst)
}

#[cfg(windows)]
pub fn install() {
    type Handler = unsafe extern "system" fn(u32) -> i32;
    #[link(name = "kernel32")]
    extern "system" {
        fn SetConsoleCtrlHandler(h: Option<Handler>, add: i32) -> i32;
    }
    unsafe extern "system" fn on_ctrl(kind: u32) -> i32 {
        // CTRL_C_EVENT = 0: Python's KeyboardInterrupt; CTRL_BREAK and the rest take the default (the process ends)
        if kind == 0 {
            COUNT.fetch_add(1, Ordering::SeqCst);
            1
        } else {
            0
        }
    }
    unsafe {
        SetConsoleCtrlHandler(Some(on_ctrl), 1);
    }
}

#[cfg(unix)]
pub fn install() {
    extern "C" fn on_int(_: libc::c_int) {
        COUNT.fetch_add(1, Ordering::SeqCst);
    }
    extern "C" fn on_term(_: libc::c_int) {
        // one stop: a later TERM must not cut the engine's QUIT short
        unsafe {
            libc::signal(libc::SIGTERM, libc::SIG_IGN);
        }
        COUNT.fetch_add(1, Ordering::SeqCst);
    }
    unsafe {
        libc::signal(libc::SIGINT, on_int as libc::sighandler_t);
        libc::signal(libc::SIGTERM, on_term as libc::sighandler_t);
    }
}
