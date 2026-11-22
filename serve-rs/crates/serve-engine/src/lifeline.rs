//! serve/engine/lifeline.py: start the engine so that it ends with the server, however the server ends.
//! Windows: a kill-on-close job object (the only handle is ours) and its own process group; Linux:
//! PR_SET_PDEATHSIG = SIGKILL set by one long-lived anchor thread, and its own process group.
use std::io;
use std::process::{Child, Command};

#[cfg(windows)]
mod win {
    use std::os::windows::io::AsRawHandle;
    use std::sync::Mutex;
    use windows_sys::Win32::Foundation::HANDLE;
    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation, SetInformationJobObject,
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
    };

    struct Job(HANDLE);
    unsafe impl Send for Job {}
    static JOB: Mutex<Option<Job>> = Mutex::new(None);

    pub fn contain(child: &std::process::Child) -> bool {
        unsafe {
            let mut g = JOB.lock().unwrap();
            if g.is_none() {
                let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
                if job.is_null() {
                    return false;
                }
                let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
                info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
                if SetInformationJobObject(
                    job,
                    JobObjectExtendedLimitInformation,
                    &info as *const _ as *const _,
                    std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
                ) == 0
                {
                    return false;
                }
                *g = Some(Job(job));
            }
            let job = g.as_ref().unwrap().0;
            AssignProcessToJobObject(job, child.as_raw_handle() as HANDLE) != 0
        }
    }
}

pub const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
pub const CREATE_NO_WINDOW: u32 = 0x0800_0000;

#[cfg(target_os = "linux")]
mod anchor {
    use std::sync::mpsc::{channel, Sender};
    use std::sync::{Mutex, OnceLock};

    type Job = Box<dyn FnOnce() + Send>;
    static TX: OnceLock<Mutex<Sender<Job>>> = OnceLock::new();

    /// Run `f` on the anchor thread (the thread every engine is forked from) and wait for it.
    pub fn call<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
        let tx = TX.get_or_init(|| {
            let (tx, rx) = channel::<Job>();
            std::thread::Builder::new()
                .name("engine-anchor".into())
                .spawn(move || {
                    for job in rx {
                        job();
                    }
                })
                .expect("anchor thread");
            Mutex::new(tx)
        });
        let (rtx, rrx) = std::sync::mpsc::channel();
        tx.lock().unwrap().send(Box::new(move || {
            let _ = rtx.send(f());
        })).expect("anchor");
        rrx.recv().expect("anchor result")
    }
}

/// `cmd.spawn()` for a process that must not outlive the server.
pub fn spawn(mut cmd: Command) -> io::Result<Child> {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW);
        let child = cmd.spawn()?;
        win::contain(&child);
        Ok(child)
    }
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::process::CommandExt;
        let parent = std::process::id() as libc::pid_t;
        unsafe {
            cmd.pre_exec(move || {
                libc::setpgid(0, 0);
                libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL as libc::c_ulong, 0, 0, 0);
                if libc::getppid() != parent {
                    libc::kill(libc::getpid(), libc::SIGKILL);
                }
                Ok(())
            });
        }
        anchor::call(move || cmd.spawn())
    }
    #[cfg(all(unix, not(target_os = "linux")))]
    {
        use std::os::unix::process::CommandExt;
        unsafe {
            cmd.pre_exec(|| {
                libc::setpgid(0, 0);
                Ok(())
            });
        }
        cmd.spawn()
    }
}
