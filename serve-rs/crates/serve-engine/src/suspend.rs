//! serve/engine/suspend.py: suspending a running engine's GPU state with the driver's process checkpoint
//! (cuCheckpointProcess*). The driver offers this on Linux only; elsewhere `available()` is false.

#[cfg(target_os = "linux")]
mod imp {
    use std::ffi::CString;
    use std::sync::OnceLock;

    type Fn2 = unsafe extern "C" fn(libc::c_int, *mut libc::c_void) -> libc::c_int;
    type Init = unsafe extern "C" fn(libc::c_uint) -> libc::c_int;

    pub struct Cu {
        pub lock: Fn2,
        pub checkpoint: Fn2,
        pub restore: Fn2,
        pub unlock: Fn2,
    }

    static CU: OnceLock<Option<Cu>> = OnceLock::new();

    unsafe fn sym<T: Copy>(h: *mut libc::c_void, name: &str) -> Option<T> {
        let c = CString::new(name).ok()?;
        let p = libc::dlsym(h, c.as_ptr());
        if p.is_null() {
            None
        } else {
            Some(std::mem::transmute_copy::<*mut libc::c_void, T>(&p))
        }
    }

    pub fn driver() -> Option<&'static Cu> {
        CU.get_or_init(|| unsafe {
            let name = CString::new("libcuda.so.1").ok()?;
            let h = libc::dlopen(name.as_ptr(), libc::RTLD_NOW);
            if h.is_null() {
                return None;
            }
            let cu = Cu {
                lock: sym(h, "cuCheckpointProcessLock")?,
                checkpoint: sym(h, "cuCheckpointProcessCheckpoint")?,
                restore: sym(h, "cuCheckpointProcessRestore")?,
                unlock: sym(h, "cuCheckpointProcessUnlock")?,
            };
            let init: Init = sym(h, "cuInit")?;
            if init(0) != 0 {
                return None;
            }
            Some(cu)
        })
        .as_ref()
    }
}

#[cfg(target_os = "linux")]
pub fn available() -> bool {
    imp::driver().is_some()
}

#[cfg(not(target_os = "linux"))]
pub fn available() -> bool {
    false
}

/// Lock `pid` and move its GPU state to host memory; None on success, else why not.
pub fn suspend(pid: u32) -> Option<String> {
    #[cfg(target_os = "linux")]
    unsafe {
        let Some(cu) = imp::driver() else { return Some("the driver cannot suspend a process here".into()) };
        let p = pid as libc::c_int;
        let rc = (cu.lock)(p, std::ptr::null_mut());
        if rc != 0 {
            return Some(format!("the driver could not lock the engine (CUresult {rc})"));
        }
        let rc = (cu.checkpoint)(p, std::ptr::null_mut());
        if rc != 0 {
            (cu.unlock)(p, std::ptr::null_mut());
            return Some(format!("the driver could not save the engine's GPU state (CUresult {rc})"));
        }
        None
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = pid;
        Some("the driver cannot suspend a process here".into())
    }
}

/// Put `pid`'s GPU state back and unlock it; None on success, else why not.
pub fn resume(pid: u32) -> Option<String> {
    #[cfg(target_os = "linux")]
    unsafe {
        let Some(cu) = imp::driver() else { return Some("the driver cannot resume a process here".into()) };
        let p = pid as libc::c_int;
        let rc = (cu.restore)(p, std::ptr::null_mut());
        if rc != 0 {
            return Some(format!("the driver could not restore the engine's GPU state (CUresult {rc})"));
        }
        let rc = (cu.unlock)(p, std::ptr::null_mut());
        if rc != 0 {
            return Some(format!("the driver could not unlock the engine (CUresult {rc})"));
        }
        None
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = pid;
        Some("the driver cannot resume a process here".into())
    }
}
