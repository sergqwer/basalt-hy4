//! serve/util/ram.py: the machine's available RAM: GlobalMemoryStatusEx on Windows, MemAvailable elsewhere.

/// Physical RAM the OS could hand out now, in bytes; None where it gives no reading.
pub fn available_bytes() -> Option<u64> {
    #[cfg(windows)]
    {
        use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
        // SAFETY: a zeroed out-struct with its length set, as the call requires.
        unsafe {
            let mut st: MEMORYSTATUSEX = std::mem::zeroed();
            st.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
            if GlobalMemoryStatusEx(&mut st) != 0 {
                return Some(st.ullAvailPhys);
            }
        }
        None
    }
    #[cfg(not(windows))]
    {
        if let Ok(text) = std::fs::read_to_string("/proc/meminfo") {
            for line in text.lines() {
                if let Some(rest) = line.strip_prefix("MemAvailable:") {
                    if let Some(kb) = rest.split_whitespace().next().and_then(|v| v.parse::<u64>().ok()) {
                        return Some(kb * 1024);
                    }
                    break;
                }
            }
        }
        // SAFETY: sysconf has no preconditions.
        let (page, avail) = unsafe { (libc::sysconf(libc::_SC_PAGESIZE), libc::sysconf(libc::_SC_AVPHYS_PAGES)) };
        if page > 0 && avail >= 0 {
            return Some(page as u64 * avail as u64);
        }
        None
    }
}
