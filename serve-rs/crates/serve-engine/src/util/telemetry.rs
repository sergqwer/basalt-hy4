//! serve/util/telemetry.py: hardware readings for GET /metrics, sampled once a second on a background thread.
//! GPU readings come from NVML, loaded at run time; CPU, RAM and disk as psutil (a requirement of the server) reads
//! them, from Win32 / /proc. Unreadable values are None and never stop the server. (The module's no-psutil fallback
//! is not ported: psutil is always installed with the server.)
use crate::progress::Event;
use crate::pyfmt;
use serde_json::{Map, Value};
use std::collections::VecDeque;
use std::ffi::c_void;
use std::sync::{Arc, Mutex};

pub const HISTORY: usize = 60;

// ------------------------------------------------------------------------------------------------ NVML
/// NVML's clock event reasons that hold the clocks down (an idle card's low clocks and a set application clock are not).
pub const THROTTLE_REASONS: [(u64, &str); 6] = [(0x4, "power cap"), (0x8, "hardware slowdown"), (0x10, "sync boost"),
                                                (0x20, "thermal"), (0x40, "hardware thermal"), (0x80, "power brake")];

mod dl {
    use std::ffi::c_void;

    #[cfg(windows)]
    #[link(name = "kernel32")]
    extern "system" {
        fn LoadLibraryW(name: *const u16) -> *mut c_void;
        fn GetProcAddress(module: *mut c_void, name: *const u8) -> *mut c_void;
    }

    /// A loaded library's handle, or None.
    pub fn open(name: &str) -> Option<usize> {
        #[cfg(windows)]
        {
            let w: Vec<u16> = name.encode_utf16().chain(std::iter::once(0)).collect();
            // SAFETY: a NUL-terminated wide string.
            let h = unsafe { LoadLibraryW(w.as_ptr()) };
            (!h.is_null()).then_some(h as usize)
        }
        #[cfg(unix)]
        {
            let c = std::ffi::CString::new(name).ok()?;
            // SAFETY: a NUL-terminated name.
            let h = unsafe { libc::dlopen(c.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
            (!h.is_null()).then_some(h as usize)
        }
        #[cfg(not(any(windows, unix)))]
        {
            let _ = name;
            None
        }
    }

    /// A symbol of the library, or None.
    pub fn sym(lib: usize, name: &str) -> Option<*mut c_void> {
        let c = std::ffi::CString::new(name).ok()?;
        #[cfg(windows)]
        // SAFETY: `lib` came from LoadLibraryW and is never freed.
        let p = unsafe { GetProcAddress(lib as *mut c_void, c.as_ptr() as *const u8) };
        #[cfg(unix)]
        // SAFETY: `lib` came from dlopen and is never closed.
        let p = unsafe { libc::dlsym(lib as *mut c_void, c.as_ptr()) };
        #[cfg(not(any(windows, unix)))]
        let p: *mut c_void = { let _ = (lib, c); std::ptr::null_mut() };
        (!p.is_null()).then_some(p)
    }
}

static NVML_LIB: Mutex<Option<usize>> = Mutex::new(None);

fn nvml_lib() -> Option<usize> {
    let mut g = NVML_LIB.lock().unwrap_or_else(|e| e.into_inner());
    if g.is_none() {
        let mut names: Vec<String> = Vec::new();
        if cfg!(windows) {
            // Windows ships NVML as nvml.dll (System32; older drivers: NVSMI)
            let windir = std::env::var("WINDIR").unwrap_or_else(|_| r"C:\Windows".into());
            let sys = std::path::Path::new(&windir).join("System32").join("nvml.dll");
            if sys.exists() {
                names.push(sys.to_string_lossy().into_owned());
            }
            let smi = std::path::Path::new(r"C:\Program Files\NVIDIA Corporation\NVSMI\nvml.dll");
            if smi.exists() {
                names.push(smi.to_string_lossy().into_owned());
            }
            names.push("nvml.dll".into());
        } else {
            names = vec!["libnvidia-ml.so.1".into(), "libnvidia-ml.so".into()];
        }
        *g = names.iter().find_map(|n| dl::open(n));
    }
    *g
}

#[repr(C)]
#[derive(Default)]
struct NvUtil {
    gpu: u32,
    memory: u32,
}

#[repr(C)]
#[derive(Default)]
struct NvMem {
    total: u64,
    free: u64,
    used: u64,
}

#[repr(C)]
struct NvPci {
    bus_id_legacy: [u8; 16],
    domain: u32,
    bus: u32,
    device: u32,
    pci_device_id: u32,
    pci_subsystem_id: u32,
    bus_id: [u8; 32],
}

/// One compute process as NVML lists it (nvmlProcessInfo_t).
#[repr(C)]
#[derive(Default, Clone, Copy)]
pub struct NvProcInfo {
    pub pid: u32,
    pub used: u64,
    pub gpu_instance: u32,
    pub compute_instance: u32,
}

type F0 = unsafe extern "C" fn() -> i32;
type FDevPtr = unsafe extern "C" fn(*mut c_void, *mut c_void) -> i32;
type FDevUintPtr = unsafe extern "C" fn(*mut c_void, u32, *mut c_void) -> i32;
type FUintPtr = unsafe extern "C" fn(u32, *mut c_void) -> i32;
type FPtr = unsafe extern "C" fn(*mut c_void) -> i32;
type FBuf = unsafe extern "C" fn(*mut c_void, *mut u8, u32) -> i32;
type FSysBuf = unsafe extern "C" fn(*mut u8, u32) -> i32;
type FProcs = unsafe extern "C" fn(*mut c_void, *mut u32, *mut NvProcInfo) -> i32;

/// One card's NVML reader (Python's _Nvml): `ok()` once the library loaded, initialised and found card `index`.
#[derive(Debug, Clone, Copy)]
pub struct Nvml {
    lib: Option<usize>,
    dev: Option<usize>,
}

fn cstr(buf: &[u8]) -> String {
    let end = buf.iter().position(|b| *b == 0).unwrap_or(buf.len());
    String::from_utf8_lossy(&buf[..end]).into_owned()
}

impl Nvml {
    pub fn new(index: u32) -> Nvml {
        let none = Nvml { lib: None, dev: None };
        let Some(lib) = nvml_lib() else { return none };
        let Some(init) = dl::sym(lib, "nvmlInit_v2").or_else(|| dl::sym(lib, "nvmlInit")) else { return none };
        // SAFETY: NVML's documented signatures.
        unsafe {
            if std::mem::transmute::<*mut c_void, F0>(init)() != 0 {
                return none;
            }
            let Some(get) = dl::sym(lib, "nvmlDeviceGetHandleByIndex_v2").or_else(|| dl::sym(lib, "nvmlDeviceGetHandleByIndex"))
            else {
                return none;
            };
            let mut h: *mut c_void = std::ptr::null_mut();
            if std::mem::transmute::<*mut c_void, FUintPtr>(get)(index, &mut h as *mut _ as *mut c_void) != 0 {
                return none;
            }
            Nvml { lib: Some(lib), dev: Some(h as usize) }
        }
    }

    pub fn ok(&self) -> bool {
        self.lib.is_some() && self.dev.is_some()
    }

    fn f(&self, name: &str) -> Option<*mut c_void> {
        self.lib.and_then(|l| dl::sym(l, name))
    }

    fn dev(&self) -> *mut c_void {
        self.dev.unwrap_or(0) as *mut c_void
    }

    /// nvmlDeviceGetMemoryInfo: (total, free, used) bytes.
    pub fn memory_info(&self) -> Option<(u64, u64, u64)> {
        let f = self.f("nvmlDeviceGetMemoryInfo")?;
        let mut m = NvMem::default();
        // SAFETY: NVML's documented signature.
        let rc = unsafe { std::mem::transmute::<*mut c_void, FDevPtr>(f)(self.dev(), &mut m as *mut _ as *mut c_void) };
        (rc == 0).then_some((m.total, m.free, m.used))
    }

    /// The card's PCI address as "domain:bus:device.0" in hex, what the engine's facts name it by.
    pub fn bus_id(&self) -> Option<String> {
        let f = self.f("nvmlDeviceGetPciInfo_v3").or_else(|| self.f("nvmlDeviceGetPciInfo"))?;
        let mut p = NvPci { bus_id_legacy: [0; 16], domain: 0, bus: 0, device: 0, pci_device_id: 0, pci_subsystem_id: 0,
                            bus_id: [0; 32] };
        // SAFETY: NVML's documented signature.
        let rc = unsafe { std::mem::transmute::<*mut c_void, FDevPtr>(f)(self.dev(), &mut p as *mut _ as *mut c_void) };
        (rc == 0).then(|| format!("{:04x}:{:02x}:{:02x}.0", p.domain, p.bus, p.device))
    }

    fn ull(&self, name: &str) -> Option<u64> {
        let f = self.f(name)?;
        let mut v = 0u64;
        // SAFETY: NVML's documented signature.
        let rc = unsafe { std::mem::transmute::<*mut c_void, FDevPtr>(f)(self.dev(), &mut v as *mut _ as *mut c_void) };
        (rc == 0).then_some(v)
    }

    fn uint(&self, name: &str, arg: Option<u32>) -> Option<u32> {
        let f = self.f(name)?;
        let mut v = 0u32;
        let out = &mut v as *mut _ as *mut c_void;
        // SAFETY: NVML's documented signatures.
        let rc = unsafe {
            match arg {
                None => std::mem::transmute::<*mut c_void, FDevPtr>(f)(self.dev(), out),
                Some(a) => std::mem::transmute::<*mut c_void, FDevUintPtr>(f)(self.dev(), a, out),
            }
        };
        (rc == 0).then_some(v)
    }

    /// Why the clocks are held down now, in words ("" when nothing holds them; None when NVML cannot say).
    pub fn throttle(&self) -> Option<String> {
        let bits = self.ull("nvmlDeviceGetCurrentClocksEventReasons").or_else(|| self.ull("nvmlDeviceGetCurrentClocksThrottleReasons"))?;
        Some(THROTTLE_REASONS.iter().filter(|(b, _)| bits & b != 0).map(|(_, n)| *n).collect::<Vec<_>>().join(", "))
    }

    pub fn name(&self) -> Option<String> {
        let f = self.f("nvmlDeviceGetName")?;
        let mut buf = [0u8; 96];
        // SAFETY: NVML's documented signature; the buffer is 96 bytes.
        let rc = unsafe { std::mem::transmute::<*mut c_void, FBuf>(f)(self.dev(), buf.as_mut_ptr(), 96) };
        (rc == 0).then(|| cstr(&buf))
    }

    pub fn driver_version(&self) -> Option<String> {
        let f = self.f("nvmlSystemGetDriverVersion")?;
        let mut buf = [0u8; 96];
        // SAFETY: NVML's documented signature; the buffer is 96 bytes.
        let rc = unsafe { std::mem::transmute::<*mut c_void, FSysBuf>(f)(buf.as_mut_ptr(), 96) };
        (rc == 0).then(|| cstr(&buf))
    }

    /// The newest CUDA the driver supports, as "13.3".
    pub fn cuda_version(&self) -> Option<String> {
        for name in ["nvmlSystemGetCudaDriverVersion_v2", "nvmlSystemGetCudaDriverVersion"] {
            let Some(f) = self.f(name) else { continue };
            let mut v = 0i32;
            // SAFETY: NVML's documented signature.
            if unsafe { std::mem::transmute::<*mut c_void, FPtr>(f)(&mut v as *mut _ as *mut c_void) } == 0 {
                return Some(format!("{}.{}", v.div_euclid(1000), v.rem_euclid(1000).div_euclid(10)));
            }
        }
        None
    }

    /// nvmlDeviceGetComputeRunningProcesses_v3, else _v2, into a 64-entry array: Some(rows) on success, None when
    /// neither call is there or succeeds.
    pub fn compute_processes(&self) -> Option<Vec<NvProcInfo>> {
        for name in ["nvmlDeviceGetComputeRunningProcesses_v3", "nvmlDeviceGetComputeRunningProcesses_v2"] {
            let Some(f) = self.f(name) else { continue };
            let mut n = 64u32;
            let mut arr = [NvProcInfo::default(); 64];
            // SAFETY: NVML's documented signature; the array holds `n` entries.
            if unsafe { std::mem::transmute::<*mut c_void, FProcs>(f)(self.dev(), &mut n, arr.as_mut_ptr()) } == 0 {
                return Some(arr[..(n as usize).min(64)].to_vec());
            }
        }
        None
    }

    /// The card's readings, keys as GPU_KEYS (util and the memory pair only when read).
    pub fn read(&self) -> Map<String, Value> {
        let mut out = Map::new();
        if let Some(f) = self.f("nvmlDeviceGetUtilizationRates") {
            let mut u = NvUtil::default();
            // SAFETY: NVML's documented signature.
            if unsafe { std::mem::transmute::<*mut c_void, FDevPtr>(f)(self.dev(), &mut u as *mut _ as *mut c_void) } == 0 {
                out.insert("util".into(), u.gpu.into());
            }
        }
        if let Some((total, _, used)) = self.memory_info() {
            out.insert("mem_used".into(), used.into());
            out.insert("mem_total".into(), total.into());
        }
        let u = |v: Option<u32>| v.map(Value::from).unwrap_or(Value::Null);
        let fl = |v: Option<f64>| v.map(Value::from).unwrap_or(Value::Null);
        out.insert("temp".into(), u(self.uint("nvmlDeviceGetTemperature", Some(0))));       // NVML_TEMPERATURE_GPU
        out.insert("power".into(), fl(self.uint("nvmlDeviceGetPowerUsage", None).map(|mw| mw as f64 / 1000.0)));
        out.insert("power_limit".into(), fl(self.uint("nvmlDeviceGetEnforcedPowerLimit", None).map(|x| x as f64 / 1000.0)));
        out.insert("pcie_gen".into(), u(self.uint("nvmlDeviceGetCurrPcieLinkGeneration", None)));    // drops at idle
        out.insert("pcie_gen_max".into(), u(self.uint("nvmlDeviceGetMaxPcieLinkGeneration", None)));
        out.insert("pcie_width".into(), u(self.uint("nvmlDeviceGetCurrPcieLinkWidth", None)));
        let rx = self.uint("nvmlDeviceGetPcieThroughput", Some(1));       // NVML_PCIE_UTIL_RX_BYTES, KB/s
        let tx = self.uint("nvmlDeviceGetPcieThroughput", Some(0));
        out.insert("pcie_rx_mb".into(), fl(rx.map(|x| x as f64 / 1024.0)));
        out.insert("pcie_tx_mb".into(), fl(tx.map(|x| x as f64 / 1024.0)));
        out.insert("sm_clock".into(), u(self.uint("nvmlDeviceGetClockInfo", Some(1))));             // NVML_CLOCK_SM, MHz
        out.insert("mem_clock".into(), u(self.uint("nvmlDeviceGetClockInfo", Some(2))));            // NVML_CLOCK_MEM
        out.insert("throttle".into(), self.throttle().map(Value::from).unwrap_or(Value::Null));
        out
    }
}

/// What the sampler needs of a card (Python's duck-typed gpu_factory result); the optional calls default to None.
pub trait GpuReader: Send + Sync {
    fn ok(&self) -> bool;
    fn read(&self) -> Map<String, Value>;
    fn name(&self) -> Option<String>;
    fn bus_id(&self) -> Option<String> {
        None
    }
    fn driver_version(&self) -> Option<String> {
        None
    }
    fn cuda_version(&self) -> Option<String> {
        None
    }
}

impl GpuReader for Nvml {
    fn ok(&self) -> bool {
        Nvml::ok(self)
    }
    fn read(&self) -> Map<String, Value> {
        Nvml::read(self)
    }
    fn name(&self) -> Option<String> {
        Nvml::name(self)
    }
    fn bus_id(&self) -> Option<String> {
        Nvml::bus_id(self)
    }
    fn driver_version(&self) -> Option<String> {
        Nvml::driver_version(self)
    }
    fn cuda_version(&self) -> Option<String> {
        Nvml::cuda_version(self)
    }
}

/// The card's readings, from NVML.
pub fn gpu_reader(index: u32) -> Box<dyn GpuReader> {
    Box::new(Nvml::new(index))
}

// ------------------------------------------------------------------------------------------------ CPU / RAM / disk
#[cfg(windows)]
mod win {
    use std::ffi::c_void;

    #[repr(C)]
    #[derive(Default, Clone, Copy)]
    pub struct FileTime {
        pub low: u32,
        pub high: u32,
    }

    #[repr(C)]
    pub struct MemoryStatusEx {
        pub length: u32,
        pub memory_load: u32,
        pub total_phys: u64,
        pub avail_phys: u64,
        pub total_page_file: u64,
        pub avail_page_file: u64,
        pub total_virtual: u64,
        pub avail_virtual: u64,
        pub avail_extended_virtual: u64,
    }

    #[link(name = "kernel32")]
    extern "system" {
        pub fn GetSystemTimes(idle: *mut FileTime, kernel: *mut FileTime, user: *mut FileTime) -> i32;
        pub fn GlobalMemoryStatusEx(buf: *mut MemoryStatusEx) -> i32;
        pub fn GetLogicalProcessorInformationEx(relation: i32, buf: *mut u8, len: *mut u32) -> i32;
        pub fn GetActiveProcessorCount(group: u16) -> u32;
        pub fn GetModuleHandleW(name: *const u16) -> *mut c_void;
        pub fn GetProcAddress(module: *mut c_void, name: *const u8) -> *mut c_void;
        pub fn CreateFileW(name: *const u16, access: u32, share: u32, sec: *mut c_void, disposition: u32, flags: u32,
                           template: *mut c_void) -> *mut c_void;
        pub fn DeviceIoControl(dev: *mut c_void, code: u32, inb: *mut c_void, insz: u32, outb: *mut c_void, outsz: u32,
                               ret: *mut u32, ov: *mut c_void) -> i32;
        pub fn CloseHandle(h: *mut c_void) -> i32;
    }

    pub const HI_T: f64 = 429.4967296;
    pub const LO_T: f64 = 1e-7;

    pub fn secs(hi: i64, lo: u32) -> f64 {
        HI_T * hi as f64 + LO_T * lo as f64
    }
}

/// One cpu_times() reading as psutil names its fields, in its order.
#[derive(Debug, Clone)]
struct CpuTimes(Vec<f64>);

// psutil's field positions: Windows (user, system, idle, interrupt, dpc), Linux (user, nice, system, idle, iowait, irq,
// softirq, steal, guest, guest_nice)
#[cfg(windows)]
const IDLE: usize = 2;
#[cfg(not(windows))]
const IDLE: usize = 3;

/// psutil's _cpu_calc_percent: the busy share of the clamped per-field deltas, rounded to a tenth.
fn calc_percent(t1: &CpuTimes, t2: &CpuTimes) -> f64 {
    let d: Vec<f64> = t1.0.iter().zip(&t2.0).map(|(a, b)| if b - a < 0.0 { 0.0 } else { b - a }).collect();
    let field = |i: usize| d.get(i).copied().unwrap_or(0.0);
    let mut all = pyfmt::py_sum(d.iter().copied());
    if cfg!(not(windows)) {
        all -= field(8);
        all -= field(9);
    }
    let mut busy = all - field(IDLE);
    if cfg!(not(windows)) {
        busy -= field(4);
    }
    if all == 0.0 {
        return 0.0;
    }
    pyfmt::round_n(busy / all * 100.0, 1)
}

#[cfg(windows)]
fn per_cpu_times() -> Option<Vec<CpuTimes>> {
    type NtQsi = unsafe extern "system" fn(u32, *mut c_void, u32, *mut u32) -> i32;
    #[repr(C)]
    #[derive(Default, Clone, Copy)]
    struct Sppi {
        idle: i64,
        kernel: i64,
        user: i64,
        dpc: i64,
        interrupt: i64,
        count: u32,
    }
    // SAFETY: ntdll is always loaded; NtQuerySystemInformation(SystemProcessorPerformanceInformation) fills `n` entries.
    unsafe {
        let ntdll: Vec<u16> = "ntdll.dll".encode_utf16().chain(Some(0)).collect();
        let m = win::GetModuleHandleW(ntdll.as_ptr());
        let p = win::GetProcAddress(m, b"NtQuerySystemInformation\0".as_ptr());
        if p.is_null() {
            return None;
        }
        let n = win::GetActiveProcessorCount(0xffff).max(1) as usize;
        let mut buf = vec![Sppi::default(); n];
        let mut got = 0u32;
        let size = (n * std::mem::size_of::<Sppi>()) as u32;
        if std::mem::transmute::<*mut c_void, NtQsi>(p)(8, buf.as_mut_ptr() as *mut c_void, size, &mut got) != 0 {
            return None;
        }
        let k = (got as usize / std::mem::size_of::<Sppi>()).min(n);
        let s = |v: i64| win::secs(v >> 32, (v & 0xffff_ffff) as u32);
        Some(buf[..k].iter().map(|x| {
            let (user, idle, kernel, interrupt, dpc) = (s(x.user), s(x.idle), s(x.kernel), s(x.interrupt), s(x.dpc));
            CpuTimes(vec![user, kernel - idle, idle, interrupt, dpc])
        }).collect())
    }
}

#[cfg(windows)]
fn cpu_times() -> Option<CpuTimes> {
    let (mut i, mut k, mut u) = (win::FileTime::default(), win::FileTime::default(), win::FileTime::default());
    // SAFETY: three out-structs.
    if unsafe { win::GetSystemTimes(&mut i, &mut k, &mut u) } == 0 {
        return None;
    }
    let ft = |f: win::FileTime| win::secs(f.high as i64, f.low);
    let (idle, user, kernel) = (ft(i), ft(u), ft(k));
    let per = per_cpu_times()?;
    let interrupt = pyfmt::py_sum(per.iter().map(|t| t.0[3]));
    let dpc = pyfmt::py_sum(per.iter().map(|t| t.0[4]));
    Some(CpuTimes(vec![user, kernel - idle, idle, interrupt, dpc]))
}

#[cfg(unix)]
fn stat_lines() -> Option<Vec<(String, CpuTimes)>> {
    let text = std::fs::read_to_string("/proc/stat").ok()?;
    // SAFETY: sysconf has no preconditions.
    let ticks = unsafe { libc::sysconf(libc::_SC_CLK_TCK) } as f64;
    Some(text.lines().filter(|l| l.starts_with("cpu")).map(|l| {
        let f: Vec<&str> = l.split_whitespace().collect();
        let v = f[1..].iter().take(10).map(|x| x.parse::<f64>().unwrap_or(0.0) / ticks).collect();
        (f[0].to_string(), CpuTimes(v))
    }).collect())
}

#[cfg(unix)]
fn cpu_times() -> Option<CpuTimes> {
    stat_lines()?.into_iter().find(|(n, _)| n == "cpu").map(|(_, t)| t)
}

#[cfg(unix)]
fn per_cpu_times() -> Option<Vec<CpuTimes>> {
    Some(stat_lines()?.into_iter().filter(|(n, _)| n != "cpu").map(|(_, t)| t).collect())
}

#[cfg(not(any(windows, unix)))]
fn cpu_times() -> Option<CpuTimes> {
    None
}

#[cfg(not(any(windows, unix)))]
fn per_cpu_times() -> Option<Vec<CpuTimes>> {
    None
}

thread_local! {
    // psutil keeps the last reading per calling thread
    static LAST_CPU: std::cell::RefCell<Option<CpuTimes>> = const { std::cell::RefCell::new(None) };
    static LAST_PER_CPU: std::cell::RefCell<Option<Vec<CpuTimes>>> = const { std::cell::RefCell::new(None) };
}

/// psutil.cpu_percent(interval=None).
fn cpu_percent() -> Option<f64> {
    let t1 = LAST_CPU.with(|c| c.borrow().clone()).or_else(cpu_times)?;
    let t2 = cpu_times()?;
    LAST_CPU.with(|c| *c.borrow_mut() = Some(t2.clone()));
    Some(calc_percent(&t1, &t2))
}

/// psutil.cpu_percent(interval=None, percpu=True).
fn cpu_percent_per_cpu() -> Option<Vec<f64>> {
    let t1 = LAST_PER_CPU.with(|c| c.borrow().clone()).or_else(per_cpu_times)?;
    let t2 = per_cpu_times()?;
    LAST_PER_CPU.with(|c| *c.borrow_mut() = Some(t2.clone()));
    Some(t1.iter().zip(&t2).map(|(a, b)| calc_percent(a, b)).collect())
}

/// psutil.virtual_memory(): (total, available, cached: Linux only).
fn virtual_memory() -> Option<(u64, u64, Option<u64>)> {
    #[cfg(windows)]
    {
        let mut m: win::MemoryStatusEx = unsafe { std::mem::zeroed() };
        m.length = std::mem::size_of::<win::MemoryStatusEx>() as u32;
        // SAFETY: an out-struct with its length set.
        if unsafe { win::GlobalMemoryStatusEx(&mut m) } == 0 {
            return None;
        }
        Some((m.total_phys, m.avail_phys, None))
    }
    #[cfg(unix)]
    {
        let text = std::fs::read_to_string("/proc/meminfo").ok()?;
        let get = |k: &str| -> Option<u64> {
            text.lines().find_map(|l| l.strip_prefix(k)).and_then(|r| r.split_whitespace().next()?.parse::<u64>().ok())
                .map(|v| v * 1024)
        };
        let total = get("MemTotal:")?;
        let avail = get("MemAvailable:").unwrap_or_else(|| get("MemFree:").unwrap_or(0) + get("Cached:").unwrap_or(0));
        let cached = get("Cached:").unwrap_or(0) + get("SReclaimable:").unwrap_or(0);
        Some((total, avail, Some(cached)))
    }
    #[cfg(not(any(windows, unix)))]
    {
        None
    }
}

/// psutil.disk_io_counters().read_bytes: the bytes read from every physical disk.
fn disk_read_bytes() -> Option<u64> {
    #[cfg(windows)]
    {
        #[repr(C)]
        struct DiskPerformance {
            bytes_read: i64,
            bytes_written: i64,
            read_time: i64,
            write_time: i64,
            idle_time: i64,
            read_count: u32,
            write_count: u32,
            queue_depth: u32,
            split_count: u32,
            query_time: i64,
            storage_device_number: u32,
            storage_manager_name: [u16; 8],
        }
        const IOCTL_DISK_PERFORMANCE: u32 = 0x70020;
        let (mut total, mut any) = (0u64, false);
        for n in 0..=32 {
            let name: Vec<u16> = format!(r"\\.\PhysicalDrive{n}").encode_utf16().chain(Some(0)).collect();
            // SAFETY: CreateFileW with no access opens the device for queries; the handle is closed below.
            unsafe {
                let h = win::CreateFileW(name.as_ptr(), 0, 3, std::ptr::null_mut(), 3, 0, std::ptr::null_mut());
                if h as isize == -1 {
                    continue;
                }
                let mut p: DiskPerformance = std::mem::zeroed();
                let mut ret = 0u32;
                let ok = win::DeviceIoControl(h, IOCTL_DISK_PERFORMANCE, std::ptr::null_mut(), 0,
                                              &mut p as *mut _ as *mut c_void, std::mem::size_of::<DiskPerformance>() as u32,
                                              &mut ret, std::ptr::null_mut());
                win::CloseHandle(h);
                if ok != 0 {
                    total = total.wrapping_add(p.bytes_read as u64);
                    any = true;
                }
            }
        }
        any.then_some(total)
    }
    #[cfg(unix)]
    {
        let text = std::fs::read_to_string("/proc/diskstats").ok()?;
        let mut total = 0u64;
        for l in text.lines() {
            let f: Vec<&str> = l.split_whitespace().collect();
            if f.len() < 14 {
                continue;
            }
            let name = f[2].replace('/', "!");
            if !std::path::Path::new(&format!("/sys/block/{name}")).exists() {
                continue;                       // partitions: their disk counts them
            }
            total += f[5].parse::<u64>().unwrap_or(0) * 512;
        }
        Some(total)
    }
    #[cfg(not(any(windows, unix)))]
    {
        None
    }
}

/// psutil.cpu_count(logical=False), None when unknown.
fn cpu_cores() -> Option<u64> {
    #[cfg(windows)]
    {
        let mut len = 0u32;
        // SAFETY: the first call sizes the buffer, the second fills it; entries are walked by their own Size field.
        unsafe {
            win::GetLogicalProcessorInformationEx(0, std::ptr::null_mut(), &mut len);
            if len == 0 {
                return None;
            }
            let mut buf = vec![0u8; len as usize];
            if win::GetLogicalProcessorInformationEx(0, buf.as_mut_ptr(), &mut len) == 0 {
                return None;
            }
            let (mut off, mut n) = (0usize, 0u64);
            while off + 8 <= len as usize {
                let rel = u32::from_ne_bytes(buf[off..off + 4].try_into().unwrap());
                let size = u32::from_ne_bytes(buf[off + 4..off + 8].try_into().unwrap()) as usize;
                if rel == 0 {
                    n += 1;
                }
                if size == 0 {
                    break;
                }
                off += size;
            }
            (n > 0).then_some(n)
        }
    }
    #[cfg(unix)]
    {
        let mut set = std::collections::BTreeSet::new();
        for leaf in ["core_cpus_list", "thread_siblings_list"] {
            if let Ok(rd) = std::fs::read_dir("/sys/devices/system/cpu") {
                for e in rd.flatten() {
                    let n = e.file_name().to_string_lossy().into_owned();
                    if n.len() > 3 && n.starts_with("cpu") && n[3..].bytes().all(|b| b.is_ascii_digit()) {
                        if let Ok(s) = std::fs::read_to_string(e.path().join("topology").join(leaf)) {
                            set.insert(s.trim().to_string());
                        }
                    }
                }
            }
            if !set.is_empty() {
                return Some(set.len() as u64);
            }
        }
        let text = std::fs::read_to_string("/proc/cpuinfo").ok()?;
        let (mut pairs, mut phys) = (std::collections::BTreeSet::new(), None::<String>);
        for l in text.lines() {
            if let Some(v) = l.strip_prefix("physical id") {
                phys = v.split(':').nth(1).map(|x| x.trim().to_string());
            } else if let Some(v) = l.strip_prefix("core id") {
                pairs.insert((phys.clone(), v.split(':').nth(1).map(|x| x.trim().to_string())));
            }
        }
        (!pairs.is_empty()).then_some(pairs.len() as u64)
    }
    #[cfg(not(any(windows, unix)))]
    {
        None
    }
}

/// os.cpu_count().
pub fn os_cpu_count() -> Option<u64> {
    #[cfg(windows)]
    {
        // SAFETY: no preconditions.
        let n = unsafe { win::GetActiveProcessorCount(0xffff) };
        (n > 0).then_some(n as u64)
    }
    #[cfg(unix)]
    {
        // SAFETY: no preconditions.
        let n = unsafe { libc::sysconf(libc::_SC_NPROCESSORS_ONLN) };
        (n > 0).then_some(n as u64)
    }
    #[cfg(not(any(windows, unix)))]
    {
        None
    }
}

/// The CPU's name: /proc/cpuinfo's model name, else platform.processor(), None when empty.
pub fn cpu_name() -> Option<String> {
    if let Ok(text) = std::fs::read("/proc/cpuinfo") {
        for line in String::from_utf8_lossy(&text).split_inclusive('\n') {
            if line.starts_with("model name") {
                return Some(pyfmt::strip(line.split_once(':').map(|x| x.1).unwrap_or("")).to_string());
            }
        }
    }
    let p = if cfg!(windows) {
        std::env::var("PROCESSOR_IDENTIFIER").ok()
            .or_else(|| std::env::var("PROCESSOR_ARCHITEW6432").ok())
            .or_else(|| std::env::var("PROCESSOR_ARCHITECTURE").ok())
            .unwrap_or_default()
    } else {
        std::process::Command::new("uname").arg("-p").output().ok()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .filter(|s| s != "unknown")
            .unwrap_or_default()
    };
    (!p.is_empty()).then_some(p)
}

// ------------------------------------------------------------------------------------------------ the sampler
pub const GPU_KEYS: [&str; 14] = ["util", "mem_used", "mem_total", "temp", "power", "power_limit", "pcie_gen",
                                  "pcie_gen_max", "pcie_width", "pcie_rx_mb", "pcie_tx_mb", "sm_clock", "mem_clock",
                                  "throttle"];
pub const SERIES: [&str; 3] = ["tok_s", "prefill_tok_s_mean", "disk_read_mb"];     // series kept as history
pub const GPU_SERIES: [&str; 3] = ["util", "power", "temp"];                        // per-card series kept as history

fn r2(v: Option<&Value>) -> Value {
    match v {
        Some(Value::Number(n)) if n.is_f64() => Value::from(pyfmt::round_n(n.as_f64().unwrap_or(0.0), 2)),
        Some(v) => v.clone(),
        None => Value::Null,
    }
}

fn push(q: &mut VecDeque<Value>, v: Value) {
    if q.len() == HISTORY {
        q.pop_front();
    }
    q.push_back(v);
}

/// The server's own speeds, merged into each sample; None stands for Python's swallowed exception.
pub type Extra = Box<dyn Fn() -> Option<Map<String, Value>> + Send + Sync>;
pub type GpuFactory = dyn Fn(u32) -> Box<dyn GpuReader>;

struct Recorded {
    now: Map<String, Value>,
    hist: Vec<VecDeque<Value>>,                         // by SERIES
    slot_hist: VecDeque<Vec<Value>>,                    // per sample: each slot's tok/s
    gpu_hist: Vec<(u32, Vec<VecDeque<Value>>)>,         // by GPU_SERIES
}

struct Inner {
    extra: Option<Extra>,
    rec: Mutex<Recorded>,
    gpus: Vec<(u32, Box<dyn GpuReader>)>,
    static_: Map<String, Value>,
    disk_prev: Mutex<Option<(f64, u64)>>,
    stopped: Event,
}

/// The sampler: a thread records one sample a second until stop().
#[derive(Clone)]
pub struct Telemetry {
    inner: Arc<Inner>,
}

impl Telemetry {
    /// `extra()` returns the server's own speeds; `gpu_indices` the engine's cards (NVML numbering; None or empty: [0]);
    /// `gpu_factory(index)` a reader (None: NVML).
    pub fn new(extra: Option<Extra>, gpu_indices: Option<&[u32]>, gpu_factory: Option<&GpuFactory>) -> Telemetry {
        let t = Self::unstarted(extra, gpu_indices, gpu_factory);
        let inner = t.inner.clone();
        std::thread::spawn(move || {
            while !inner.stopped.is_set() {
                let s = Self::sample_of(&inner);
                Self::record_into(&inner, s);
                inner.stopped.wait(1.0);
            }
        });
        t
    }

    /// The sampler without its thread (tests drive sample() and record()).
    pub fn unstarted(extra: Option<Extra>, gpu_indices: Option<&[u32]>, gpu_factory: Option<&GpuFactory>) -> Telemetry {
        let mut idx: Vec<u32> = Vec::new();
        let given = gpu_indices.filter(|g| !g.is_empty()).map(|g| g.to_vec()).unwrap_or_else(|| vec![0]);
        for i in given {
            if !idx.contains(&i) {
                idx.push(i);
            }
        }
        let gpus: Vec<(u32, Box<dyn GpuReader>)> = idx.into_iter()
            .map(|i| (i, gpu_factory.map(|f| f(i)).unwrap_or_else(|| gpu_reader(i))))
            .filter(|(_, g)| g.ok())
            .collect();
        let mut now = Map::new();
        now.insert("gpus".into(), Value::Array(vec![]));
        for k in ["cpu", "cpu_cores", "ram_used", "ram_total", "ram_cached", "disk_read_mb"] {
            now.insert(k.into(), Value::Null);
        }
        let opt = |v: Option<String>| v.map(Value::from).unwrap_or(Value::Null);
        let mut st = Map::new();
        st.insert("gpus".into(), Value::Array(gpus.iter().map(|(i, g)| {
            let mut m = Map::new();
            m.insert("index".into(), (*i).into());
            m.insert("name".into(), opt(g.name()));
            m.insert("bus".into(), opt(g.bus_id()));
            Value::Object(m)
        }).collect()));
        st.insert("cpu_name".into(), opt(cpu_name()));
        st.insert("cores".into(), cpu_cores().filter(|n| *n > 0).map(Value::from).unwrap_or(Value::Null));
        st.insert("threads".into(), os_cpu_count().map(Value::from).unwrap_or(Value::Null));
        let first = gpus.first().map(|g| &g.1);
        st.insert("driver".into(), opt(first.and_then(|g| g.driver_version())));
        st.insert("cuda".into(), opt(first.and_then(|g| g.cuda_version())));
        let rec = Recorded {
            now,
            hist: SERIES.iter().map(|_| VecDeque::new()).collect(),
            slot_hist: VecDeque::new(),
            gpu_hist: gpus.iter().map(|(i, _)| (*i, GPU_SERIES.iter().map(|_| VecDeque::new()).collect())).collect(),
        };
        Telemetry { inner: Arc::new(Inner { extra, rec: Mutex::new(rec), gpus, static_: st, disk_prev: Mutex::new(None),
                                            stopped: Event::new() }) }
    }

    /// End the sampling thread; the readings taken so far stay readable.
    pub fn stop(&self) {
        self.inner.stopped.set();
    }

    pub fn stopped(&self) -> bool {
        self.inner.stopped.is_set()
    }

    fn disk(inner: &Inner) -> Option<f64> {
        let c = disk_read_bytes()?;
        let t = crate::util::logbuf::now_s();
        let mut prev = inner.disk_prev.lock().unwrap();
        let p = prev.replace((t, c));
        let (pt, pc) = p?;
        if t <= pt {
            return None;
        }
        Some((c as f64 - pc as f64) / (t - pt) / (1u64 << 20) as f64)
    }

    /// One reading of every card, the CPU, RAM and disk, and the server's own figures.
    pub fn sample(&self) -> Map<String, Value> {
        Self::sample_of(&self.inner)
    }

    fn sample_of(inner: &Inner) -> Map<String, Value> {
        let mut s = Map::new();
        let gpus: Vec<Value> = inner.gpus.iter().map(|(i, g)| {
            let r = g.read();
            let mut m = Map::new();
            m.insert("index".into(), (*i).into());
            for k in GPU_KEYS {
                m.insert(k.into(), r.get(k).cloned().unwrap_or(Value::Null));
            }
            Value::Object(m)
        }).collect();
        s.insert("gpus".into(), Value::Array(gpus));
        for k in ["cpu", "ram_used", "ram_total", "cpu_cores", "ram_cached"] {
            s.insert(k.into(), Value::Null);
        }
        if let Some(c) = cpu_percent() {
            s.insert("cpu".into(), c.into());
            if let Some(per) = cpu_percent_per_cpu() {
                s.insert("cpu_cores".into(), per.into_iter().map(Value::from).collect());      // by logical processor
                if let Some((total, avail, cached)) = virtual_memory() {
                    s.insert("ram_used".into(), (total - avail.min(total)).into());
                    s.insert("ram_total".into(), total.into());
                    s.insert("ram_cached".into(), cached.map(Value::from).unwrap_or(Value::Null));     // the page cache (Linux)
                }
            }
        }
        s.insert("disk_read_mb".into(), Self::disk(inner).map(Value::from).unwrap_or(Value::Null));
        if let Some(f) = &inner.extra {
            if let Some(e) = f() {
                for (k, v) in e {
                    s.insert(k, v);
                }
            }
        }
        s
    }

    /// Make sample `s` the current reading and add it to the history.
    pub fn record(&self, s: Map<String, Value>) {
        Self::record_into(&self.inner, s)
    }

    fn record_into(inner: &Inner, s: Map<String, Value>) {
        let mut r = inner.rec.lock().unwrap();
        for (i, k) in SERIES.iter().enumerate() {
            push(&mut r.hist[i], r2(s.get(*k)));
        }
        if s.contains_key("slot_tok_s") {
            let row = match s.get("slot_tok_s") {
                Some(Value::Array(a)) => a.iter().map(|v| r2(Some(v))).collect(),
                _ => Vec::new(),
            };
            if r.slot_hist.len() == HISTORY {
                r.slot_hist.pop_front();
            }
            r.slot_hist.push_back(row);
        }
        if let Some(Value::Array(gs)) = s.get("gpus") {
            for g in gs {
                let idx = g.get("index").and_then(Value::as_u64);
                if let Some((_, h)) = r.gpu_hist.iter_mut().find(|(i, _)| Some(*i as u64) == idx) {
                    for (j, k) in GPU_SERIES.iter().enumerate() {
                        push(&mut h[j], r2(g.get(*k)));
                    }
                }
            }
        }
        r.now = s;
    }

    /// {"now", "history", "static"} as GET /metrics shows them.
    pub fn snapshot(&self) -> Map<String, Value> {
        let r = self.inner.rec.lock().unwrap();
        let mut now: Map<String, Value> = r.now.iter()
            .filter(|(k, _)| !SERIES.contains(&k.as_str()) && k.as_str() != "slot_tok_s")
            .map(|(k, v)| (k.clone(), v.clone())).collect();
        now.insert("disk_read_mb".into(), r.now.get("disk_read_mb").cloned().unwrap_or(Value::Null));
        let mut hist = Map::new();
        for (i, k) in SERIES.iter().enumerate() {
            if *k != "disk_read_mb" {
                hist.insert((*k).into(), Value::Array(r.hist[i].iter().cloned().collect()));
            }
        }
        now.insert("disk_hist".into(), Value::Array(r.hist[2].iter().cloned().collect()));   // the read rate's last 60 s
        // one series per slot, oldest first; a sample from before a slot existed is None
        let n = r.slot_hist.iter().map(Vec::len).max().unwrap_or(0);
        hist.insert("slot_tok_s".into(), Value::Array((0..n).map(|j| {
            Value::Array(r.slot_hist.iter().map(|x| x.get(j).cloned().unwrap_or(Value::Null)).collect())
        }).collect()));
        hist.insert("gpus".into(), Value::Array(r.gpu_hist.iter().map(|(i, h)| {
            let mut m = Map::new();
            m.insert("index".into(), (*i).into());
            for (j, k) in GPU_SERIES.iter().enumerate() {
                m.insert((*k).into(), Value::Array(h[j].iter().cloned().collect()));
            }
            Value::Object(m)
        }).collect()));
        let mut out = Map::new();
        out.insert("now".into(), Value::Object(now));
        out.insert("history".into(), Value::Object(hist));
        out.insert("static".into(), Value::Object(self.inner.static_.clone()));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    struct Fake(u32);
    impl GpuReader for Fake {
        fn ok(&self) -> bool {
            self.0 != 9
        }
        fn read(&self) -> Map<String, Value> {
            json!({"util": 50, "power": 123.456, "temp": 60}).as_object().unwrap().clone()
        }
        fn name(&self) -> Option<String> {
            Some(format!("card{}", self.0))
        }
    }

    #[test]
    fn samples_history_and_shapes() {
        let factory = |i: u32| -> Box<dyn GpuReader> { Box::new(Fake(i)) };
        let extra: Extra = Box::new(|| Some(json!({"tok_s": 12.345, "slot_tok_s": [1.234, 2.0]}).as_object().unwrap().clone()));
        let t = Telemetry::unstarted(Some(extra), Some(&[1, 9, 1, 0]), Some(&factory));
        let snap = t.snapshot();
        assert_eq!(snap["now"].as_object().unwrap().keys().collect::<Vec<_>>(),
                   ["gpus", "cpu", "cpu_cores", "ram_used", "ram_total", "ram_cached", "disk_read_mb", "disk_hist"]);
        assert_eq!(snap["static"]["gpus"], json!([{"index": 1, "name": "card1", "bus": null}, {"index": 0, "name": "card0", "bus": null}]));
        assert_eq!(snap["static"].as_object().unwrap().keys().collect::<Vec<_>>(),
                   ["gpus", "cpu_name", "cores", "threads", "driver", "cuda"]);
        let s = t.sample();
        assert_eq!(s.keys().collect::<Vec<_>>(),
                   ["gpus", "cpu", "ram_used", "ram_total", "cpu_cores", "ram_cached", "disk_read_mb", "tok_s", "slot_tok_s"]);
        assert_eq!(s["gpus"][0].as_object().unwrap().keys().count(), 1 + GPU_KEYS.len());
        assert_eq!(s["gpus"][0]["mem_used"], Value::Null);
        t.record(s.clone());
        let mut s2 = s;
        s2.insert("slot_tok_s".into(), json!([3.0]));
        t.record(s2);
        let snap = t.snapshot();
        assert_eq!(snap["history"]["tok_s"], json!([12.35, 12.35]));
        assert_eq!(snap["history"]["slot_tok_s"], json!([[1.23, 3.0], [2.0, null]]));
        assert_eq!(snap["history"]["gpus"][0], json!({"index": 1, "util": [50, 50], "power": [123.46, 123.46], "temp": [60, 60]}));
        assert_eq!(snap["history"].as_object().unwrap().keys().collect::<Vec<_>>(), ["tok_s", "prefill_tok_s_mean", "slot_tok_s", "gpus"]);
        let now = snap["now"].as_object().unwrap();
        assert!(!now.contains_key("tok_s") && !now.contains_key("slot_tok_s"));
        assert_eq!(now.keys().rev().take(2).collect::<Vec<_>>(), ["disk_hist", "disk_read_mb"]);
    }

    #[test]
    fn host_readings() {
        let s = Telemetry::unstarted(None, None, Some(&|_| -> Box<dyn GpuReader> { Box::new(Fake(9)) })).sample();
        assert!(s["cpu"].is_f64());
        assert!(s["ram_total"].as_u64().unwrap() > 0);
        assert!(s["cpu_cores"].as_array().unwrap().len() as u64 >= 1);
        assert!(os_cpu_count().unwrap() >= cpu_cores().unwrap_or(1));
        assert!(cpu_name().is_some());
    }

    #[test]
    fn cpu_percent_math() {
        let a = CpuTimes(vec![10.0, 5.0, 85.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let b = CpuTimes(vec![20.0, 10.0, 170.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let p = calc_percent(&a, &b);
        assert!(p > 0.0 && p < 100.0);
        assert_eq!(calc_percent(&a, &a), 0.0);
    }
}

#[cfg(test)]
mod live_nvml {
    /// Prints card 0's readings (run by hand: `cargo test -p serve-engine nvml_card0 -- --ignored --nocapture`).
    #[test]
    #[ignore]
    fn nvml_card0() {
        let n = super::Nvml::new(0);
        println!("ok={} name={:?} bus={:?} drv={:?} cuda={:?}", n.ok(), n.name(), n.bus_id(), n.driver_version(), n.cuda_version());
        println!("{}", serde_json::Value::Object(n.read()));
        println!("procs={:?}", super::super::vram::processes(0));
    }
}
