//! serve/util/vram.py: who holds the GPUs' memory: per-process figures from NVML, for the sleep and wake checks.
use super::telemetry::Nvml;

/// NVML's answer for a process whose memory it cannot read (Windows, containers).
const NOT_AVAILABLE: u64 = u64::MAX;

/// The first words of a process's command line, else its pid alone.
fn command(pid: u32) -> String {
    match std::fs::read(format!("/proc/{pid}/cmdline")) {
        Ok(b) => {
            let text = String::from_utf8_lossy(&b).into_owned();
            let words: Vec<&str> = text.split('\0').take(3).filter(|w| !w.is_empty()).collect();
            let s = crate::pyfmt::head_chars(&words.join(" "), 80).to_string();
            if s.is_empty() { format!("pid {pid}") } else { s }
        }
        Err(_) => format!("pid {pid}"),
    }
}

/// (pid, bytes or None when NVML cannot read it) of the compute processes on GPU `index`; [] when NVML cannot say.
pub fn processes(index: u32) -> Vec<(u32, Option<u64>)> {
    let nv = Nvml::new(index);
    if !nv.ok() {
        return Vec::new();
    }
    nv.compute_processes().unwrap_or_default().iter()
        .map(|p| (p.pid, if p.used == NOT_AVAILABLE { None } else { Some(p.used) }))
        .collect()
}

/// The GPUs NVML has, by trying indices.
pub fn gpu_count() -> u32 {
    let mut n = 0;
    while n < 16 && Nvml::new(n).ok() {
        n += 1;
    }
    n
}

/// The MiB process `pid` holds on each GPU it is on, as (gpu, MiB) in `gpus` order; None when NVML lists it nowhere or
/// cannot read its memory: the caller must not take that for zero.
pub fn process_by_gpu(pid: u32, gpus: Option<&[u32]>) -> Option<Vec<(u32, u64)>> {
    let all: Vec<u32> = match gpus {
        Some(g) => g.to_vec(),
        None => (0..gpu_count()).collect(),
    };
    let mut out: Vec<(u32, u64)> = Vec::new();
    for g in all {
        let mine: Vec<Option<u64>> = processes(g).into_iter().filter(|(p, _)| *p == pid).map(|(_, b)| b).collect();
        if mine.is_empty() {
            continue;
        }
        if mine.iter().any(Option::is_none) {
            return None;
        }
        let mib = mine.iter().map(|b| b.unwrap_or(0)).sum::<u64>() >> 20;
        match out.iter_mut().find(|(k, _)| *k == g) {
            Some(e) => e.1 = mib,
            None => out.push((g, mib)),
        }
    }
    (!out.is_empty()).then_some(out)
}

/// The MiB process `pid` holds across `gpus`; None as for process_by_gpu.
pub fn process_mib(pid: u32, gpus: Option<&[u32]>) -> Option<u64> {
    process_by_gpu(pid, gpus).map(|by| by.iter().map(|x| x.1).sum())
}

/// Free memory on GPU `index` in MiB; None when unreadable.
pub fn free_mib(index: u32) -> Option<u64> {
    let nv = Nvml::new(index);
    if !nv.ok() {
        return None;
    }
    nv.memory_info().map(|(_, free, _)| free >> 20)
}

/// The processes holding GPU memory, as "pid 123 (python -m ...) 20.0 GiB on GPU 0; ..."; "" when none is listed.
pub fn holders(gpus: Option<&[u32]>, ignore: Option<u32>) -> String {
    let all: Vec<u32> = match gpus {
        Some(g) => g.to_vec(),
        None => (0..gpu_count()).collect(),
    };
    let me = std::process::id();
    let mut rows = Vec::new();
    for g in all {
        for (pid, used) in processes(g) {
            if Some(pid) != ignore && pid != me {
                rows.push(format!("pid {pid} ({}) {} on GPU {g}", command(pid), size_text(used)));
            }
        }
    }
    rows.join("; ")
}

fn size_text(used: Option<u64>) -> String {
    match used {
        None => "an unreadable amount".into(),
        Some(b) => format!("{:.1} GiB", b as f64 / (1u64 << 30) as f64),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn texts() {
        assert_eq!(size_text(None), "an unreadable amount");
        assert_eq!(size_text(Some(20 << 30)), "20.0 GiB");
        assert_eq!(command(u32::MAX), format!("pid {}", u32::MAX));
        // a card that is not there reads as nothing, never as zero
        assert_eq!(processes(99), vec![]);
        assert_eq!(free_mib(99), None);
        assert_eq!(process_by_gpu(1, Some(&[99])), None);
        assert_eq!(holders(Some(&[99]), None), "");
    }
}
