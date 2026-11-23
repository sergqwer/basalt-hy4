//! The prompt read's clock, from the lines the engine already sends: a `PP <done> <total> <ms> <tok/s>` line ends each
//! prompt chunk (`ms` since the request started, on the engine's clock), `PL` moves the position inside a chunk, and
//! `RESUME <n>` says where the read starts. It gives the stats and the console the chunk count, the last chunk's time,
//! the time spent in the current chunk (a stalled chunk shows its own age) and an ETA at the last chunk's rate, which
//! grows honestly when the chunks slow down. Nothing here touches the engine: it runs as the lines arrive.

/// Seconds since the epoch, as the service's `now()`.
pub fn wall() -> f64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0)
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ReadClock {
    started: f64,          // when the request was sent (wall s)
    last_at: Option<f64>,  // when the last chunk ended (wall s)
    prev_done: i64,        // the position the last chunk ended at (or the reused tokens)
    prev_ms: f64,          // the engine's ms at the last chunk's end
    done: i64,
    total: i64,
    chunks: u32,
    last_chunk_ms: Option<f64>,
    last_chunk_tokens: i64,
}

/// What `/v1/stats` and the console show of a prompt read.
#[derive(Clone, Debug, PartialEq)]
pub struct ReadProgress {
    pub done: i64,
    pub total: i64,
    pub chunks: u32,
    pub last_chunk_ms: Option<f64>,
    pub in_chunk_s: f64,
    pub eta_s: Option<f64>,
}

impl ReadClock {
    pub fn start(now: f64) -> Self {
        ReadClock { started: now, ..Default::default() }
    }
    /// `RESUME <n>`: the first chunk starts after the reused tokens.
    pub fn resume(&mut self, reused: i64) {
        if self.chunks == 0 && reused > self.prev_done {
            self.prev_done = reused;
            self.done = self.done.max(reused);
        }
    }
    /// A `PP` line: a chunk ended at `done` of `total`, `ms` after the request started (engine clock).
    pub fn chunk(&mut self, done: i64, total: i64, ms: Option<f64>, now: f64) {
        self.total = total;
        if done > self.prev_done {
            let took = match ms {
                Some(ms) if ms >= self.prev_ms => {
                    let d = ms - self.prev_ms;
                    self.prev_ms = ms;
                    d
                }
                _ => (now - self.last_at.unwrap_or(self.started)).max(0.0) * 1000.0,
            };
            self.last_chunk_ms = Some(took);
            self.last_chunk_tokens = done - self.prev_done;
            self.prev_done = done;
            self.chunks += 1;
            self.last_at = Some(now);
        }
        self.done = self.done.max(done);
    }
    /// A `PL` line: the position inside the current chunk.
    pub fn live(&mut self, done: i64, total: i64) {
        self.total = total;
        self.done = self.done.max(done);
    }
    /// The read at `now`; None before the engine has named the prompt's length.
    pub fn snapshot(&self, now: f64) -> Option<ReadProgress> {
        if self.total <= 0 {
            return None;
        }
        let eta_s = match self.last_chunk_ms {
            Some(ms) if ms > 0.0 && self.last_chunk_tokens > 0 => {
                Some((self.total - self.done).max(0) as f64 * ms / self.last_chunk_tokens as f64 / 1000.0)
            }
            _ => None,
        };
        Some(ReadProgress {
            done: self.done.min(self.total),
            total: self.total,
            chunks: self.chunks,
            last_chunk_ms: self.last_chunk_ms,
            in_chunk_s: (now - self.last_at.unwrap_or(self.started)).max(0.0),
            eta_s,
        })
    }
}

impl ReadProgress {
    /// The `prompt_progress` object of a `/v1/stats` slot row (ms and s to 0.1).
    pub fn json(&self) -> serde_json::Value {
        let r1 = |x: f64| (x * 10.0).round() / 10.0;
        serde_json::json!({"done": self.done, "total": self.total, "chunks": self.chunks,
                           "last_chunk_ms": self.last_chunk_ms.map(r1), "in_chunk_s": r1(self.in_chunk_s),
                           "eta_s": self.eta_s.map(r1)})
    }
    /// The console's addition to a `read pp=` line: " chunk 3 (last 24.1s, in chunk 37s) eta 5m10s".
    pub fn console(&self) -> String {
        let mut s = format!(" chunk {}", self.chunks + 1);
        let last = self.last_chunk_ms.map(|ms| format!("last {:.1}s, ", ms / 1000.0)).unwrap_or_default();
        s.push_str(&format!(" ({last}in chunk {:.0}s)", self.in_chunk_s));
        if let Some(e) = self.eta_s {
            s.push_str(&format!(" eta {}", mmss(e)));
        }
        s
    }
}

/// 75.4 -> "1m15s", 9.2 -> "9s".
pub fn mmss(s: f64) -> String {
    let t = s.max(0.0).round() as u64;
    if t >= 60 { format!("{}m{:02}s", t / 60, t % 60) } else { format!("{t}s") }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_time_from_the_engine_clock_and_eta_follows_the_last_chunk() {
        let mut c = ReadClock::start(100.0);
        assert_eq!(c.snapshot(101.0), None, "no total yet");
        c.chunk(16384, 65536, Some(4000.0), 104.0);
        let p = c.snapshot(106.0).unwrap();
        assert_eq!((p.done, p.total, p.chunks), (16384, 65536, 1));
        assert_eq!(p.last_chunk_ms, Some(4000.0));
        assert!((p.in_chunk_s - 2.0).abs() < 1e-9);
        assert!((p.eta_s.unwrap() - 12.0).abs() < 1e-9, "49152 left at 4 s per 16384");
        // a slower chunk: the ETA grows with it (the long-prompt cliff shows instead of hiding)
        c.chunk(32768, 65536, Some(14000.0), 114.0);
        let p = c.snapshot(114.0).unwrap();
        assert_eq!(p.last_chunk_ms, Some(10000.0));
        assert!((p.eta_s.unwrap() - 20.0).abs() < 1e-9);
        // a stalled chunk: its age grows, nothing else moves
        let p = c.snapshot(714.0).unwrap();
        assert_eq!((p.done, p.chunks), (32768, 2));
        assert!((p.in_chunk_s - 600.0).abs() < 1e-9);
    }

    #[test]
    fn resume_and_live_lines() {
        let mut c = ReadClock::start(0.0);
        c.resume(60000);
        c.chunk(60100, 60101, Some(50.0), 0.1);
        let p = c.snapshot(0.1).unwrap();
        assert_eq!(c.last_chunk_tokens, 100, "the first chunk starts after the reused tokens");
        assert_eq!(p.chunks, 1);
        let mut c = ReadClock::start(0.0);
        c.live(5000, 100000);
        let p = c.snapshot(3.0).unwrap();
        assert_eq!((p.done, p.chunks, p.eta_s), (5000, 0, None), "a PL line moves the position, not the chunks");
        assert!((p.in_chunk_s - 3.0).abs() < 1e-9);
        // a PP line without the ms field times the chunk on the server's clock
        c.chunk(16384, 100000, None, 8.0);
        assert_eq!(c.snapshot(8.0).unwrap().last_chunk_ms, Some(8000.0));
        // a repeated PP at the same position is not a chunk
        c.chunk(16384, 100000, Some(9000.0), 9.0);
        assert_eq!(c.snapshot(9.0).unwrap().chunks, 1);
    }

    #[test]
    fn json_and_console_shapes() {
        let mut c = ReadClock::start(0.0);
        c.chunk(16384, 400000, Some(24123.0), 24.0);
        let p = c.snapshot(61.0).unwrap();
        let j = p.json();
        let keys: Vec<&str> = j.as_object().unwrap().keys().map(|k| k.as_str()).collect();
        assert_eq!(keys, vec!["done", "total", "chunks", "last_chunk_ms", "in_chunk_s", "eta_s"]);
        assert_eq!(j["last_chunk_ms"], serde_json::json!(24123.0));
        assert_eq!(j["in_chunk_s"], serde_json::json!(37.0));
        assert_eq!(p.console(), " chunk 2 (last 24.1s, in chunk 37s) eta 9m25s");
        assert_eq!(mmss(9.2), "9s");
        assert_eq!(mmss(75.4), "1m15s");
    }
}
