//! serve/engine/mock.py: MockEngine, a scripted engine that needs no GPU. Replays scripted completions as token
//! ids plus the end-of-turn token; a list gives one script per request, the last repeating.
use std::any::Any;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use serve_text::text::IM_END;
use serve_text::Tokenizer;

use crate::gen::{Gen, Stop, Yielder};
use crate::iface::{Engine, Flag, GenStream, JsonMap, Prompt, Sampling};
use crate::logprobs::Tok;
use crate::slots::AdmissionQueue;

pub type Route = Box<dyn Fn(&[u32]) -> usize + Send + Sync>;

pub struct MockState {
    pub busy: usize,
    pub admission: AdmissionQueue,
    /// most requests generating at once
    pub peak: usize,
    /// (prompt, sampling) of each request, in arrival order
    pub requests: Vec<(Vec<u32>, JsonMap)>,
    pub script: Vec<u32>,
    pub turns: usize,
    pub last_prompt: Vec<u32>,
    pub last_embeddings: Option<String>,
}

pub struct MockInner {
    pub tok: Arc<dyn Tokenizer>,
    pub max_context: i64,
    pub delay: f64,
    pub batch: usize,
    pub route: Option<Route>,
    pub scripts: Vec<Vec<u32>>,
    pub st: Mutex<MockState>,
    pub cv: Condvar,
}

pub struct MockEngine {
    pub inner: Arc<MockInner>,
}

impl MockEngine {
    pub fn new(tok: Arc<dyn Tokenizer>, scripts: Vec<String>, max_context: i64, delay_s: f64, batch: usize,
               route: Option<Route>) -> Self {
        let end = tok.encode(IM_END, true, &[]);
        let scripts: Vec<Vec<u32>> = scripts
            .iter()
            .map(|x| {
                let mut v = tok.encode(x, true, &[]);
                v.extend_from_slice(&end);
                v
            })
            .collect();
        let first = scripts.first().cloned().unwrap_or_default();
        MockEngine {
            inner: Arc::new(MockInner {
                tok,
                max_context,
                delay: delay_s,
                batch: if batch >= 2 { batch } else { 0 },
                route,
                scripts,
                st: Mutex::new(MockState {
                    busy: 0,
                    admission: AdmissionQueue::default(),
                    peak: 0,
                    requests: Vec::new(),
                    script: first,
                    turns: 0,
                    last_prompt: Vec::new(),
                    last_embeddings: None,
                }),
                cv: Condvar::new(),
            }),
        }
    }

    /// The default mock (server.py): 32768 context, no delay.
    pub fn with_scripts(tok: Arc<dyn Tokenizer>, scripts: Vec<String>, batch: usize) -> Self {
        Self::new(tok, scripts, 32768, 0.0, batch, None)
    }

    pub fn peak(&self) -> usize {
        self.inner.st.lock().unwrap().peak
    }
    pub fn requests(&self) -> Vec<(Vec<u32>, JsonMap)> {
        self.inner.st.lock().unwrap().requests.clone()
    }
    pub fn last_prompt(&self) -> Vec<u32> {
        self.inner.st.lock().unwrap().last_prompt.clone()
    }
}

fn run(m: &MockInner, ids: &[u32], max_new: i64, sampling: &JsonMap, cancel: &Flag, embeddings: Option<String>,
       y: &Yielder<Option<Tok>>) -> Result<(), Stop> {
    let script = {
        let mut st = m.st.lock().unwrap();
        st.requests.push((ids.to_vec(), sampling.clone()));
        st.last_prompt = ids.to_vec();
        st.last_embeddings = embeddings;
        let mut script = st.script.clone();
        if let Some(route) = &m.route {
            script = m.scripts[route(ids)].clone();
        } else if m.scripts.len() > 1 {
            script = m.scripts[st.turns.min(m.scripts.len() - 1)].clone();
            st.turns += 1;
        }
        st.script = script.clone();
        script
    };
    // script[:max_new]: Python slicing, a negative bound counts from the end
    let n = script.len() as i64;
    let end = if max_new < 0 { (n + max_new).max(0) } else { max_new.min(n) } as usize;
    for &t in &script[..end] {
        if cancel.is_set() {
            return Ok(());
        }
        if m.delay > 0.0 {
            std::thread::sleep(Duration::from_secs_f64(m.delay));
        }
        y.yield_(Some(Tok::plain(t)))?;
    }
    Ok(())
}

impl Engine for MockEngine {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn generate(&self, prompt: Prompt, max_new: i64, sampling: &Sampling, cancel: Flag, embeddings: Option<String>,
                _session: Option<String>, _stop_soon: Option<Flag>) -> GenStream {
        let m = self.inner.clone();
        let sampling = sampling.map.clone();
        let ids = match prompt {
            Prompt::Ids(v) => v,
            Prompt::Text(t) => t.ids(),
        };
        Gen::new(move |y| {
            if m.batch == 0 {
                return run(&m, &ids, max_new, &sampling, &cancel, embeddings, y);
            }
            {
                let mut st = m.st.lock().unwrap();
                let ticket = st.admission.join();
                loop {
                    let ok = st.busy < m.batch && st.admission.may_go(ticket, false);
                    if ok {
                        break;
                    }
                    st = m.cv.wait_timeout(st, Duration::from_millis(50)).unwrap().0;
                    if cancel.is_set() {
                        st.admission.leave(ticket);
                        m.cv.notify_all();
                        return Ok(());
                    }
                }
                st.busy += 1;
                st.peak = st.peak.max(st.busy);
                st.admission.leave(ticket);
                m.cv.notify_all();
            }
            let r = run(&m, &ids, max_new, &sampling, &cancel, embeddings, y);
            {
                let mut st = m.st.lock().unwrap();
                st.busy -= 1;
                m.cv.notify_all();
            }
            r
        })
    }

    fn max_context(&self) -> i64 {
        self.inner.max_context
    }
    fn batch(&self) -> usize {
        self.inner.batch
    }
    fn queue_depth(&self) -> Option<usize> {
        Some(self.inner.st.lock().unwrap().admission.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serve_text::ByteTokenizer;

    #[test]
    fn replays_scripts() {
        let tok: Arc<dyn Tokenizer> = Arc::new(ByteTokenizer);
        let e = MockEngine::with_scripts(tok.clone(), vec!["ab".into(), "c".into()], 0);
        let toks: Vec<u32> = e
            .generate(Prompt::Ids(vec![1, 2]), 100, &Sampling::default(), Flag::new(), None, None, None)
            .map(|t| t.unwrap().unwrap().id)
            .collect();
        let mut want = tok.encode("ab", true, &[]);
        want.extend(tok.encode(IM_END, true, &[]));
        assert_eq!(toks, want);
        let toks: Vec<u32> = e
            .generate(Prompt::Ids(vec![]), 1, &Sampling::default(), Flag::new(), None, None, None)
            .map(|t| t.unwrap().unwrap().id)
            .collect();
        assert_eq!(toks, tok.encode("c", true, &[]));
        assert_eq!(e.requests().len(), 2);
    }
}
