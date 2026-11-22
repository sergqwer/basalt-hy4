//! Python generators on a thread: the body runs only while the consumer waits for an item, so it runs exactly the
//! code a generator would between two next() calls. close() (or Drop) makes the pending yield fail with Closed, the
//! body unwinds through its cleanup (Python's `finally`), and close() returns the error that cleanup raised.
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::thread::JoinHandle;

use crate::errors::{EResult, EngineErr};

/// The consumer left: the body must return (running its cleanup).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Closed;

enum Msg<T> {
    Item(T),
    Err(EngineErr),
}

/// The producer's side: `yield_` hands one item over and waits for the next request.
pub struct Yielder<T> {
    out: SyncSender<Msg<T>>,
    resume: Receiver<()>,
}

impl<T> Yielder<T> {
    /// `yield item`: Err(Closed) when the consumer closed the generator instead of asking for more.
    pub fn yield_(&self, item: T) -> Result<(), Closed> {
        self.out.send(Msg::Item(item)).map_err(|_| Closed)?;
        self.resume.recv().map_err(|_| Closed)
    }
}

/// What a generator body returns: Ok at its end, Err(Raised) for an exception, Err(Closed) when the consumer left.
#[derive(Debug)]
pub enum Stop {
    Closed,
    Raised(EngineErr),
}

impl From<Closed> for Stop {
    fn from(_: Closed) -> Self {
        Stop::Closed
    }
}

impl From<EngineErr> for Stop {
    fn from(e: EngineErr) -> Self {
        Stop::Raised(e)
    }
}

pub type Body<T> = Box<dyn FnOnce(&Yielder<T>) -> Result<(), Stop> + Send>;

/// A generator: an iterator of EResult<T>; an Err item ends it (the exception at next()).
pub struct Gen<T: Send + 'static> {
    body: Option<Body<T>>,
    resume: Option<SyncSender<()>>,
    items: Option<Receiver<Msg<T>>>,
    thread: Option<JoinHandle<Option<EngineErr>>>,
    done: bool,
}

impl<T: Send + 'static> Gen<T> {
    pub fn new(body: impl FnOnce(&Yielder<T>) -> Result<(), Stop> + Send + 'static) -> Self {
        Gen { body: Some(Box::new(body)), resume: None, items: None, thread: None, done: false }
    }

    /// A generator that raises at its first next().
    pub fn raising(e: EngineErr) -> Self {
        Gen::new(move |_| Err(Stop::Raised(e)))
    }

    fn start(&mut self) {
        let body = self.body.take().expect("body");
        let (otx, orx) = sync_channel::<Msg<T>>(0);
        let (rtx, rrx) = sync_channel::<()>(1);
        let h = std::thread::Builder::new()
            .name("gen".into())
            .spawn(move || {
                let y = Yielder { out: otx, resume: rrx };
                // wait for the first next()
                if y.resume.recv().is_err() {
                    return None;
                }
                match body(&y) {
                    Ok(()) => None,
                    Err(Stop::Closed) => None,
                    Err(Stop::Raised(e)) => {
                        // an exception: to the consumer at next(), else (closed) to close()
                        match y.out.send(Msg::Err(e)) {
                            Ok(()) => None,
                            Err(std::sync::mpsc::SendError(Msg::Err(e))) => Some(e),
                            Err(_) => None,
                        }
                    }
                }
            })
            .expect("generator thread");
        self.resume = Some(rtx);
        self.items = Some(orx);
        self.thread = Some(h);
    }

    /// generator.close(): the body's cleanup runs; the error it raised, if any.
    pub fn close(&mut self) -> EResult<()> {
        self.done = true;
        self.body = None;
        self.resume = None;              // the pending yield sees Closed
        self.items = None;
        if let Some(h) = self.thread.take() {
            if let Ok(Some(e)) = h.join() {
                return Err(e);
            }
        }
        Ok(())
    }

    pub fn is_started(&self) -> bool {
        self.body.is_none()
    }
}

impl<T: Send + 'static> Iterator for Gen<T> {
    type Item = EResult<T>;
    fn next(&mut self) -> Option<EResult<T>> {
        if self.done {
            return None;
        }
        if self.body.is_some() {
            self.start();
        }
        if self.resume.as_ref().map(|r| r.send(()).is_err()).unwrap_or(true) {
            self.finish();
            return None;
        }
        match self.items.as_ref().and_then(|r| r.recv().ok()) {
            Some(Msg::Item(t)) => Some(Ok(t)),
            Some(Msg::Err(e)) => {
                self.finish();
                Some(Err(e))
            }
            None => {
                self.finish();
                None
            }
        }
    }
}

impl<T: Send + 'static> Gen<T> {
    fn finish(&mut self) {
        self.done = true;
        self.resume = None;
        self.items = None;
        if let Some(h) = self.thread.take() {
            let _ = h.join();
        }
    }
}

impl<T: Send + 'static> Drop for Gen<T> {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[test]
    fn lazy_items_and_finally() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let l2 = log.clone();
        let mut g = Gen::new(move |y: &Yielder<i32>| {
            l2.lock().unwrap().push("start");
            let r = (|| -> Result<(), Stop> {
                for i in 0..10 {
                    y.yield_(i)?;
                }
                Ok(())
            })();
            l2.lock().unwrap().push("finally");
            r
        });
        assert!(log.lock().unwrap().is_empty());
        assert_eq!(g.next().unwrap().unwrap(), 0);
        assert_eq!(g.next().unwrap().unwrap(), 1);
        g.close().unwrap();
        assert_eq!(*log.lock().unwrap(), vec!["start", "finally"]);
        assert!(g.next().is_none());
    }

    #[test]
    fn raise_at_next_and_in_finally() {
        let mut g: Gen<i32> = Gen::new(|y| {
            y.yield_(1)?;
            Err(Stop::Raised(EngineErr::value("bad")))
        });
        assert_eq!(g.next().unwrap().unwrap(), 1);
        assert_eq!(g.next().unwrap().unwrap_err(), EngineErr::value("bad"));
        assert!(g.next().is_none());
        let mut g: Gen<i32> = Gen::new(|y| {
            let r = y.yield_(1);
            if r.is_err() {
                return Err(Stop::Raised(EngineErr::Silent("late".into())));
            }
            Ok(())
        });
        g.next();
        assert_eq!(g.close().unwrap_err(), EngineErr::Silent("late".into()));
    }
}
