//! serve/engine/slots.py: first-come admission to the batch slots, with bounded overtaking.
//! All calls are made under the caller's lock.

pub const OVERTAKE_MAX: usize = 4;

#[derive(Debug)]
struct Waiter {
    ticket: u64,
    skipped: usize,
}

/// A waiting request's place (Python's Waiter, compared by identity).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ticket(u64);

#[derive(Debug)]
pub struct AdmissionQueue {
    pub overtake_max: usize,
    waiters: Vec<Waiter>,
    next: u64,
}

impl Default for AdmissionQueue {
    fn default() -> Self {
        Self::new(OVERTAKE_MAX)
    }
}

impl AdmissionQueue {
    pub fn new(overtake_max: usize) -> Self {
        AdmissionQueue { overtake_max, waiters: Vec::new(), next: 0 }
    }
    pub fn join(&mut self) -> Ticket {
        let t = self.next;
        self.next += 1;
        self.waiters.push(Waiter { ticket: t, skipped: 0 });
        Ticket(t)
    }
    pub fn leave(&mut self, w: Ticket) {
        if let Some(i) = self.waiters.iter().position(|x| x.ticket == w.0) {
            self.waiters.remove(i);
        }
    }
    /// Whether `w` may take a free slot now; a granted overtake is charged to the oldest waiter.
    pub fn may_go(&mut self, w: Ticket, hit: bool) -> bool {
        let Some(head) = self.waiters.first_mut() else { return true };
        if head.ticket == w.0 {
            return true;
        }
        if hit && head.skipped < self.overtake_max {
            head.skipped += 1;
            return true;
        }
        false
    }
    pub fn len(&self) -> usize {
        self.waiters.len()
    }
    pub fn is_empty(&self) -> bool {
        self.waiters.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn overtake() {
        let mut q = AdmissionQueue::new(1);
        let a = q.join();
        let b = q.join();
        assert!(!q.may_go(b, false));
        assert!(q.may_go(b, true));
        assert!(!q.may_go(b, true));
        assert!(q.may_go(a, false));
        q.leave(a);
        assert!(q.may_go(b, false));
    }
}
