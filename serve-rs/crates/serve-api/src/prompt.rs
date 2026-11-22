//! serve/prompt.py (the generator half): primed and tokens_known. PromptText itself lives in serve-text.

use crate::error::{AResult, ApiError};
use serve_engine::errors::EngineErr;
use serve_engine::gen::{Gen, Stop};
use serve_engine::iface::Prompt;

pub use serve_text::prompt::{overflow_error, room_error, PromptText, CTX_SLACK};

/// tokens_known: the prompt's token count is known (ids, or a PromptText the engine has counted).
pub fn tokens_known(prompt: &Prompt) -> bool {
    match prompt {
        Prompt::Ids(_) => true,
        Prompt::Text(t) => t.n().is_some(),
    }
}

/// primed: `events` advanced until the engine has counted a text prompt, so an error the count finds (a prompt over
/// the context) is raised before a stream's headers go out. The items read on the way are replayed first.
pub fn primed<T: Send + 'static>(mut events: Gen<T>, prompt: &Prompt) -> AResult<Gen<T>> {
    if tokens_known(prompt) {
        return Ok(events);
    }
    let mut held: Vec<T> = vec![];
    let mut later: Option<EngineErr> = None;
    let mut ended = false;
    while !tokens_known(prompt) {
        match events.next() {
            None => {
                ended = true;
                break;
            }
            Some(Ok(x)) => held.push(x),
            Some(Err(e)) if e.is_value() => {
                let _ = events.close();
                return Err(ApiError::Engine(e));
            }
            Some(Err(e)) => {
                later = Some(e);
                break;
            }
        }
    }
    Ok(Gen::new(move |y| {
        let mut events = events;
        let r = (|| -> Result<(), Stop> {
            for x in held {
                y.yield_(x)?;
            }
            if let Some(e) = later {
                return Err(Stop::Raised(e));
            }
            if ended {
                return Ok(());
            }
            while let Some(item) = events.next() {
                match item {
                    Ok(x) => y.yield_(x)?,
                    Err(e) => return Err(Stop::Raised(e)),
                }
            }
            Ok(())
        })();
        let c = events.close();
        match (r, c) {
            (Err(Stop::Closed), Err(e)) => Err(Stop::Raised(e)),
            (Ok(()), Err(e)) => Err(Stop::Raised(e)),
            (r, _) => r,
        }
    }))
}
