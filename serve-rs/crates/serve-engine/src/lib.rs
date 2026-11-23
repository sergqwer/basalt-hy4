//! The engine side of the Basalt server: serve/engine/*.py (but vision and image sources), serve/live.py,
//! serve/util/*.py and config.py's request checks.
pub mod config_checks;
pub mod console;
pub mod errors;
pub mod flags;
pub mod gen;
pub mod iface;
pub mod lifeline;
pub mod live;
pub mod logprobs;
pub mod mock;
pub mod model_file;
pub mod modelfile;
pub mod process;
pub mod progress;
pub mod py;
pub mod read_clock;
pub mod pyfmt;
pub mod slots;
pub mod suspend;
pub mod util;
