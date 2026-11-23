//! The API layer of the Basalt server (serve/service.py, frontend.py, prompt.py, models/, api/, and the image input of
//! engine/vision.py and engine/image_source.py), ported one to one.

pub mod api;
pub mod error;
pub mod frontend;
pub mod image_source;
pub mod models;
pub mod prompt;
pub mod py;
pub mod service;
mod service_api;
mod service_run;
mod service_stats;
pub use service_stats::{STATS_REQUESTS_DEFAULT, STATS_REQUESTS_MAX};
pub mod template;
pub mod vision;
pub mod xml;
