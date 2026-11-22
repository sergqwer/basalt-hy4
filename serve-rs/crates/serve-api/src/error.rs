//! The exceptions the API layer raises, as the handler's except-chain tells them apart.

pub use serve_engine::errors::EngineErr;

/// A Responses-format request error (api/responses.py ResponsesError, a ValueError).
#[derive(Debug, Clone, PartialEq)]
pub struct ResponsesError {
    pub message: String,
    pub param: Option<String>,
    pub code: Option<String>,
    pub status: u16,
    pub kind: String,
}

impl ResponsesError {
    pub fn new(message: impl Into<String>, param: Option<&str>, code: Option<&str>) -> Self {
        ResponsesError {
            message: message.into(),
            param: param.map(String::from),
            code: code.map(String::from),
            status: 400,
            kind: "invalid_request_error".into(),
        }
    }

    pub fn body(&self) -> serde_json::Value {
        crate::api::responses::error_body(&self.message, &self.kind, self.param.as_deref(), self.code.as_deref())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ApiError {
    /// Any engine-side class, and ValueError as EngineErr::Value (a 400 with its message).
    Engine(EngineErr),
    /// ResponsesError (also a ValueError: str(e) is its message).
    Responses(ResponsesError),
    /// An exception Python does not catch in the handler (TypeError, KeyError, a template's own error...): the
    /// request ends without a response, the connection closes.
    Internal(String),
}

impl ApiError {
    pub fn value(msg: impl Into<String>) -> Self {
        ApiError::Engine(EngineErr::Value(msg.into()))
    }

    pub fn internal(msg: impl Into<String>) -> Self {
        ApiError::Internal(msg.into())
    }

    /// isinstance(e, ValueError)
    pub fn is_value(&self) -> bool {
        match self {
            ApiError::Engine(e) => e.is_value(),
            ApiError::Responses(_) => true,
            ApiError::Internal(_) => false,
        }
    }

    /// str(e)
    pub fn message(&self) -> String {
        match self {
            ApiError::Engine(e) => e.message(),
            ApiError::Responses(r) => r.message.clone(),
            ApiError::Internal(m) => m.clone(),
        }
    }
}

impl From<EngineErr> for ApiError {
    fn from(e: EngineErr) -> Self {
        ApiError::Engine(e)
    }
}

impl From<ResponsesError> for ApiError {
    fn from(e: ResponsesError) -> Self {
        ApiError::Responses(e)
    }
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message())
    }
}

pub type AResult<T> = Result<T, ApiError>;
