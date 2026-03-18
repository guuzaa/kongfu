use thiserror::Error;

#[derive(Error, Debug)]
pub enum KongfuError {
    #[error("Tool not found: {0}")]
    ToolNotFound(String),

    #[error("Tool execution failed: {0}")]
    ToolExecutionFailed(String),

    #[error("Memory error: {0}")]
    MemoryError(String),

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    #[error("Kongfu execution error: {0}")]
    ExecutionError(String),
}

pub type Result<T> = std::result::Result<T, KongfuError>;
