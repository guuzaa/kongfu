use crate::error::Result;
use crate::message::Message;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// Tool choice configuration for function calling
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolChoice {
    Auto,
    None,
    Required,
    Specific(String),
}

/// Configuration for model initialization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub model: String,
    pub base_url: String,
    pub api_key: String,
    pub temperature: f32,
    pub max_tokens: Option<u32>,
    pub top_p: Option<f32>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            model: "gpt-4".to_string(),
            base_url: "".to_string(),
            api_key: "".to_string(),
            temperature: 0.7,
            max_tokens: None,
            top_p: None,
        }
    }
}

/// Options for individual requests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestOptions {
    pub stream: bool,
    pub tool_choice: Option<ToolChoice>,
}

impl Default for RequestOptions {
    fn default() -> Self {
        Self {
            stream: false,
            tool_choice: Some(ToolChoice::Auto),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelResponse {
    pub content: String,
    pub model: String,
    pub usage: Option<Usage>,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Deserialize)]
pub struct Choice {
    pub(crate) message: MessageContent,
    pub(crate) finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
// TODO
pub struct MessageContent {
    pub(crate) content: String,
}

#[async_trait]
pub trait Provider: Send + Sync {
    fn name(&self) -> &str;
    async fn generate(
        &self,
        messages: Vec<Message>,
        options: &RequestOptions,
    ) -> Result<ModelResponse>;
    async fn stream_generate(
        &self,
        messages: Vec<Message>,
        options: &RequestOptions,
    ) -> Result<Box<dyn futures::Stream<Item = Result<String>> + Unpin + Send>>;
}
