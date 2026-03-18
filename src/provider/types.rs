use crate::error::Result;
use crate::message::Message;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub model: String,
    pub temperature: f32,
    pub max_tokens: Option<usize>,
    pub top_p: Option<f32>,
    pub stream: bool,
    pub tool_choice: Option<String>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            model: "gpt-4".to_string(),
            temperature: 0.7,
            max_tokens: None,
            top_p: None,
            stream: false,
            tool_choice: Some("auto".to_string()),
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
    async fn generate(&self, messages: Vec<Message>, config: &ModelConfig)
        -> Result<ModelResponse>;
    async fn stream_generate(
        &self,
        messages: Vec<Message>,
        config: &ModelConfig,
    ) -> Result<Box<dyn futures::Stream<Item = Result<String>> + Unpin + Send>>;
}
