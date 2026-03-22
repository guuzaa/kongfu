use crate::error::Result;
use crate::message::{ContentBlock, Message};
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
    pub tool_choice: Option<ToolChoice>,
}

impl Default for RequestOptions {
    fn default() -> Self {
        Self {
            tool_choice: Some(ToolChoice::Auto),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ProviderName {
    OpenAI,
    Anthropic,
    Xai,
    Zai,
    Custom(String),
}

impl ProviderName {
    pub fn as_str(&self) -> &str {
        match self {
            ProviderName::OpenAI => "openai",
            ProviderName::Anthropic => "anthropic",
            ProviderName::Xai => "xAI",
            ProviderName::Zai => "z.ai",
            ProviderName::Custom(name) => name.as_str(),
        }
    }
}

impl std::fmt::Display for ProviderName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl From<&str> for ProviderName {
    fn from(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "openai" => ProviderName::OpenAI,
            "anthropic" => ProviderName::Anthropic,
            "z.ai" | "zai" => ProviderName::Zai,
            "xai" => ProviderName::Xai,
            other => ProviderName::Custom(other.to_string()),
        }
    }
}

impl From<String> for ProviderName {
    fn from(s: String) -> Self {
        ProviderName::from(s.as_str())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Capabilities {
    pub streaming: bool,
    pub tool_use: bool,
    pub vision: bool,
    pub max_context_tokens: u32,
}

impl Default for Capabilities {
    fn default() -> Self {
        Self {
            streaming: false,
            tool_use: false,
            vision: false,
            max_context_tokens: 4096,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelResponse {
    pub content: ContentBlock,
    pub model: String,
    pub usage: Option<Usage>,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub cached_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Clone)]
pub enum StreamingUpdate {
    /// Incremental thinking/reasoning content (reasoning_content field)
    Thinking(String),
    /// Incremental response content (content field)
    Content(String),
    /// Final complete response when stream finishes
    Done(ModelResponse),
}

#[derive(Debug, Deserialize)]
pub struct Choice {
    pub(crate) message: MessageContent,
    pub(crate) finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct MessageContent {
    #[serde(default)]
    pub(crate) content: Option<String>,
    #[serde(default)]
    pub(crate) tool_calls: Option<Vec<ToolCall>>,
    #[serde(default)]
    pub(crate) tool_use_id: Option<String>,
    #[serde(default)]
    pub(crate) role: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub function: FunctionCall,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    // TODO
    pub arguments: String,
}

#[async_trait]
pub trait Provider: Send + Sync {
    fn name(&self) -> ProviderName;

    async fn generate(
        &self,
        messages: &[Message],
        options: &RequestOptions,
    ) -> Result<ModelResponse>;
}

#[async_trait]
pub trait StreamingProvider: Provider {
    async fn stream_generate(
        &self,
        messages: &[Message],
        options: &RequestOptions,
    ) -> Result<Box<dyn futures::Stream<Item = Result<StreamingUpdate>> + Unpin + Send>>;
}

pub trait Model: Send + Sync {
    fn name(&self) -> &str;
    fn capabilities(&self) -> Capabilities;
}
