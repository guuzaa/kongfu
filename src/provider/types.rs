use crate::error::{KongfuError, Result};
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
    pub temperature: f64,
    pub max_tokens: Option<u32>,
    pub top_p: Option<f64>,
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
    Ollama,
    Custom(String),
}

impl ProviderName {
    pub fn as_str(&self) -> &str {
        match self {
            ProviderName::OpenAI => "openai",
            ProviderName::Anthropic => "anthropic",
            ProviderName::Xai => "xAI",
            ProviderName::Zai => "z.ai",
            ProviderName::Ollama => "ollama",
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
            "ollama" => ProviderName::Ollama,
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

/// Common builder struct that can be used by all providers
/// Contains the shared configuration fields and setter methods
#[derive(Default)]
pub struct CommonBuilder {
    pub model: Option<String>,
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub temperature: Option<f64>,
    pub max_tokens: Option<u32>,
    pub top_p: Option<f64>,
}

impl CommonBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    pub fn temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    pub fn top_p(mut self, top_p: f64) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Convert the builder into a ModelConfig with provider-specific defaults and env vars
    pub fn into_config(
        self,
        api_key_env: &str,
        base_url_env: &str,
        default_base_url: &str,
        default_model: &str,
    ) -> Result<ModelConfig> {
        let api_key = self
            .api_key
            .or_else(|| std::env::var(api_key_env).ok())
            .ok_or_else(|| {
                KongfuError::InvalidConfig(format!(
                    "api_key is required. Set it via builder.api_key() or {} environment variable",
                    api_key_env
                ))
            })?;

        let base_url = self
            .base_url
            .or_else(|| std::env::var(base_url_env).ok())
            .unwrap_or_else(|| default_base_url.to_string());

        let model = self.model.unwrap_or_else(|| default_model.to_string());

        Ok(ModelConfig {
            model,
            base_url,
            api_key,
            temperature: self.temperature.unwrap_or(0.7),
            max_tokens: self.max_tokens,
            top_p: self.top_p,
        })
    }

    /// Convert the builder into a ModelConfig for providers that don't require API keys (like Ollama)
    pub fn into_config_no_auth(
        self,
        base_url_env: &str,
        default_base_url: &str,
        default_model: &str,
    ) -> ModelConfig {
        let base_url = self
            .base_url
            .or_else(|| std::env::var(base_url_env).ok())
            .unwrap_or_else(|| default_base_url.to_string());

        let model = self.model.unwrap_or_else(|| default_model.to_string());

        ModelConfig {
            model,
            base_url,
            api_key: String::new(), // No API key required
            temperature: self.temperature.unwrap_or(0.7),
            max_tokens: self.max_tokens,
            top_p: self.top_p,
        }
    }
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
    pub content: Vec<ContentBlock>,
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
    /// Tool call received during streaming
    ToolCall(ToolCall),
    /// Final complete response when stream finishes
    Done(ModelResponse),
}

#[derive(Debug, Clone, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub function: Function,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Function {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "function", rename_all = "lowercase")]
pub enum Tool {
    Function(FunctionDefinition),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

#[async_trait]
pub trait Provider: Send + Sync {
    fn name(&self) -> ProviderName;

    async fn generate(
        &self,
        messages: &[Message],
        tools: Option<&[Tool]>,
        options: &RequestOptions,
    ) -> Result<ModelResponse>;
}

#[async_trait]
pub trait StreamingProvider: Provider {
    async fn stream_generate(
        &self,
        messages: &[Message],
        tools: Option<&[Tool]>,
        options: &RequestOptions,
    ) -> Result<Box<dyn futures::Stream<Item = Result<StreamingUpdate>> + Unpin + Send>>;
}

pub trait Model: Send + Sync {
    fn name(&self) -> &str;
    fn capabilities(&self) -> Capabilities;
}
