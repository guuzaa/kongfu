use crate::error::{KongfuError, Result};
use crate::http_client::HttpClient;
use crate::message::{ContentBlock, Message, ToolUseBlock};
use crate::provider::sse_stream::{SseChunk, SseStream};
use crate::provider::types::{StreamingProvider, StreamingUpdate};
use crate::provider::{
    CommonBuilder, ModelConfig, ModelResponse, Provider, ProviderName, RequestOptions, Tool,
    ToolCall, Usage,
};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;
use std::collections::HashMap;

pub struct Xai {
    config: ModelConfig,
    client: HttpClient,
}

impl Xai {
    pub fn new(config: ModelConfig) -> Self {
        let client = HttpClient::new(Some(config.api_key.clone()), config.base_url.clone());
        Self { config, client }
    }

    pub fn builder() -> XaiBuilder {
        XaiBuilder::new()
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn build_request_body(
        &self,
        messages: &[Message],
        tools: Option<&[Tool]>,
        options: &RequestOptions,
        stream: bool,
    ) -> serde_json::Value {
        let mut body = json!({
            "model": self.config.model,
            "messages": messages,
            "stream": stream,
            "temperature": self.config.temperature,
        });

        if let Some(max_tokens) = self.config.max_tokens {
            body["max_completion_tokens"] = json!(max_tokens);
        }

        if let Some(top_p) = self.config.top_p {
            body["top_p"] = json!(top_p);
        }

        if let Some(tool_choice) = &options.tool_choice {
            body["tool_choice"] = json!(tool_choice);
        }

        if let Some(tools) = tools {
            body["tools"] = json!(tools);
        }

        body
    }
}

/// Builder for Xai instances
#[derive(Default)]
pub struct XaiBuilder {
    inner: CommonBuilder,
}

impl XaiBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.inner = self.inner.model(model);
        self
    }

    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.inner = self.inner.api_key(api_key);
        self
    }

    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.inner = self.inner.base_url(base_url);
        self
    }

    pub fn temperature(mut self, temperature: f64) -> Self {
        self.inner = self.inner.temperature(temperature);
        self
    }

    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.inner = self.inner.max_tokens(max_tokens);
        self
    }

    pub fn top_p(mut self, top_p: f64) -> Self {
        self.inner = self.inner.top_p(top_p);
        self
    }

    pub fn build(self) -> Result<Xai> {
        let config = self.inner.into_config(
            "XAI_API_KEY",
            "XAI_BASE_URL",
            "https://api.x.ai/v1",
            "grok-4-1-fast-reasoning",
        )?;

        let client = HttpClient::new(Some(config.api_key.clone()), config.base_url.clone());

        Ok(Xai { config, client })
    }
}

#[derive(Debug, Deserialize, Default, Clone)]
struct PromptTokensDetails {
    #[serde(default)]
    cached_tokens: usize,
}

#[derive(Debug, Deserialize, Default, Clone)]
struct CompletionTokensDetails {
    #[serde(default)]
    reasoning_tokens: usize,
    #[serde(default)]
    audio_tokens: usize,
    #[serde(default)]
    accepted_prediction_tokens: usize,
    #[serde(default)]
    rejected_prediction_tokens: usize,
}

#[derive(Debug, Deserialize, Clone)]
struct XaiUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    #[serde(default)]
    prompt_tokens_details: PromptTokensDetails,
    #[serde(default)]
    completion_tokens_details: Option<CompletionTokensDetails>,
    pub total_tokens: usize,
}

#[derive(Debug, Deserialize)]
struct XaiMessage {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    role: Option<String>,
    #[serde(default)]
    refusal: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Debug, Deserialize)]
struct XaiChoice {
    message: XaiMessage,
    finish_reason: String,
    #[serde(default)]
    index: Option<usize>,
    #[serde(default)]
    logprobs: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct XaiResponse {
    choices: Vec<XaiChoice>,
    usage: XaiUsage,
    model: String,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    created: Option<u64>,
    #[serde(default)]
    object: Option<String>,
    #[serde(default)]
    system_fingerprint: Option<String>,
}

#[derive(Debug, Deserialize)]
struct XaiStreamChunk {
    id: Option<String>,
    object: Option<String>,
    created: Option<u64>,
    model: String,
    choices: Vec<XaiStreamChoice>,
    #[serde(default)]
    usage: Option<XaiUsage>,
}

#[derive(Debug, Deserialize)]
struct XaiStreamChoice {
    index: Option<usize>,
    delta: XaiStreamDelta,
    #[serde(default)]
    finish_reason: Option<String>,
    #[serde(default)]
    logprobs: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize, Default)]
struct XaiStreamDelta {
    #[serde(default)]
    role: Option<String>,
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<ToolCall>>,
}

// Implement SseChunk trait for XAI chunks
impl SseChunk for XaiStreamChunk {
    fn model(&self) -> &str {
        &self.model
    }

    fn usage(&self) -> Option<Usage> {
        self.usage.clone().map(Usage::from)
    }

    fn finish_reason(&self) -> Option<&str> {
        self.choices
            .first()
            .and_then(|c| c.finish_reason.as_deref())
    }

    fn thinking_delta(&self) -> Option<&str> {
        self.choices
            .first()
            .and_then(|c| c.delta.reasoning_content.as_deref())
    }

    fn content_delta(&self) -> Option<&str> {
        self.choices
            .first()
            .and_then(|c| c.delta.content.as_deref())
    }

    fn tool_call_deltas(&self) -> Option<&[ToolCall]> {
        self.choices
            .first()
            .and_then(|c| c.delta.tool_calls.as_deref())
    }
}

impl From<XaiUsage> for Usage {
    fn from(usage: XaiUsage) -> Self {
        Self {
            prompt_tokens: usage.prompt_tokens,
            completion_tokens: usage.completion_tokens,
            cached_tokens: usage.prompt_tokens_details.cached_tokens,
            total_tokens: usage.total_tokens,
        }
    }
}

impl TryFrom<XaiResponse> for ModelResponse {
    type Error = KongfuError;

    fn try_from(response: XaiResponse) -> Result<Self> {
        let choice = response
            .choices
            .first()
            .ok_or_else(|| KongfuError::ExecutionError("No choices in Xai response".to_string()))?;

        let mut content_blocks = Vec::new();

        // Handle reasoning content (for reasoning models)
        if let Some(reasoning) = &choice.message.reasoning_content {
            content_blocks.push(ContentBlock::thinking(reasoning.clone()));
        }

        // Handle regular content
        if let Some(text) = &choice.message.content {
            content_blocks.push(ContentBlock::text(text.clone()));
        }

        // Handle tool calls
        if let Some(tool_calls) = &choice.message.tool_calls {
            for tool_call in tool_calls {
                let args_map: HashMap<String, serde_json::Value> =
                    serde_json::from_str(&tool_call.function.arguments).map_err(|e| {
                        KongfuError::ResponseParseError(format!(
                            "Failed to parse tool call arguments: {}",
                            e
                        ))
                    })?;

                content_blocks.push(ContentBlock::ToolUse(ToolUseBlock::new(
                    tool_call.id.clone(),
                    tool_call.function.name.clone(),
                    args_map,
                )));
            }
        }

        if content_blocks.is_empty() {
            content_blocks.push(ContentBlock::text(""));
        }

        Ok(Self {
            content: content_blocks,
            model: response.model,
            usage: Some(response.usage.into()),
            finish_reason: Some(choice.finish_reason.clone()),
        })
    }
}

// Type alias for the generic SSE stream using XAI chunks
type XaiResponseStream = SseStream<XaiStreamChunk>;

#[async_trait]
impl Provider for Xai {
    fn name(&self) -> ProviderName {
        ProviderName::Xai
    }

    async fn generate(
        &self,
        messages: &[Message],
        tools: Option<&[Tool]>,
        options: &RequestOptions,
    ) -> Result<ModelResponse> {
        let body = self.build_request_body(messages, tools, options, false);
        let response = self.client.post("chat/completions", &body).await?;

        let api_response: XaiResponse = response.json().await.map_err(|e| {
            KongfuError::ResponseParseError(format!("Failed to parse Xai response: {}", e))
        })?;

        Ok(api_response.try_into()?)
    }
}

#[async_trait]
impl StreamingProvider for Xai {
    async fn stream_generate(
        &self,
        messages: &[Message],
        tools: Option<&[Tool]>,
        options: &RequestOptions,
    ) -> Result<Box<dyn futures::Stream<Item = Result<StreamingUpdate>> + Unpin + Send>> {
        let body = self.build_request_body(messages, tools, options, true);
        let response = self.client.post("chat/completions", &body).await?;

        let byte_stream = response.bytes_stream();
        let stream = XaiResponseStream::new(byte_stream, self.config.model.clone());

        Ok(Box::new(stream))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::ToolChoice;
    use std::io::Write;

    #[tokio::test]
    #[ignore = "needs api key and takes time"]
    async fn test_xai_generate() {
        let xai = Xai::builder()
            .model("grok-4-1-fast-reasoning")
            .temperature(0.7)
            .max_tokens(1000)
            .build()
            .unwrap();

        let options = RequestOptions { tool_choice: None };
        let messages = vec![
            Message::system("You are a helpful AI helper"),
            Message::user("Explain what an LLM is in 20 words"),
        ];

        let resp = xai.generate(&messages, None, &options).await;
        match resp {
            Ok(response) => {
                let content = &response.content;
                let usage = response.usage.unwrap();
                assert!(usage.total_tokens > 0);
                // Print first text block if any
                for block in content {
                    if let Some(text) = block.as_text() {
                        println!("text: {}", text);
                        break;
                    }
                }
                println!("usage: {:?}", usage);
            }
            Err(err) => {
                panic!("{}", err.to_string());
            }
        }
    }

    #[tokio::test]
    #[ignore = "needs api key and takes time"]
    async fn test_xai_stream_generate() {
        use futures::StreamExt;

        let xai = Xai::builder()
            .model("grok-4-1-fast-reasoning")
            .temperature(0.7)
            .max_tokens(1000)
            .build()
            .unwrap();

        let options = RequestOptions { tool_choice: None };

        let messages = vec![
            Message::system("You are a helpful AI helper"),
            Message::user("Explain what an LLM is in 20 words"),
        ];

        let mut stream = xai
            .stream_generate(&messages, None, &options)
            .await
            .unwrap();

        let mut response_content = String::new();
        let mut content_count = 0;
        let mut final_response = None;

        while let Some(update_result) = stream.next().await {
            match update_result {
                Ok(StreamingUpdate::Content(chunk)) => {
                    content_count += 1;
                    response_content.push_str(&chunk);
                    print!("{}", chunk);
                    std::io::stdout().flush().unwrap();
                }
                Ok(StreamingUpdate::Done(response)) => {
                    final_response = Some(response);
                    println!("\n[Stream complete]");
                }
                Ok(StreamingUpdate::ToolCall(_tool_call)) => {
                    println!("\n[Tool call received]");
                }
                Ok(StreamingUpdate::Thinking(_chunk)) => {
                    // Handle thinking if needed
                }
                Err(e) => {
                    eprintln!("Stream error: {}", e);
                }
            }
        }

        println!("\n=== Stream Summary ===");
        println!("Content chunks: {}", content_count);
        println!("Total content: {} bytes", response_content.len());

        if let Some(response) = &final_response {
            println!("Model: {}", response.model);
            if let Some(reason) = &response.finish_reason {
                println!("Finish reason: {}", reason);
            }
        }

        assert!(content_count > 0, "Should receive at least one chunk");
    }

    #[test]
    fn test_build_request_body() {
        let xai = Xai::builder()
            .api_key("test-key")
            .base_url("https://api.x.ai/v1")
            .model("grok-4-1-fast-reasoning")
            .temperature(0.7)
            .max_tokens(1000)
            .top_p(0.9)
            .build()
            .unwrap();

        let messages = vec![Message::user("Hello")];
        let tools = vec![Tool::Function(crate::provider::types::FunctionDefinition {
            name: "test_tool".to_string(),
            description: "A test tool".to_string(),
            parameters: serde_json::json!({"type": "object"}),
        })];
        let options = RequestOptions {
            tool_choice: Some(ToolChoice::Auto),
        };

        // Test with stream=false
        let body = xai.build_request_body(&messages, Some(&tools), &options, false);

        assert_eq!(body["model"], "grok-4-1-fast-reasoning");
        assert_eq!(body["stream"], false);
        assert_eq!(body["temperature"].as_f64().unwrap(), 0.7);
        assert_eq!(body["max_completion_tokens"], 1000);
        assert_eq!(body["top_p"].as_f64().unwrap(), 0.9);
        assert!(body["tool_choice"].is_string());
        assert!(body["tools"].is_array());
        assert_eq!(body["messages"].as_array().unwrap().len(), 1);

        // Test with stream=true
        let body_stream = xai.build_request_body(&messages, None, &options, true);
        assert_eq!(body_stream["stream"], true);

        // Test without optional parameters
        let xai_minimal = Xai::builder().api_key("test-key").build().unwrap();

        let body_minimal = xai_minimal.build_request_body(
            &messages,
            None,
            &RequestOptions { tool_choice: None },
            false,
        );

        assert_eq!(body_minimal["model"], "grok-4-1-fast-reasoning");
        assert!(body_minimal.get("temperature").is_some()); // temperature is optional
        assert!(body_minimal.get("max_completion_tokens").is_none());
        assert!(body_minimal.get("top_p").is_none());
        assert!(body_minimal.get("tool_choice").is_none());
        assert!(body_minimal.get("tools").is_none());
    }
}
