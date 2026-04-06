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

pub struct Zai {
    config: ModelConfig,
    client: HttpClient,
}

impl Zai {
    pub fn new(config: ModelConfig) -> Self {
        let client = HttpClient::new(Some(config.api_key.clone()), config.base_url.clone());
        Self { config, client }
    }

    pub fn builder() -> ZaiBuilder {
        ZaiBuilder::new()
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
            "temperature": self.config.temperature,
            "stream": stream,
        });

        if let Some(max_tokens) = self.config.max_tokens {
            body["max_tokens"] = json!(max_tokens);
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

/// Builder for Zai instances
#[derive(Default)]
pub struct ZaiBuilder {
    inner: CommonBuilder,
}

impl ZaiBuilder {
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

    pub fn build(self) -> Result<Zai> {
        let config = self.inner.into_config(
            "ZAI_API_KEY",
            "ZAI_BASE_URL",
            "https://api.z.ai/api/paas/v4",
            "gpt-4",
        )?;

        let client = HttpClient::new(Some(config.api_key.clone()), config.base_url.clone());

        Ok(Zai { config, client })
    }
}

#[derive(Debug, Deserialize)]
struct ZaiResponse {
    choices: Vec<ZaiChoice>,
    usage: ZaiUsage,
    model: String,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    request_id: Option<String>,
    #[serde(default)]
    created: Option<u64>,
    #[serde(default)]
    object: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ZaiChoice {
    message: ZaiMessage,
    finish_reason: String,
    #[serde(default)]
    index: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct ZaiMessage {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    role: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Debug, Deserialize, Clone)]
struct PromptTokensDetails {
    cached_tokens: usize,
}

#[derive(Debug, Deserialize, Clone)]
struct CompletionTokensDetails {
    #[serde(default)]
    reasoning_tokens: usize,
}

#[derive(Debug, Deserialize, Clone)]
struct ZaiUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    prompt_tokens_details: PromptTokensDetails,
    #[serde(default)]
    completion_tokens_details: Option<CompletionTokensDetails>,
    pub total_tokens: usize,
}

#[derive(Debug, Deserialize)]
struct ZaiStreamChunk {
    id: Option<String>,
    object: Option<String>,
    created: Option<u64>,
    model: String,
    choices: Vec<ZaiStreamChoice>,
    #[serde(default)]
    usage: Option<ZaiUsage>,
}

#[derive(Debug, Deserialize)]
struct ZaiStreamChoice {
    index: Option<usize>,
    delta: ZaiStreamDelta,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ZaiStreamDelta {
    ToolCalls {
        tool_calls: Vec<ToolCall>,
    },
    Content {
        role: String,
        content: String,
    },
    Reasoning {
        role: String,
        reasoning_content: String,
    },
}

// Implement SseChunk trait for ZAI chunks
impl SseChunk for ZaiStreamChunk {
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
        self.choices.first().and_then(|c| match &c.delta {
            ZaiStreamDelta::Reasoning {
                reasoning_content, ..
            } => Some(reasoning_content.as_str()),
            _ => None,
        })
    }

    fn content_delta(&self) -> Option<&str> {
        self.choices.first().and_then(|c| match &c.delta {
            ZaiStreamDelta::Content { content, .. } => Some(content.as_str()),
            _ => None,
        })
    }

    fn tool_call_deltas(&self) -> Option<&[ToolCall]> {
        self.choices.first().and_then(|c| match &c.delta {
            ZaiStreamDelta::ToolCalls { tool_calls } => Some(tool_calls.as_slice()),
            _ => None,
        })
    }
}

impl From<ZaiUsage> for Usage {
    fn from(usage: ZaiUsage) -> Self {
        Self {
            prompt_tokens: usage.prompt_tokens,
            completion_tokens: usage.completion_tokens,
            cached_tokens: usage.prompt_tokens_details.cached_tokens,
            total_tokens: usage.total_tokens,
        }
    }
}

impl TryFrom<ZaiResponse> for ModelResponse {
    type Error = KongfuError;

    fn try_from(response: ZaiResponse) -> Result<Self> {
        let choice = response
            .choices
            .first()
            .ok_or_else(|| KongfuError::ExecutionError("No choices in Zai response".to_string()))?;

        let mut content_blocks = Vec::new();

        if let Some(reasoning) = &choice.message.reasoning_content {
            content_blocks.push(ContentBlock::thinking(reasoning.clone()));
        }

        if let Some(text) = &choice.message.content {
            content_blocks.push(ContentBlock::text(text.clone()));
        }

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

// Type alias for the generic SSE stream using ZAI chunks
type ZaiResponseStream = SseStream<ZaiStreamChunk>;

#[async_trait]
impl Provider for Zai {
    fn name(&self) -> ProviderName {
        ProviderName::Zai
    }

    async fn generate(
        &self,
        messages: &[Message],
        tools: Option<&[Tool]>,
        options: &RequestOptions,
    ) -> Result<ModelResponse> {
        let body = self.build_request_body(messages, tools, options, false);
        let response = self.client.post("chat/completions", &body).await?;

        let api_response: ZaiResponse = response.json().await.map_err(|e| {
            KongfuError::ResponseParseError(format!("Failed to parse Zai response: {}", e))
        })?;

        Ok(api_response.try_into()?)
    }
}

#[async_trait]
impl StreamingProvider for Zai {
    async fn stream_generate(
        &self,
        messages: &[Message],
        tools: Option<&[Tool]>,
        options: &RequestOptions,
    ) -> Result<Box<dyn futures::Stream<Item = Result<StreamingUpdate>> + Unpin + Send>> {
        let body = self.build_request_body(messages, tools, options, true);
        let response = self.client.post("chat/completions", &body).await?;

        let byte_stream = response.bytes_stream();
        let stream = ZaiResponseStream::new(byte_stream, self.config.model.clone());

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
    async fn test_zai_generate() {
        let zai = Zai::builder()
            .base_url("https://api.z.ai/api/coding/paas/v4")
            .model("glm-4.7")
            .temperature(1.0)
            .max_tokens(48000)
            .build()
            .unwrap();
        let options = RequestOptions { tool_choice: None };
        let messages = vec![
            Message::system("You are a helpful AI helper"),
            Message::user("Explain what's an LLM in short?"),
        ];
        let resp = zai.generate(&messages, None, &options).await;
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
    async fn test_zai_stream_generate() {
        use futures::StreamExt;

        let zai = Zai::builder()
            .base_url("https://api.z.ai/api/coding/paas/v4")
            .model("glm-4.7")
            .temperature(1.0)
            .max_tokens(48_000)
            .build()
            .unwrap();

        let options = RequestOptions { tool_choice: None };

        let messages = vec![
            Message::system("You are a helpful AI helper"),
            Message::user("Explain what's an LLM in short?"),
        ];

        let mut stream = zai
            .stream_generate(&messages, None, &options)
            .await
            .unwrap();

        let mut thinking_content = String::new();
        let mut response_content = String::new();
        let mut thinking_count = 0;
        let mut content_count = 0;
        let mut total_results = 0;
        let mut final_response = None;

        while let Some(update_result) = stream.next().await {
            total_results += 1;
            match update_result {
                Ok(StreamingUpdate::Thinking(chunk)) => {
                    thinking_count += 1;
                    thinking_content.push_str(&chunk);
                    if thinking_count == 1 {
                        print!("<think> {}", chunk);
                    } else {
                        print!("{}", chunk);
                    }
                    std::io::stdout().flush().unwrap();
                }
                Ok(StreamingUpdate::Content(chunk)) => {
                    content_count += 1;
                    response_content.push_str(&chunk);
                    if content_count == 1 && thinking_count > 0 {
                        print!("\n</think>\n{}", chunk);
                    } else {
                        print!("{}", chunk); // Stream content in real-time
                    }
                    std::io::stdout().flush().unwrap();
                }
                Ok(StreamingUpdate::ToolCall(_tool_call)) => {
                    // Handle tool calls if any
                    println!("\n[Tool call received]");
                }
                Ok(StreamingUpdate::Done(response)) => {
                    final_response = Some(response);
                    println!("\n[Stream complete]");
                }
                Err(e) => {
                    eprintln!("Stream error: {}", e);
                }
            }
        }

        println!("\n=== Stream Summary ===");
        println!("Total updates: {}", total_results);
        println!("Thinking chunks: {}", thinking_count);
        println!("Content chunks: {}", content_count);
        if !thinking_content.is_empty() {
            println!("Total thinking: {} bytes", thinking_content.len());
        }
        println!("Total content: {} bytes", response_content.len());

        if let Some(response) = &final_response {
            println!("Model: {}", response.model);
            if let Some(reason) = &response.finish_reason {
                println!("Finish reason: {}", reason);
            }
        }

        assert!(
            content_count > 0 || thinking_count > 0,
            "Should receive at least one chunk"
        );
    }

    #[tokio::test]
    #[ignore = "needs api key and takes time"]
    async fn test_zai_stream_generate_with_tools() {
        use futures::StreamExt;

        let zai = Zai::builder()
            .base_url("https://open.bigmodel.cn/api/paas/v4")
            .model("glm-4.7-flash")
            .temperature(1.0)
            .max_tokens(48_000)
            .build()
            .unwrap();

        let tools = vec![Tool::Function(crate::provider::types::FunctionDefinition {
            name: "get_current_time".to_string(),
            description: "Get the current time".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {}
            }),
        })];

        let options = RequestOptions {
            tool_choice: Some(ToolChoice::Auto),
        };

        let messages = vec![
            Message::system("You are a helpful AI helper"),
            Message::user("What time is it? Use the get_current_time tool."),
        ];

        let mut stream = zai
            .stream_generate(&messages, Some(&tools), &options)
            .await
            .unwrap();

        let mut tool_call_count = 0;
        let mut content_count = 0;
        let mut final_response = None;

        while let Some(update_result) = stream.next().await {
            match update_result {
                Ok(StreamingUpdate::ToolCall(_tool_call)) => {
                    tool_call_count += 1;
                    println!("\n[Tool call received]");
                }
                Ok(StreamingUpdate::Content(_chunk)) => {
                    content_count += 1;
                }
                Ok(StreamingUpdate::Done(response)) => {
                    final_response = Some(response);
                    println!("\n[Stream complete]");
                }
                Ok(StreamingUpdate::Thinking(_chunk)) => {
                    // Ignore thinking for this test
                }
                Err(e) => {
                    eprintln!("Stream error: {}", e);
                }
            }
        }

        println!("\n=== Tool Call Test Summary ===");
        println!("Tool calls received: {}", tool_call_count);
        println!("Content chunks: {}", content_count);

        // Verify we got a final response
        assert!(final_response.is_some(), "Should receive final response");

        // Verify the response contains tool use blocks
        if let Some(response) = &final_response {
            let has_tool_use = response
                .content
                .iter()
                .any(|block| matches!(block, ContentBlock::ToolUse(_)));
            assert!(has_tool_use, "Response should contain tool use blocks");
        }
    }

    #[test]
    fn test_build_request_body() {
        let zai = Zai::builder()
            .api_key("test-key")
            .base_url("https://api.test.com")
            .model("test-model")
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
        let body = zai.build_request_body(&messages, Some(&tools), &options, false);

        assert_eq!(body["model"], "test-model");
        assert_eq!(body["stream"], false);
        assert_eq!(body["temperature"].as_f64().unwrap(), 0.7);
        assert_eq!(body["max_tokens"], 1000);
        assert_eq!(body["top_p"].as_f64().unwrap(), 0.9);
        assert!(body["tool_choice"].is_string());
        assert!(body["tools"].is_array());
        assert_eq!(body["messages"].as_array().unwrap().len(), 1);

        // Test with stream=true
        let body_stream = zai.build_request_body(&messages, None, &options, true);
        assert_eq!(body_stream["stream"], true);

        // Test without optional parameters
        let zai_minimal = Zai::builder().api_key("test-key").build().unwrap();

        let body_minimal = zai_minimal.build_request_body(
            &messages,
            None,
            &RequestOptions { tool_choice: None },
            false,
        );

        assert_eq!(body_minimal["model"], "gpt-4");
        assert!((body_minimal["temperature"].as_f64().unwrap() - 0.7).abs() < 0.01);
        assert!(body_minimal.get("max_tokens").is_none());
        assert!(body_minimal.get("top_p").is_none());
        assert!(body_minimal.get("tool_choice").is_none());
        assert!(body_minimal.get("tools").is_none());
    }
}
