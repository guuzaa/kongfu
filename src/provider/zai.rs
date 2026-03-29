use crate::error::{KongfuError, Result};
use crate::message::{ContentBlock, Message, ToolUseBlock};
use crate::provider::types::{StreamingProvider, StreamingUpdate};
use crate::provider::{
    ModelConfig, ModelResponse, Provider, ProviderName, RequestOptions, Tool, ToolCall, Usage,
};
use async_trait::async_trait;
use futures::Stream;
use serde::Deserialize;
use serde_json::json;
use std::collections::HashMap;
use std::pin::Pin;

struct ZaiClient {
    http: reqwest::Client,
    api_key: String,
    base_url: String,
}

impl ZaiClient {
    fn new(api_key: String, base_url: String) -> Self {
        Self {
            http: reqwest::Client::new(),
            api_key,
            base_url,
        }
    }

    fn endpoint(&self, path: &str) -> String {
        format!(
            "{}/{}",
            self.base_url.trim_end_matches('/'),
            path.trim_start_matches('/')
        )
    }

    async fn post(&self, path: &str, body: &serde_json::Value) -> Result<reqwest::Response> {
        let response = self
            .http
            .post(self.endpoint(path))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(body)
            .send()
            .await
            .map_err(|e| KongfuError::NetworkError(format!("Zai API request failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let error_text = response.text().await.unwrap_or_default();
            return Err(KongfuError::ApiError {
                status,
                message: error_text,
            });
        }

        Ok(response)
    }
}

pub struct Zai {
    config: ModelConfig,
    client: ZaiClient,
}

impl Zai {
    pub fn new(config: ModelConfig) -> Self {
        let client = ZaiClient::new(config.api_key.clone(), config.base_url.clone());
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

#[derive(Default)]
pub struct ZaiBuilder {
    model: Option<String>,
    api_key: Option<String>,
    base_url: Option<String>,
    temperature: Option<f64>,
    max_tokens: Option<u32>,
    top_p: Option<f64>,
}

impl ZaiBuilder {
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

    pub fn build(self) -> Result<Zai> {
        let api_key = self
            .api_key
            .or_else(|| std::env::var("ZAI_API_KEY").ok())
            .ok_or_else(|| KongfuError::InvalidConfig(
                "api_key is required. Set it via ZaiBuilder::api_key() or ZAI_API_KEY environment variable".to_string()
            ))?;

        let base_url = self
            .base_url
            .or_else(|| std::env::var("ZAI_BASE_URL").ok())
            .unwrap_or_else(|| "https://api.z.ai/api/paas/v4".to_string());

        let model = self.model.unwrap_or_else(|| "gpt-4".to_string());

        let config = ModelConfig {
            model,
            base_url,
            api_key,
            temperature: self.temperature.unwrap_or(0.7),
            max_tokens: self.max_tokens,
            top_p: self.top_p,
        };

        let client = ZaiClient::new(config.api_key.clone(), config.base_url.clone());

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

#[derive(Debug, Deserialize)]
struct PromptTokensDetails {
    cached_tokens: usize,
}

#[derive(Debug, Deserialize)]
struct CompletionTokensDetails {
    #[serde(default)]
    reasoning_tokens: usize,
}

#[derive(Debug, Deserialize)]
struct ZaiUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    #[serde(rename = "prompt_tokens_details")]
    prompt_tokens_details: PromptTokensDetails,
    #[serde(default)]
    #[serde(rename = "completion_tokens_details")]
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

struct ZaiResponseStream {
    byte_stream: Pin<Box<dyn Stream<Item = reqwest::Result<bytes::Bytes>> + Send>>,
    buffer: String,
    is_done: bool,
    thinking_content: String,
    response_content: String,
    finish_reason: Option<String>,
    model: String,
    usage: Option<Usage>,
    tool_calls: Vec<ToolCall>,
}

impl ZaiResponseStream {
    fn new<S>(byte_stream: S, model: String) -> Self
    where
        S: Stream<Item = reqwest::Result<bytes::Bytes>> + Send + 'static,
    {
        Self {
            byte_stream: Box::pin(byte_stream),
            buffer: String::new(),
            is_done: false,
            thinking_content: String::new(),
            response_content: String::new(),
            finish_reason: None,
            model,
            usage: None,
            tool_calls: Vec::new(),
        }
    }

    fn parse_sse_event(&mut self, event: &str) -> Option<Result<StreamingUpdate>> {
        let event = event.trim();
        if event.is_empty() {
            return None;
        }

        if !event.starts_with("data: ") {
            return None;
        }

        let data_str = &event[6..]; // to trim "data: "

        if data_str.trim() == "[DONE]" {
            self.is_done = true;
            let mut content_blocks = Vec::new();

            if !self.thinking_content.is_empty() {
                content_blocks.push(ContentBlock::thinking(self.thinking_content.clone()));
            }
            if !self.response_content.is_empty() {
                content_blocks.push(ContentBlock::text(self.response_content.clone()));
            }

            for tool_call in &self.tool_calls {
                let args_map: HashMap<String, serde_json::Value> =
                    match serde_json::from_str(&tool_call.function.arguments) {
                        Ok(map) => map,
                        Err(e) => {
                            return Some(Err(KongfuError::ResponseParseError(format!(
                                "Failed to parse tool call arguments: {}",
                                e
                            ))));
                        }
                    };

                content_blocks.push(ContentBlock::ToolUse(ToolUseBlock::new(
                    tool_call.id.clone(),
                    tool_call.function.name.clone(),
                    args_map,
                )));
            }

            if content_blocks.is_empty() {
                content_blocks.push(ContentBlock::text(String::new()));
            }

            let response = ModelResponse {
                content: content_blocks,
                model: self.model.clone(),
                usage: self.usage.take(), // Include usage if available
                finish_reason: self.finish_reason.clone(),
            };
            return Some(Ok(StreamingUpdate::Done(response)));
        }

        match serde_json::from_str::<ZaiStreamChunk>(data_str) {
            Ok(chunk) => {
                // Update model from the first chunk
                if self.model.is_empty() {
                    self.model = chunk.model;
                }

                if let Some(usage) = chunk.usage {
                    self.usage = Some(usage.into());
                }

                if let Some(choice) = chunk.choices.first() {
                    // Store finish reason if present
                    if let Some(reason) = &choice.finish_reason {
                        self.finish_reason = Some(reason.clone());
                    }

                    match &choice.delta {
                        ZaiStreamDelta::Reasoning {
                            reasoning_content, ..
                        } => {
                            if !reasoning_content.is_empty() {
                                self.thinking_content.push_str(reasoning_content);
                                return Some(Ok(StreamingUpdate::Thinking(
                                    reasoning_content.clone(),
                                )));
                            }
                        }
                        ZaiStreamDelta::Content { content, .. } => {
                            if !content.is_empty() {
                                self.response_content.push_str(content);
                                return Some(Ok(StreamingUpdate::Content(content.clone())));
                            }
                        }
                        ZaiStreamDelta::ToolCalls { tool_calls } => {
                            for tool_call in tool_calls {
                                // Check if we already have this tool call (by id)
                                let existing_pos =
                                    self.tool_calls.iter().position(|tc| tc.id == tool_call.id);

                                if let Some(pos) = existing_pos {
                                    // Append to existing tool call's arguments
                                    self.tool_calls[pos]
                                        .function
                                        .arguments
                                        .push_str(&tool_call.function.arguments);
                                } else {
                                    // Add new tool call
                                    self.tool_calls.push(tool_call.clone());
                                }

                                // Emit the tool call update
                                // Note: We emit it when complete (when it has both id and arguments)
                                if !tool_call.id.is_empty()
                                    && !tool_call.function.arguments.is_empty()
                                {
                                    return Some(Ok(StreamingUpdate::ToolCall(tool_call.clone())));
                                }
                            }
                        }
                    }
                }
                None
            }
            Err(e) => Some(Err(KongfuError::ResponseParseError(format!(
                "Failed to parse SSE chunk: {}",
                e
            )))),
        }
    }
}

impl futures::Stream for ZaiResponseStream {
    type Item = Result<StreamingUpdate>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        if self.is_done {
            return std::task::Poll::Ready(None);
        }

        while let Some(event_end) = self.buffer.find("\n\n") {
            let event = self.buffer.drain(..event_end + 2).collect::<String>();

            if let Some(result) = self.parse_sse_event(&event) {
                return std::task::Poll::Ready(Some(result));
            }
        }

        match std::pin::Pin::new(&mut self.byte_stream).poll_next(cx) {
            std::task::Poll::Ready(Some(Ok(bytes))) => {
                let chunk = String::from_utf8_lossy(bytes.as_ref());
                self.buffer.push_str(&chunk);

                if let Some(event_end) = self.buffer.find("\n\n") {
                    let event = self.buffer.drain(..event_end + 2).collect::<String>();
                    if let Some(result) = self.parse_sse_event(&event) {
                        return std::task::Poll::Ready(Some(result));
                    }
                }

                cx.waker().wake_by_ref();
                std::task::Poll::Pending
            }
            std::task::Poll::Ready(Some(Err(e))) => std::task::Poll::Ready(Some(Err(
                KongfuError::StreamError(format!("Stream error: {}", e)),
            ))),
            std::task::Poll::Ready(None) => std::task::Poll::Ready(None),
            std::task::Poll::Pending => std::task::Poll::Pending,
        }
    }
}

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

    #[test]
    fn test_zai_client_endpoint() {
        let client = ZaiClient::new(
            "test-key".to_string(),
            "https://api.test.com/v1/".to_string(),
        );

        assert_eq!(
            client.endpoint("chat/completions"),
            "https://api.test.com/v1/chat/completions"
        );
        assert_eq!(client.endpoint("/models"), "https://api.test.com/v1/models");
    }
}
