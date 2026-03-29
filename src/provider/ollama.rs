use crate::error::{KongfuError, Result};
use crate::message::{ContentBlock, Message, Role, ToolUseBlock};
use crate::provider::types::{StreamingProvider, StreamingUpdate};
use crate::provider::{
    ModelConfig, ModelResponse, Provider, ProviderName, RequestOptions, Tool, Usage,
};
use async_trait::async_trait;
use futures::Stream;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;
use std::pin::Pin;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize)]
struct OllamaRequestMessage {
    role: Role,
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<serde_json::Value>>,
}

impl From<&Message> for OllamaRequestMessage {
    fn from(msg: &Message) -> Self {
        let role = msg.role.clone();

        match &msg.content {
            ContentBlock::Text(text) => Self {
                role,
                content: text.text.clone(),
                tool_calls: None,
            },
            ContentBlock::Thinking(thinking) => Self {
                role,
                content: thinking.thinking.clone(),
                tool_calls: None,
            },
            ContentBlock::ToolResult(result) => {
                // For Ollama, tool results don't have tool_call_id
                let content_str = match &result.content {
                    Some(crate::message::ToolResultContent::Text(s)) => s.clone(),
                    Some(crate::message::ToolResultContent::Objects(objs)) => {
                        serde_json::to_string(objs).unwrap_or_default()
                    }
                    None => String::new(),
                };
                Self {
                    role,
                    content: content_str,
                    tool_calls: None,
                }
            }
            ContentBlock::ToolUse(tool_use) => {
                // For Ollama, arguments should be an object, not a string
                let function_call = serde_json::json!({
                    "name": tool_use.name,
                    "arguments": tool_use.input  // Pass the object directly
                });

                let tool_call_json = serde_json::json!({
                    "id": tool_use.id,
                    "type": "function",
                    "function": function_call
                });

                Self {
                    role,
                    content: String::new(),
                    tool_calls: Some(vec![tool_call_json]),
                }
            }
        }
    }
}

struct OllamaClient {
    http: reqwest::Client,
    base_url: String,
}

impl OllamaClient {
    fn new(base_url: String) -> Self {
        Self {
            http: reqwest::Client::new(),
            base_url,
        }
    }

    fn endpoint(&self, path: &str) -> String {
        format!("{}/{}", self.base_url, path)
    }

    async fn post(&self, path: &str, body: &serde_json::Value) -> Result<reqwest::Response> {
        let response = self
            .http
            .post(self.endpoint(path))
            .header("Content-Type", "application/json")
            .json(body)
            .send()
            .await
            .map_err(|e| {
                KongfuError::ExecutionError(format!("Ollama API request failed: {}", e))
            })?;

        if !response.status().is_success() {
            let code = response.status().as_u16();
            let error_text = response.text().await.unwrap_or_default();
            return Err(KongfuError::ExecutionError(format!(
                "Ollama API error {}: {}",
                code, error_text
            )));
        }

        Ok(response)
    }
}

pub struct Ollama {
    config: ModelConfig,
    client: OllamaClient,
}

impl Ollama {
    pub fn new(config: ModelConfig) -> Self {
        let client = OllamaClient::new(config.base_url.clone());
        Self { config, client }
    }

    pub fn builder() -> OllamaBuilder {
        OllamaBuilder::new()
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn build_request_body(
        &self,
        messages: &[Message],
        tools: Option<&[Tool]>,
        stream: bool,
    ) -> serde_json::Value {
        // Convert messages to Ollama format (without tool_call_id)
        let ollama_messages: Vec<OllamaRequestMessage> =
            messages.iter().map(OllamaRequestMessage::from).collect();

        let mut body = json!({
            "model": self.config.model,
            "messages": ollama_messages,
            "stream": stream,
            "options": {
                "temperature": self.config.temperature
            }
        });

        if let Some(num_predict) = self.config.max_tokens {
            body["options"]["num_predict"] = json!(num_predict);
        }

        if let Some(top_p) = self.config.top_p {
            body["options"]["top_p"] = json!(top_p);
        }

        if let Some(tools) = tools {
            body["tools"] = json!(tools);
        }

        body
    }
}

#[derive(Default)]
pub struct OllamaBuilder {
    model: Option<String>,
    base_url: Option<String>,
    temperature: Option<f64>,
    max_tokens: Option<u32>,
    top_p: Option<f64>,
}

impl OllamaBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
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

    pub fn build(self) -> Ollama {
        let base_url = self
            .base_url
            .or_else(|| std::env::var("OLLAMA_BASE_URL").ok())
            .unwrap_or_else(|| "http://127.0.0.1:11434".to_string());

        let model = self.model.unwrap_or_else(|| "llama3.2".to_string());

        let config = ModelConfig {
            model,
            base_url,
            api_key: String::new(), // Ollama doesn't require API key
            temperature: self.temperature.unwrap_or(0.7),
            max_tokens: self.max_tokens,
            top_p: self.top_p,
        };

        let client = OllamaClient::new(config.base_url.clone());

        Ollama { config, client }
    }
}

#[derive(Debug, Deserialize)]
struct OllamaMessage {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    role: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<OllamaToolCall>>,
    #[serde(default)]
    thinking: Option<String>,
}

#[derive(Debug, Deserialize, Clone)]
struct OllamaToolCall {
    #[serde(default)]
    id: String,
    #[serde(default)]
    function: OllamaFunction,
}

#[derive(Debug, Deserialize, Clone, Default)]
struct OllamaFunction {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub arguments: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct OllamaResponse {
    message: OllamaMessage,
    model: String,
    #[serde(default)]
    done: bool,
    #[serde(default)]
    prompt_eval_count: Option<usize>,
    #[serde(default)]
    eval_count: Option<usize>,
    #[serde(default)]
    total_duration: Option<u64>,
    #[serde(default)]
    load_duration: Option<u64>,
    #[serde(default)]
    prompt_eval_duration: Option<u64>,
    #[serde(default)]
    eval_duration: Option<u64>,
    #[serde(default)]
    done_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OllamaStreamChunk {
    model: String,
    #[serde(default)]
    created_at: Option<String>,
    #[serde(default)]
    message: Option<OllamaMessage>,
    #[serde(default)]
    done: bool,
    #[serde(default)]
    prompt_eval_count: Option<usize>,
    #[serde(default)]
    eval_count: Option<usize>,
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    thinking: Option<String>,
    #[serde(default)]
    done_reason: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<OllamaToolCall>>,
}

/// Convert Ollama tool calls to standard ContentBlock format
fn convert_tool_calls(tool_calls: &[OllamaToolCall]) -> Vec<ContentBlock> {
    tool_calls
        .iter()
        .map(|tool_call| {
            let args_map: HashMap<String, serde_json::Value> =
                if tool_call.function.arguments.is_object() {
                    serde_json::from_value(tool_call.function.arguments.clone())
                        .unwrap_or_else(|_| HashMap::new())
                } else if tool_call.function.arguments.is_string() {
                    serde_json::from_str(tool_call.function.arguments.as_str().unwrap_or("{}"))
                        .unwrap_or_else(|_| HashMap::new())
                } else {
                    HashMap::new()
                };

            let tool_id = if tool_call.id.is_empty() {
                format!("call_{}", Uuid::new_v4())
            } else {
                tool_call.id.clone()
            };

            ContentBlock::ToolUse(ToolUseBlock::new(
                tool_id,
                tool_call.function.name.clone(),
                args_map,
            ))
        })
        .collect()
}

/// Build usage from prompt and completion counts
fn build_usage(prompt_eval: Option<usize>, eval: Option<usize>) -> Usage {
    let prompt_tokens = prompt_eval.unwrap_or(0);
    let completion_tokens = eval.unwrap_or(0);
    Usage {
        prompt_tokens,
        completion_tokens,
        cached_tokens: 0,
        total_tokens: prompt_tokens + completion_tokens,
    }
}

/// Ensure content_blocks has at least one element
fn ensure_content_not_empty(content_blocks: &mut Vec<ContentBlock>) {
    if content_blocks.is_empty() {
        content_blocks.push(ContentBlock::text(""));
    }
}

impl TryFrom<OllamaResponse> for ModelResponse {
    type Error = KongfuError;

    fn try_from(response: OllamaResponse) -> Result<Self> {
        let mut content_blocks = Vec::new();

        if let Some(thinking) = &response.message.thinking
            && !thinking.is_empty()
        {
            content_blocks.push(ContentBlock::thinking(thinking.clone()));
        }

        if let Some(text) = &response.message.content
            && !text.is_empty()
        {
            content_blocks.push(ContentBlock::text(text.clone()));
        }

        if let Some(tool_calls) = &response.message.tool_calls {
            content_blocks.extend(convert_tool_calls(tool_calls));
        }

        ensure_content_not_empty(&mut content_blocks);

        let usage = build_usage(response.prompt_eval_count, response.eval_count);

        Ok(Self {
            content: content_blocks,
            model: response.model,
            usage: Some(usage),
            finish_reason: response.done_reason,
        })
    }
}

struct OllamaResponseStream {
    byte_stream: Pin<Box<dyn Stream<Item = reqwest::Result<bytes::Bytes>> + Send>>,
    buffer: String,
    is_done: bool,
    response_content: String,
    finish_reason: Option<String>,
    model: String,
    tool_calls: Vec<OllamaToolCall>,
}

impl OllamaResponseStream {
    fn new<S>(byte_stream: S, model: String) -> Self
    where
        S: Stream<Item = reqwest::Result<bytes::Bytes>> + Send + 'static,
    {
        Self {
            byte_stream: Box::pin(byte_stream),
            buffer: String::new(),
            is_done: false,
            response_content: String::new(),
            finish_reason: None,
            model,
            tool_calls: Vec::new(),
        }
    }

    fn parse_line(&mut self, line: &str) -> Option<Result<StreamingUpdate>> {
        let line = line.trim();
        if line.is_empty() {
            return None;
        }

        match serde_json::from_str::<OllamaStreamChunk>(line) {
            Ok(chunk) => {
                // Update model from the first chunk
                if self.model.is_empty() {
                    self.model = chunk.model.clone();
                }

                if chunk.done {
                    self.is_done = true;
                    let mut content_blocks = Vec::new();

                    if !self.response_content.is_empty() {
                        content_blocks.push(ContentBlock::text(self.response_content.clone()));
                    }

                    // Convert Ollama tool calls to standard format
                    content_blocks.extend(convert_tool_calls(&self.tool_calls));

                    ensure_content_not_empty(&mut content_blocks);

                    let usage = build_usage(chunk.prompt_eval_count, chunk.eval_count);

                    let response = ModelResponse {
                        content: content_blocks,
                        model: self.model.clone(),
                        usage: Some(usage),
                        finish_reason: self.finish_reason.clone(),
                    };
                    return Some(Ok(StreamingUpdate::Done(response)));
                }

                // Handle content chunks
                if let Some(content) = &chunk.content
                    && !content.is_empty()
                {
                    self.response_content.push_str(content);
                    return Some(Ok(StreamingUpdate::Content(content.clone())));
                }

                // Handle thinking chunks
                if let Some(thinking) = &chunk.thinking
                    && !thinking.is_empty()
                {
                    return Some(Ok(StreamingUpdate::Thinking(thinking.clone())));
                }

                // Handle message content (for non-streaming content in stream)
                if let Some(message) = &chunk.message {
                    if let Some(content) = &message.content
                        && !content.is_empty()
                    {
                        self.response_content.push_str(content);
                        return Some(Ok(StreamingUpdate::Content(content.clone())));
                    }

                    // Handle thinking from message
                    if let Some(thinking) = &message.thinking
                        && !thinking.is_empty()
                    {
                        return Some(Ok(StreamingUpdate::Thinking(thinking.clone())));
                    }

                    // Handle tool calls from message
                    if let Some(tool_calls) = &message.tool_calls {
                        for ollama_tool_call in tool_calls {
                            // For Ollama, we just collect the tool calls
                            // They don't stream incrementally like OpenAI
                            self.tool_calls.push(ollama_tool_call.clone());
                        }
                    }
                }

                // Handle tool calls at chunk level (non-message format)
                if let Some(tool_calls) = &chunk.tool_calls {
                    for ollama_tool_call in tool_calls {
                        self.tool_calls.push(ollama_tool_call.clone());
                    }
                }

                None
            }
            Err(e) => Some(Err(KongfuError::ExecutionError(format!(
                "Failed to parse stream chunk: {}",
                e
            )))),
        }
    }
}

impl futures::Stream for OllamaResponseStream {
    type Item = Result<StreamingUpdate>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        if self.is_done {
            return std::task::Poll::Ready(None);
        }

        // Process any complete lines in the buffer
        while let Some(line_end) = self.buffer.find('\n') {
            let line = self.buffer.drain(..line_end + 1).collect::<String>();

            if let Some(result) = self.parse_line(&line) {
                return std::task::Poll::Ready(Some(result));
            }
        }

        match std::pin::Pin::new(&mut self.byte_stream).poll_next(cx) {
            std::task::Poll::Ready(Some(Ok(bytes))) => {
                let chunk = String::from_utf8_lossy(bytes.as_ref());
                self.buffer.push_str(&chunk);

                // Try to process lines again
                while let Some(line_end) = self.buffer.find('\n') {
                    let line = self.buffer.drain(..line_end + 1).collect::<String>();

                    if let Some(result) = self.parse_line(&line) {
                        return std::task::Poll::Ready(Some(result));
                    }
                }

                cx.waker().wake_by_ref();
                std::task::Poll::Pending
            }
            std::task::Poll::Ready(Some(Err(e))) => std::task::Poll::Ready(Some(Err(
                KongfuError::ExecutionError(format!("Stream error: {}", e)),
            ))),
            std::task::Poll::Ready(None) => std::task::Poll::Ready(None),
            std::task::Poll::Pending => std::task::Poll::Pending,
        }
    }
}

#[async_trait]
impl Provider for Ollama {
    fn name(&self) -> ProviderName {
        ProviderName::Ollama
    }

    async fn generate(
        &self,
        messages: &[Message],
        tools: Option<&[Tool]>,
        _options: &RequestOptions,
    ) -> Result<ModelResponse> {
        let body = self.build_request_body(messages, tools, false);
        let response = self.client.post("api/chat", &body).await?;

        let api_response: OllamaResponse = response.json().await.map_err(|e| {
            KongfuError::ExecutionError(format!("Failed to parse Ollama response: {}", e))
        })?;

        Ok(api_response.try_into()?)
    }
}

#[async_trait]
impl StreamingProvider for Ollama {
    async fn stream_generate(
        &self,
        messages: &[Message],
        tools: Option<&[Tool]>,
        _options: &RequestOptions,
    ) -> Result<Box<dyn futures::Stream<Item = Result<StreamingUpdate>> + Unpin + Send>> {
        let body = self.build_request_body(messages, tools, true);
        let response = self.client.post("api/chat", &body).await?;

        let byte_stream = response.bytes_stream();
        let stream = OllamaResponseStream::new(byte_stream, self.config.model.clone());

        Ok(Box::new(stream))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[tokio::test]
    #[ignore = "needs ollama server running"]
    async fn test_ollama_generate() {
        let ollama = Ollama::builder()
            .base_url("http://127.0.0.1:11434")
            .model("qwen3.5:9b")
            .temperature(0.7)
            .max_tokens(10_000)
            .build();

        let options = RequestOptions { tool_choice: None };
        let messages = vec![
            Message::system("You are a helpful AI helper"),
            Message::user("What's an LLM in short?"),
        ];

        let resp = ollama.generate(&messages, None, &options).await;
        match resp {
            Ok(response) => {
                let content = &response.content;
                if let Some(usage) = &response.usage {
                    println!("usage: {:?}", usage);
                }
                // Print first text block if any
                for block in content {
                    if let Some(text) = block.as_text() {
                        println!("text: {}", text);
                        break;
                    }
                }
            }
            Err(err) => {
                panic!("{}", err.to_string());
            }
        }
    }

    #[tokio::test]
    #[ignore = "needs ollama server running"]
    async fn test_ollama_stream_generate() {
        use futures::StreamExt;

        let ollama = Ollama::builder()
            .base_url("http://127.0.0.1:11434")
            .model("qwen3.5:9b")
            .temperature(0.7)
            .build();

        let options = RequestOptions { tool_choice: None };

        let messages = vec![
            Message::system("You are a helpful AI helper"),
            Message::user("What's an LLM in short?"),
        ];

        let mut stream = ollama
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
                    // Ollama doesn't typically support thinking
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
            if let Some(usage) = &response.usage {
                println!("Usage: {:?}", usage);
            }
        }

        assert!(content_count > 0, "Should receive at least one chunk");
    }

    #[test]
    fn test_build_request_body() {
        let ollama = Ollama::builder()
            .base_url("http://127.0.0.1:11434")
            .model("llama3.2")
            .temperature(0.7)
            .max_tokens(1000)
            .top_p(0.9)
            .build();

        let messages = vec![Message::user("Hello")];

        // Test with stream=false
        let body = ollama.build_request_body(&messages, None, false);

        assert_eq!(body["model"], "llama3.2");
        assert_eq!(body["stream"], false);
        assert_eq!(body["messages"].as_array().unwrap().len(), 1);
        assert_eq!(body["options"]["temperature"].as_f64().unwrap(), 0.7);
        assert_eq!(body["options"]["num_predict"], 1000);
        assert_eq!(body["options"]["top_p"].as_f64().unwrap(), 0.9);

        // Test with stream=true
        let body_stream = ollama.build_request_body(&messages, None, true);
        assert_eq!(body_stream["stream"], true);

        // Test without optional parameters
        let ollama_minimal = Ollama::builder().build();

        let body_minimal = ollama_minimal.build_request_body(&messages, None, false);

        assert_eq!(body_minimal["model"], "llama3.2");
        assert!((body_minimal["options"]["temperature"].as_f64().unwrap() - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_ollama_client_endpoint() {
        let client = OllamaClient::new("http://127.0.0.1:11434".to_string());

        assert_eq!(
            client.endpoint("api/chat"),
            "http://127.0.0.1:11434/api/chat"
        );
        assert_eq!(
            client.endpoint("api/tags"),
            "http://127.0.0.1:11434/api/tags"
        );
    }

    #[test]
    fn test_ollama_builder() {
        let ollama = Ollama::builder()
            .model("custom-model")
            .base_url("http://example.com:11434")
            .temperature(0.5)
            .max_tokens(500)
            .top_p(0.8)
            .build();

        assert_eq!(ollama.config().model, "custom-model");
        assert_eq!(ollama.config().base_url, "http://example.com:11434");
        assert_eq!(ollama.config().temperature, 0.5);
        assert_eq!(ollama.config().max_tokens, Some(500));
        assert_eq!(ollama.config().top_p, Some(0.8));
    }

    #[test]
    fn test_ollama_builder_defaults() {
        let ollama = Ollama::builder().build();

        assert_eq!(ollama.config().model, "llama3.2");
        assert_eq!(ollama.config().base_url, "http://127.0.0.1:11434");
        assert_eq!(ollama.config().temperature, 0.7);
        assert_eq!(ollama.config().max_tokens, None);
        assert_eq!(ollama.config().top_p, None);
    }

    #[test]
    fn test_tool_serialization() {
        use crate::provider::Tool;

        let tool = Tool::Function(crate::provider::FunctionDefinition {
            name: "list_directory".to_string(),
            description: "List files in a directory".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path to list"
                    }
                },
                "required": ["path"]
            }),
        });

        let serialized = serde_json::to_string_pretty(&tool).unwrap();

        // Test that it can be deserialized back
        let deserialized: Tool = serde_json::from_str(&serialized).unwrap();
        match deserialized {
            Tool::Function(func) => {
                assert_eq!(func.name, "list_directory");
            }
        }
    }

    #[test]
    fn test_ollama_message_format() {
        use crate::message::{ContentBlock, Message, Role, ToolResultContent};

        // Test that OllamaRequestMessage correctly formats tool messages (without tool_call_id)
        let tool_result_msg = Message::tool(ContentBlock::tool_result(
            "call_0",
            Some(ToolResultContent::Text("Result content".to_string())),
            Some(false),
        ));

        let ollama_msg: OllamaRequestMessage = (&tool_result_msg).into();

        // Verify tool_call_id is not included for Ollama compatibility
        assert_eq!(ollama_msg.role, Role::Tool);
        assert_eq!(ollama_msg.content, "Result content");
        assert!(ollama_msg.tool_calls.is_none());

        // Test tool_use message with arguments as object (not string)
        let mut input = std::collections::HashMap::new();
        input.insert("path".to_string(), serde_json::json!("."));

        let tool_use_msg =
            Message::assistant(ContentBlock::tool_use("call_0", "list_directory", input));

        let ollama_tool_use: OllamaRequestMessage = (&tool_use_msg).into();
        assert_eq!(ollama_tool_use.role, Role::Assistant);
        assert!(ollama_tool_use.tool_calls.is_some());

        // Verify arguments is an object, not a string
        if let Some(tool_calls) = ollama_tool_use.tool_calls {
            if let Some(function) = tool_calls[0].get("function") {
                if let Some(args) = function.get("arguments") {
                    assert!(
                        args.is_object(),
                        "arguments should be an object, not string"
                    );
                }
            }
        }
    }
}
