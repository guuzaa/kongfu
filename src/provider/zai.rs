use crate::error::{KongfuError, Result};
use crate::message::Message;
use crate::provider::types::StreamingProvider;
use crate::provider::{ModelConfig, ModelResponse, Provider, ProviderName, RequestOptions, Usage};
use async_trait::async_trait;
use futures::Stream;
use serde::Deserialize;
use serde_json::json;
use std::pin::Pin;

pub struct Zai {
    config: ModelConfig,
    client: reqwest::Client,
}

impl Zai {
    pub fn new(config: ModelConfig) -> Self {
        Self {
            config,
            client: reqwest::Client::new(),
        }
    }

    pub fn builder() -> ZaiBuilder {
        ZaiBuilder::new()
    }
}

pub struct ZaiBuilder {
    model: Option<String>,
    api_key: Option<String>,
    base_url: Option<String>,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
    top_p: Option<f32>,
}

impl ZaiBuilder {
    pub fn new() -> Self {
        Self {
            model: None,
            api_key: None,
            base_url: None,
            temperature: None,
            max_tokens: None,
            top_p: None,
        }
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

    pub fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    pub fn top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }

    pub fn build(self) -> Result<Zai> {
        let api_key = self
            .api_key
            .or_else(|| std::env::var("ZAI_API_KEY").ok())
            .ok_or_else(|| KongfuError::ExecutionError(
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

        Ok(Zai {
            config,
            client: reqwest::Client::new(),
        })
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
    content: String,
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    role: Option<String>,
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
}

#[derive(Debug, Deserialize)]
struct ZaiStreamChoice {
    index: Option<usize>,
    delta: ZaiStreamDelta,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ZaiStreamDelta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    role: Option<String>,
    #[serde(default)]
    reasoning_content: Option<String>,
}

impl From<ZaiUsage> for Usage {
    fn from(usage: ZaiUsage) -> Self {
        Self {
            prompt_tokens: usage.prompt_tokens,
            completion_tokens: usage.completion_tokens,
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

        Ok(Self {
            content: choice.message.content.clone(),
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
}

impl ZaiResponseStream {
    fn new<S>(byte_stream: S) -> Self
    where
        S: Stream<Item = reqwest::Result<bytes::Bytes>> + Send + 'static,
    {
        Self {
            byte_stream: Box::pin(byte_stream),
            buffer: String::new(),
            is_done: false,
        }
    }

    fn parse_sse_event(&mut self, event: &str) -> Option<Result<String>> {
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
            return Some(Ok(String::new()));
        }

        match serde_json::from_str::<ZaiStreamChunk>(data_str) {
            Ok(chunk) => {
                if let Some(choice) = chunk.choices.first() {
                    if let Some(content) = &choice.delta.content {
                        if !content.is_empty() {
                            return Some(Ok(content.clone()));
                        }
                    }
                    if let Some(reasoning) = &choice.delta.reasoning_content {
                        if !reasoning.is_empty() {
                            return Some(Ok(reasoning.clone()));
                        }
                    }
                }
                None
            }
            Err(e) => Some(Err(KongfuError::ExecutionError(format!(
                "Failed to parse SSE chunk: {}",
                e
            )))),
        }
    }
}

impl futures::Stream for ZaiResponseStream {
    type Item = Result<String>;

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
                KongfuError::ExecutionError(format!("Stream error: {}", e)),
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
        options: &RequestOptions,
    ) -> Result<ModelResponse> {
        let url = format!("{}/chat/completions", self.config.base_url);

        let mut body = json!({
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "stream": options.stream,
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

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| KongfuError::ExecutionError(format!("Zai API request failed: {}", e)))?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(KongfuError::ExecutionError(format!(
                "Zai API error: {}",
                error_text
            )));
        }

        let api_response: ZaiResponse = response.json().await.map_err(|e| {
            KongfuError::ExecutionError(format!("Failed to parse Zai response: {}", e))
        })?;

        Ok(api_response.try_into()?)
    }
}

#[async_trait]
impl StreamingProvider for Zai {
    async fn stream_generate(
        &self,
        messages: &[Message],
        options: &RequestOptions,
    ) -> Result<Box<dyn futures::Stream<Item = Result<String>> + Unpin + Send>> {
        let url = format!("{}/chat/completions", self.config.base_url);

        let mut body = json!({
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "stream": options.stream,
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

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| KongfuError::ExecutionError(format!("Zai API request failed: {}", e)))?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(KongfuError::ExecutionError(format!(
                "Zai API error: {}",
                error_text
            )));
        }

        let byte_stream = response.bytes_stream();
        let stream = ZaiResponseStream::new(byte_stream);

        Ok(Box::new(stream))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let options = RequestOptions {
            stream: false,
            tool_choice: None,
        };
        let messages = vec![
            Message::system("You are a helpful AI helper"),
            Message::user("Explain what's an LLM in short?"),
        ];
        let resp = zai.generate(&messages, &options).await;
        match resp {
            Ok(response) => {
                let content = response.content;
                let usage = response.usage.unwrap();
                assert!(usage.total_tokens > 0);
                println!("content: {:?}", content);
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

        let options = RequestOptions {
            stream: true,
            tool_choice: None,
        };

        let messages = vec![
            Message::system("You are a helpful AI helper"),
            Message::user("Explain what's an LLM in short?"),
        ];

        let mut stream = zai.stream_generate(&messages, &options).await.unwrap();

        let mut full_content = String::new();
        let mut chunk_count = 0;
        let mut total_results = 0;

        while let Some(chunk_result) = stream.next().await {
            total_results += 1;
            match chunk_result {
                Ok(chunk) => {
                    println!("Received chunk: {:?}", chunk);
                    if !chunk.is_empty() {
                        chunk_count += 1;
                        full_content.push_str(&chunk);
                    }
                }
                Err(e) => {
                    eprintln!("Stream error: {}", e);
                }
            }
        }

        println!("Total results from stream: {}", total_results);
        println!("Non-empty chunks: {}", chunk_count);
        println!("Full content: {}", full_content);

        assert!(chunk_count > 0, "Should receive at least one chunk");
    }
}
