use crate::error::{KongfuError, Result};
use crate::message::{ContentBlock, ToolUseBlock};
use crate::provider::types::{ModelResponse, StreamingUpdate, ToolCall, Usage};
use futures::Stream;
use serde::de::DeserializeOwned;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::pin::Pin;

/// Trait that SSE chunks must implement to work with the generic stream parser
pub trait SseChunk: DeserializeOwned {
    /// Extract the model name from this chunk
    fn model(&self) -> &str;

    /// Extract the usage information from this chunk (if available)
    fn usage(&self) -> Option<Usage>;

    /// Extract the finish reason from this chunk (if available)
    fn finish_reason(&self) -> Option<&str>;

    /// Extract the thinking/reasoning delta from this chunk (if available)
    fn thinking_delta(&self) -> Option<&str>;

    /// Extract the content delta from this chunk (if available)
    fn content_delta(&self) -> Option<&str>;

    /// Extract tool call deltas from this chunk (if available)
    fn tool_call_deltas(&self) -> Option<&[ToolCall]>;
}

/// Generic SSE stream that handles the common streaming logic for all providers
///
/// This struct contains all the shared streaming logic including:
/// - Buffer management for SSE events
/// - Event parsing and dispatch
/// - State tracking (thinking content, response content, tool calls, etc.)
/// - Stream polling and completion handling
///
/// Each provider only needs to implement the `SseChunk` trait for their specific chunk type.
pub struct SseStream<C: SseChunk> {
    byte_stream: Pin<Box<dyn Stream<Item = reqwest::Result<bytes::Bytes>> + Send>>,
    buffer: String,
    is_done: bool,
    thinking_content: String,
    response_content: String,
    finish_reason: Option<String>,
    model: String,
    usage: Option<Usage>,
    tool_calls: Vec<ToolCall>,
    _phantom: PhantomData<C>,
}

impl<C: SseChunk> SseStream<C> {
    /// Create a new SSE stream from a byte stream
    pub fn new<S>(byte_stream: S, model: String) -> Self
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
            _phantom: PhantomData,
        }
    }

    /// Parse a single SSE event and return the corresponding update (if any)
    fn parse_sse_event(&mut self, event: &str) -> Option<Result<StreamingUpdate>> {
        let event = event.trim();
        if event.is_empty() {
            return None;
        }

        if !event.starts_with("data: ") {
            return None;
        }

        let data_str = &event[6..]; // trim "data: "

        // Handle the [DONE] sentinel that indicates stream completion
        if data_str.trim() == "[DONE]" {
            self.is_done = true;
            return Some(self.build_final_response());
        }

        // Parse the JSON chunk
        match serde_json::from_str::<C>(data_str) {
            Ok(chunk) => self.process_chunk(chunk),
            Err(e) => Some(Err(KongfuError::ResponseParseError(format!(
                "Failed to parse SSE chunk: {}",
                e
            )))),
        }
    }

    /// Process a parsed chunk and return the corresponding update (if any)
    fn process_chunk(&mut self, chunk: C) -> Option<Result<StreamingUpdate>> {
        // Update model from the first chunk
        if self.model.is_empty() {
            self.model = chunk.model().to_string();
        }

        // Update usage if available
        if let Some(usage) = chunk.usage() {
            self.usage = Some(usage);
        }

        // Update finish reason if available
        if let Some(reason) = chunk.finish_reason() {
            self.finish_reason = Some(reason.to_string());
        }

        // Handle thinking/reasoning content
        if let Some(thinking_delta) = chunk.thinking_delta() {
            if !thinking_delta.is_empty() {
                self.thinking_content.push_str(thinking_delta);
                return Some(Ok(StreamingUpdate::Thinking(thinking_delta.to_string())));
            }
        }

        // Handle regular content
        if let Some(content_delta) = chunk.content_delta() {
            if !content_delta.is_empty() {
                self.response_content.push_str(content_delta);
                return Some(Ok(StreamingUpdate::Content(content_delta.to_string())));
            }
        }

        // Handle tool calls
        if let Some(tool_call_deltas) = chunk.tool_call_deltas() {
            for tool_call in tool_call_deltas {
                // Check if we already have this tool call (by id)
                let existing_pos = self.tool_calls.iter().position(|tc| tc.id == tool_call.id);

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

                // Emit the tool call update when it has both id and arguments
                if !tool_call.id.is_empty() && !tool_call.function.arguments.is_empty() {
                    return Some(Ok(StreamingUpdate::ToolCall(tool_call.clone())));
                }
            }
        }

        None
    }

    /// Build the final response when stream completes
    fn build_final_response(&mut self) -> Result<StreamingUpdate> {
        let mut content_blocks = Vec::new();

        // Add thinking content if present
        if !self.thinking_content.is_empty() {
            content_blocks.push(ContentBlock::thinking(self.thinking_content.clone()));
        }

        // Add response content if present
        if !self.response_content.is_empty() {
            content_blocks.push(ContentBlock::text(self.response_content.clone()));
        }

        // Add tool calls
        for tool_call in &self.tool_calls {
            let args_map: HashMap<String, serde_json::Value> =
                match serde_json::from_str(&tool_call.function.arguments) {
                    Ok(map) => map,
                    Err(e) => {
                        return Err(KongfuError::ResponseParseError(format!(
                            "Failed to parse tool call arguments: {}",
                            e
                        )));
                    }
                };

            content_blocks.push(ContentBlock::ToolUse(ToolUseBlock::new(
                tool_call.id.clone(),
                tool_call.function.name.clone(),
                args_map,
            )));
        }

        // Ensure we have at least one content block
        if content_blocks.is_empty() {
            content_blocks.push(ContentBlock::text(String::new()));
        }

        let response = ModelResponse {
            content: content_blocks,
            model: self.model.clone(),
            usage: self.usage.take(),
            finish_reason: self.finish_reason.clone(),
        };

        Ok(StreamingUpdate::Done(response))
    }
}

impl<C: SseChunk + Unpin> futures::Stream for SseStream<C> {
    type Item = Result<StreamingUpdate>;

    fn poll_next(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        // Use pin projection to get mutable access to fields
        let this = self.get_mut();

        if this.is_done {
            return std::task::Poll::Ready(None);
        }

        // Process any complete events already in the buffer
        while let Some(event_end) = this.buffer.find("\n\n") {
            let event = this.buffer.drain(..event_end + 2).collect::<String>();

            if let Some(result) = this.parse_sse_event(&event) {
                return std::task::Poll::Ready(Some(result));
            }
        }

        // Read more data from the byte stream
        match std::pin::Pin::new(&mut this.byte_stream).poll_next(cx) {
            std::task::Poll::Ready(Some(Ok(bytes))) => {
                let chunk = String::from_utf8_lossy(bytes.as_ref());
                this.buffer.push_str(&chunk);

                // Try to parse complete events from the new data
                if let Some(event_end) = this.buffer.find("\n\n") {
                    let event = this.buffer.drain(..event_end + 2).collect::<String>();
                    if let Some(result) = this.parse_sse_event(&event) {
                        return std::task::Poll::Ready(Some(result));
                    }
                }

                // Request more data
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
