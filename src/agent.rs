//! Agent abstraction for autonomous LLM interactions with tools
//!
//! The Agent struct encapsulates the agentic loop pattern, providing a clean
//! interface for running multi-step conversations with tool use.

use crate::error::{KongfuError, Result};
use crate::memory::{Memory, MemoryStore};
use crate::message::{ContentBlock, Message, Role, ToolResultContent};
use crate::provider::{Provider, RequestOptions, Usage};
use crate::tools::{ToolHandler, ToolRegistry};
use futures::future::join_all;
use serde_json::Value;

// ============================================================================
// Response Types
// ============================================================================

/// Aggregated response from an agent run
///
/// Collects results across all steps of the agentic loop so callers
/// don't need to inspect ModelResponse directly.
#[derive(Debug, Clone)]
pub struct AgentResponse {
    /// Final text response from the assistant
    pub text: String,
    /// Thinking/reasoning content (for caching and inspection)
    pub thinking: String,
    /// Number of agentic loop steps taken
    pub steps_taken: usize,
    /// Token usage accumulated across all steps
    pub usage: Option<Usage>,
    /// Reason the conversation ended
    pub finish_reason: Option<String>,
}

/// Events emitted during streaming agent execution
#[derive(Debug, Clone)]
pub enum AgentEvent {
    /// Incremental thinking/reasoning content
    Thinking(String),
    /// Incremental text response content
    Content(String),
    /// Tool being invoked (name only)
    ToolCall(String),
    /// Tool output (truncated for display)
    ToolResult(String),
    /// Agent finished with final response
    Done(AgentResponse),
}

// ============================================================================
// Agent Struct (Non-Streaming)
// ============================================================================

/// Agent for autonomous LLM interactions with tools
///
/// The Agent encapsulates the agentic loop pattern:
/// 1. Receive user input
/// 2. Call LLM
/// 3. Execute tools if requested
/// 4. Repeat until final response
///
/// # Type Parameters
///
/// * `P` - Provider type (must implement Provider trait)
///
/// # Example
///
/// ```rust,ignore
/// use kongfu::{Agent, ListDirectory, ReadFile};
/// use kongfu::provider::Ollama;
///
/// let mut agent = Agent::builder(Ollama::builder().model("qwen3.5:9b").build())
///     .system_prompt("You are a helpful assistant with file access.")
///     .tool(ListDirectory)
///     .tool(ReadFile)
///     .max_steps(10)
///     .build();
///
/// let response = agent.run("What files are in src?").await?;
/// println!("{}", response.text);
/// ```
pub struct Agent<P: Provider> {
    provider: P,
    tools: ToolRegistry,
    memory: MemoryStore,
    options: RequestOptions,
    max_steps: Option<usize>,
    system_prompt: Option<String>,
}

impl<P: Provider> Agent<P> {
    /// Create a new agent builder
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let agent = Agent::builder(provider)
    ///     .system_prompt("You are helpful.")
    ///     .tool(MyTool)
    ///     .build();
    /// ```
    pub fn builder(provider: P) -> AgentBuilder<P> {
        AgentBuilder::new(provider)
    }

    /// Run one user turn through the full agentic loop
    ///
    /// This method:
    /// 1. Adds the user input to conversation history
    /// 2. Calls the LLM (may loop multiple times for tool use)
    /// 3. Returns the final response aggregated across all steps
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Provider call fails
    /// - Tool execution fails
    /// - Max steps limit is exceeded
    pub async fn run(&mut self, user_input: &str) -> Result<AgentResponse> {
        // Add system prompt to memory on first call if memory is empty
        if let Some(ref system_prompt) = self.system_prompt
            && self.memory.is_empty().await
        {
            self.memory
                .add(Message::system(system_prompt.as_str()))
                .await?;
        }

        // Add user message to memory
        self.memory.add(Message::user(user_input)).await?;

        let mut steps_taken = 0;
        let mut total_usage: Option<Usage> = None;
        let tools = self.tools.to_tools();

        loop {
            // Check step limit
            if let Some(max) = self.max_steps
                && steps_taken >= max
            {
                return Err(KongfuError::MaxStepsExceeded(max));
            }

            // Build message list: system prompt + conversation history
            let messages = self.history().await?;

            // Call provider
            let response = self
                .provider
                .generate(&messages, Some(&tools), &self.options)
                .await?;

            // Accumulate usage
            if let Some(usage) = response.usage {
                total_usage = Some(match total_usage {
                    Some(prev) => Usage {
                        prompt_tokens: prev.prompt_tokens + usage.prompt_tokens,
                        completion_tokens: prev.completion_tokens + usage.completion_tokens,
                        cached_tokens: prev.cached_tokens + usage.cached_tokens,
                        total_tokens: prev.total_tokens + usage.total_tokens,
                    },
                    None => usage,
                });
            }

            // Partition response into tool uses, final text, and thinking content
            let mut tool_uses = Vec::new();
            let mut final_text = String::new();
            let mut thinking_content = String::new();

            for block in response.content {
                match block {
                    ContentBlock::ToolUse(tool_use) => {
                        tool_uses.push(tool_use.clone());
                    }
                    ContentBlock::Text(text) => {
                        final_text = text.text;
                    }
                    ContentBlock::Thinking(thinking) => {
                        thinking_content.push_str(&thinking.thinking);
                        thinking_content.push('\n');
                    }
                    ContentBlock::ToolResult(_) => {
                        // Tool results shouldn't appear in assistant responses
                    }
                }
            }

            // Branch: no tool uses → we're done
            if tool_uses.is_empty() {
                let mut blocks = Vec::new();
                if !thinking_content.is_empty() {
                    blocks.push(ContentBlock::thinking(&thinking_content));
                }
                if !final_text.is_empty() {
                    blocks.push(ContentBlock::text(&final_text));
                }
                if !blocks.is_empty() {
                    self.memory
                        .add(Message::contents(Role::Assistant, blocks))
                        .await?;
                }

                return Ok(AgentResponse {
                    text: final_text,
                    thinking: thinking_content,
                    steps_taken: steps_taken + 1,
                    usage: total_usage,
                    finish_reason: response.finish_reason,
                });
            }

            // Branch: tool uses → execute and continue
            steps_taken += 1;

            // Add ONE assistant message with ALL tool-use blocks
            // This fixes the "one message per block" bug from the original examples
            let assistant_msg = tool_uses
                .iter()
                .fold(Message::assistant(ContentBlock::text("")), |msg, block| {
                    msg.push(ContentBlock::ToolUse(block.clone()))
                });
            self.memory.add(assistant_msg).await?;

            // Execute all tools CONCURRENTLY
            let tool_futures: Vec<_> = tool_uses
                .iter()
                .map(|tool_use| {
                    let tools = &self.tools;
                    let input = Value::Object(tool_use.input.clone().into_iter().collect());
                    async move {
                        (
                            tool_use.id.clone(),
                            tool_use.name.clone(),
                            tools.execute(&tool_use.name, input).await,
                        )
                    }
                })
                .collect();

            let tool_results = join_all(tool_futures).await;

            // Add tool result messages
            for (tool_use_id, _tool_name, result) in tool_results {
                let tool_result = match result {
                    Ok(output) => {
                        // Convert Value to string
                        let output_str = if output.is_string() {
                            output.as_str().unwrap_or(&output.to_string()).to_string()
                        } else {
                            serde_json::to_string(&output).unwrap_or_else(|_| output.to_string())
                        };
                        ContentBlock::tool_result(
                            &tool_use_id,
                            Some(ToolResultContent::Text(output_str)),
                            Some(false),
                        )
                    }
                    Err(error) => ContentBlock::tool_result(
                        &tool_use_id,
                        Some(ToolResultContent::Text(error)),
                        Some(true),
                    ),
                };
                self.memory.add(Message::tool(tool_result)).await?;
            }
        }
    }

    /// Clear conversation history, keeping system prompt and tools
    pub async fn clear(&mut self) -> Result<()> {
        self.memory.clear().await?;

        // Re-add system prompt if configured
        if let Some(ref prompt) = self.system_prompt {
            self.memory.add(Message::system(prompt.as_str())).await?;
        }

        Ok(())
    }

    /// Read the current conversation history
    pub async fn history(&self) -> Result<Vec<Message>> {
        self.memory.get_all().await
    }
}

// ============================================================================
// Agent Builder
// ============================================================================

/// Builder for creating Agent instances
///
/// # Example
///
/// ```rust,ignore
/// let agent = Agent::builder(provider)
///     .system_prompt("You are helpful.")
///     .tool(ListDirectory)
///     .tool(ReadFile)
///     .max_steps(10)
///     .memory_limit(50)
///     .build();
/// ```
pub struct AgentBuilder<P: Provider> {
    provider: P,
    tools: ToolRegistry,
    memory_limit: Option<usize>,
    options: RequestOptions,
    max_steps: Option<usize>,
    system_prompt: Option<String>,
}

impl<P: Provider> AgentBuilder<P> {
    fn new(provider: P) -> Self {
        Self {
            provider,
            tools: ToolRegistry::new(),
            memory_limit: None,
            options: RequestOptions::default(),
            max_steps: None,
            system_prompt: None,
        }
    }

    /// Register a tool with the agent
    pub fn tool<H: ToolHandler + 'static>(mut self, tool: H) -> Self {
        self.tools = self.tools.add(tool);
        self
    }

    /// Set the system prompt
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Set the maximum number of agentic loop steps
    pub fn max_steps(mut self, max: impl Into<Option<usize>>) -> Self {
        self.max_steps = max.into();
        self
    }

    /// Set the maximum number of messages to keep in memory
    pub fn memory_limit(mut self, limit: usize) -> Self {
        self.memory_limit = Some(limit);
        self
    }

    /// Set request options (e.g., tool_choice)
    pub fn options(mut self, options: RequestOptions) -> Self {
        self.options = options;
        self
    }

    /// Build the agent
    pub fn build(self) -> Agent<P> {
        let memory = match self.memory_limit {
            Some(limit) => MemoryStore::with_max_size(limit),
            None => MemoryStore::new(),
        };

        Agent {
            provider: self.provider,
            tools: self.tools,
            memory,
            options: self.options,
            max_steps: self.max_steps,
            system_prompt: self.system_prompt,
        }
    }
}

// ============================================================================
// Streaming Agent
// ============================================================================

/// Streaming variant of Agent for real-time responses
///
/// Only available for providers that implement StreamingProvider.
///
/// **Note:** The current implementation streams the LLM response but does not
/// execute tools automatically. For full tool support, use the non-streaming
/// `Agent::run` method.
///
/// # Example
///
/// ```rust,ignore
/// use kongfu::{StreamingAgent, ListDirectory};
/// use futures::StreamExt;
///
/// let mut agent = StreamingAgent::builder(provider)
///     .tool(ListDirectory)
///     .build();
///
/// let mut stream = agent.run("What files in src?").await?;
///
/// while let Some(event) = stream.next().await {
///     match event? {
///         AgentEvent::Thinking(chunk) => print!("\x1b[90m{}\x1b[0m", chunk),
///         AgentEvent::Content(chunk) => print!("{}", chunk),
///         AgentEvent::ToolCall(name) => println!("\n🔧 {}", name),
///         AgentEvent::Done(resp) => println!("\n[{} steps]", resp.steps_taken),
///     }
/// }
/// ```
pub struct StreamingAgent<P: crate::provider::StreamingProvider> {
    agent: Agent<P>,
}

impl<P: crate::provider::StreamingProvider> From<Agent<P>> for StreamingAgent<P> {
    fn from(agent: Agent<P>) -> Self {
        Self { agent }
    }
}

impl<P: crate::provider::StreamingProvider> StreamingAgent<P> {
    /// Create a new streaming agent builder
    pub fn builder(provider: P) -> AgentBuilder<P> {
        AgentBuilder::new(provider)
    }

    /// Clear conversation history, keeping system prompt and tools
    pub async fn clear(&mut self) -> Result<()> {
        self.agent.clear().await
    }

    /// Read the current conversation history
    pub async fn history(&self) -> Result<Vec<Message>> {
        self.agent.history().await
    }

    /// Run the agent with streaming output
    ///
    /// Returns a stream of AgentEvent values for real-time processing.
    ///
    /// This supports full agentic loop with tool execution, similar to `Agent::run`
    /// but with streaming output for better user experience.
    pub async fn run(
        &mut self,
        user_input: &str,
    ) -> Result<impl futures::Stream<Item = Result<AgentEvent>>> {
        use futures::StreamExt;

        // Add system prompt to memory on first call if memory is empty
        if let Some(ref prompt) = self.agent.system_prompt
            && self.agent.memory.is_empty().await
        {
            self.agent
                .memory
                .add(Message::system(prompt.as_str()))
                .await?;
        }

        // Add user message to memory
        self.agent.memory.add(Message::user(user_input)).await?;
        let tools = self.agent.tools.to_tools();

        // Create a custom stream that handles the agentic loop
        let stream = async_stream::stream! {
            let mut steps_taken = 0;
            let mut total_usage: Option<Usage> = None;

            loop {
                // Check step limit
                if let Some(max) = self.agent.max_steps && steps_taken >= max {
                    yield Err(KongfuError::MaxStepsExceeded(max));
                    return;
                }

                // Build messages
                let messages = match self.agent.history().await {
                    Ok(m) => m,
                    Err(e) => {
                        yield Err(e);
                        return;
                    }
                };

                // Start streaming from provider
                let mut provider_stream = match self.agent.provider
                    .stream_generate(&messages, Some(&tools), &self.agent.options)
                    .await
                {
                    Ok(s) => s,
                    Err(e) => {
                        yield Err(KongfuError::NetworkError(e.to_string()));
                        return;
                    }
                };

                let mut final_text = String::new();
                let mut thinking_content = String::new();
                let mut tool_calls = Vec::new();

                // Process streaming updates
                while let Some(update_result) = provider_stream.next().await {
                    let update = match update_result {
                        Ok(u) => u,
                        Err(e) => {
                            yield Err(KongfuError::StreamError(e.to_string()));
                            return;
                        }
                    };

                    match update {
                        crate::provider::StreamingUpdate::Thinking(chunk) => {
                            thinking_content.push_str(&chunk);
                            thinking_content.push('\n');
                            yield Ok(AgentEvent::Thinking(chunk));
                        }
                        crate::provider::StreamingUpdate::Content(chunk) => {
                            yield Ok(AgentEvent::Content(chunk.clone()));
                            final_text.push_str(&chunk);
                        }
                        crate::provider::StreamingUpdate::ToolCall(tool_call) => {
                            yield Ok(AgentEvent::ToolCall(tool_call.function.name.clone()));
                            tool_calls.push(tool_call);
                        }
                        crate::provider::StreamingUpdate::Done(response) => {
                            // Accumulate usage
                            if let Some(usage) = response.usage {
                                total_usage = Some(match total_usage {
                                    Some(prev) => Usage {
                                        prompt_tokens: prev.prompt_tokens + usage.prompt_tokens,
                                        completion_tokens: prev.completion_tokens + usage.completion_tokens,
                                        cached_tokens: prev.cached_tokens + usage.cached_tokens,
                                        total_tokens: prev.total_tokens + usage.total_tokens,
                                    },
                                    None => usage,
                                });
                            }
                            break;
                        }
                    }
                }

                // Branch: no tool calls → we're done
                if tool_calls.is_empty() {
                    // Add assistant message to memory with thinking content for KV cache
                    // Order: thinking first, then final text
                    if !thinking_content.is_empty() || !final_text.is_empty() {
                        let mut blocks = Vec::new();
                        if !thinking_content.is_empty() {
                            blocks.push(ContentBlock::thinking(&thinking_content));
                        }
                        if !final_text.is_empty() {
                            blocks.push(ContentBlock::text(&final_text));
                        }
                        if let Err(e) = self.agent.memory.add(Message::contents(Role::Assistant, blocks)).await {
                            yield Err(e);
                            return;
                        }
                    }

                    yield Ok(AgentEvent::Done(AgentResponse {
                        text: final_text,
                        thinking: thinking_content,
                        steps_taken: steps_taken + 1,
                        usage: total_usage,
                        finish_reason: None,
                    }));
                    return;
                }

                // Branch: tool calls → execute and continue
                steps_taken += 1;

                // Add assistant message with tool calls to memory
                for tool_call in &tool_calls {
                    // Parse arguments
                    let args_value: serde_json::Value = serde_json::from_str(&tool_call.function.arguments)
                        .unwrap_or_else(|_| serde_json::Value::Object(Default::default()));

                    let args_map: std::collections::HashMap<String, serde_json::Value> = args_value
                        .as_object()
                        .map(|obj| obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
                        .unwrap_or_default();

                    if let Err(e) = self.agent.memory.add(
                        Message::assistant(ContentBlock::ToolUse(crate::message::ToolUseBlock {
                            id: tool_call.id.clone(),
                            name: tool_call.function.name.clone(),
                            input: args_map,
                        }))
                    ).await {
                        yield Err(e);
                        return;
                    }
                }

                // Execute all tools CONCURRENTLY
                let tool_futures: Vec<_> = tool_calls
                    .iter()
                    .map(|tool_call| {
                        let tools = &self.agent.tools;
                        let args_str = &tool_call.function.arguments;
                        async move {
                            let input: serde_json::Value = serde_json::from_str(args_str)
                                .unwrap_or_else(|_| serde_json::Value::Object(Default::default()));

                            let result = tools.execute(&tool_call.function.name, input).await;

                            (tool_call.id.clone(), tool_call.function.name.clone(), result)
                        }
                    })
                    .collect();

                let tool_results = join_all(tool_futures).await;

                // Add tool result messages to memory
                for (tool_use_id, _tool_name, result) in tool_results {
                    let tool_result = match result {
                        Ok(output) => {
                            let output_str = if output.is_string() {
                                output.as_str().unwrap_or(&output.to_string()).to_string()
                            } else {
                                serde_json::to_string(&output).unwrap_or_else(|_| output.to_string())
                            };
                            ContentBlock::tool_result(
                                &tool_use_id,
                                Some(ToolResultContent::Text(output_str)),
                                Some(false),
                            )
                        }
                        Err(error) => ContentBlock::tool_result(
                            &tool_use_id,
                            Some(ToolResultContent::Text(error)),
                            Some(true),
                        ),
                    };

                    if let Err(e) = self.agent.memory.add(Message::tool(tool_result)).await {
                        yield Err(e);
                        return;
                    }
                }
            }
        };

        Ok(stream)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_response() {
        let response = AgentResponse {
            text: "Hello".to_string(),
            thinking: String::new(),
            steps_taken: 2,
            usage: None,
            finish_reason: Some("stop".to_string()),
        };

        assert_eq!(response.text, "Hello");
        assert_eq!(response.steps_taken, 2);
    }
}
