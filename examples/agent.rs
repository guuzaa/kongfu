use kongfu::{
    ContentBlock, ListDirectory, Message, Provider, ReadFile, RequestOptions, ToolChoice,
    ToolRegistry, ToolResultContent, provider::Ollama,
};
use std::io::Write;

// ============================================================================
// Agent implementation
// ============================================================================

/// Run the agent loop with the new tool system
async fn run_agent(
    provider: &dyn Provider,
    messages: &mut Vec<Message>,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let tool_registry = ToolRegistry::new().add(ListDirectory).add(ReadFile);
    let tools = tool_registry.to_tools();

    let options = RequestOptions {
        tool_choice: Some(ToolChoice::Auto),
    };

    let mut step = 0;
    let max_steps = 10;

    loop {
        if step >= max_steps {
            println!("\n⚠️  Maximum steps ({}) reached, stopping.", max_steps);
            break;
        }

        step += 1;
        println!("\n--- Step {} ---", step);

        // Call the LLM
        let response = provider.generate(&messages, Some(&tools), &options).await?;

        // Process all content blocks
        let mut has_tool_calls = false;
        let mut final_text = String::new();

        for block in &response.content {
            match block {
                ContentBlock::Thinking(thinking) => {
                    println!("\n🤔 Thinking: {}", thinking.thinking);
                }
                ContentBlock::Text(text) => {
                    final_text = text.text.clone();
                }
                ContentBlock::ToolUse(tool_use) => {
                    has_tool_calls = true;
                    println!("\n🔧 Tool Call: {}", tool_use.name);

                    // Execute the tool using the registry
                    let result = tool_registry
                        .execute(&tool_use.name, serde_json::json!(tool_use.input))
                        .await;

                    // Prepare the tool result message
                    let tool_result = match result {
                        Ok(output) => {
                            // Convert Value to appropriate string representation
                            let output_str = if output.is_string() {
                                // Extract string content directly
                                output.as_str().unwrap_or(&output.to_string()).to_string()
                            } else {
                                // For non-string values, format as JSON
                                serde_json::to_string_pretty(&output)
                                    .unwrap_or_else(|_| output.to_string())
                            };
                            if output_str.len() <= 200 {
                                println!("   Tool Result: {}", output_str);
                            } else {
                                println!("   Tool Result: ({} bytes)", output_str.len());
                            }
                            ContentBlock::tool_result(
                                &tool_use.id,
                                Some(ToolResultContent::Text(output_str)),
                                Some(false),
                            )
                        }
                        Err(error) => {
                            println!("   ❌ Error: {}", error);
                            ContentBlock::tool_result(
                                &tool_use.id,
                                Some(ToolResultContent::Text(error)),
                                Some(true),
                            )
                        }
                    };

                    // Add assistant message with the tool_use block
                    messages.push(Message::assistant(block.clone()));

                    // Add tool result message
                    messages.push(Message::tool(tool_result));
                }
                ContentBlock::ToolResult(_) => {
                    println!("\n⚠️  Unexpected tool result in response");
                }
            }
        }

        // If there were tool calls, continue the loop
        if has_tool_calls {
            continue;
        }

        // If we have text and no tool calls, we're done with this turn
        if !final_text.is_empty() {
            println!("\n🤖 Assistant: {}", final_text);
            // Add the assistant's response to messages
            messages.push(Message::assistant(final_text));
        }
        break;
    }

    Ok(())
}

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║     🤖 Agent with Type-Safe Tool System                    ║");
    println!("║                                                            ║");
    println!("║  Demonstrates the new ToolHandler + ToolParams system:     ║");
    println!("║    • Type-safe parameters with validation                  ║");
    println!("║    • Auto-generated JSON schemas                           ║");
    println!("║    • Clean separation of tool logic                         ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!();

    let client = Ollama::builder()
        .base_url("http://127.0.0.1:11434")
        .model("qwen3.5:9b")
        .temperature(0.7)
        .build();

    println!(
        "✓ Connected to {} provider (model: {})\n",
        client.name(),
        client.config().model
    );

    // Initialize conversation with system message
    let mut messages = vec![Message::system(
        "You are a helpful AI assistant with access to tools. \
             When you need to explore the file system or read files, use the available tools. \
             Always explain what you're doing before using a tool. \
             After getting tool results, provide a clear summary of what you found.",
    )];

    // Example queries to try:
    let example_queries = vec![
        "What files are in the current directory?",
        "Read the Cargo.toml file and tell me about this project",
        "Explore the src directory and list all Rust source files",
        "What's in the examples directory?",
    ];

    println!("📝 Example queries you could ask:");
    for (i, query) in example_queries.iter().enumerate() {
        println!("   {}. \"{}\"", i + 1, query);
    }
    println!("   Type 'quit', 'exit', or 'clear' to control the session");
    println!();

    // Multi-turn conversation loop
    loop {
        // Get user input
        print!("👤 Your query: ");
        std::io::stdout().flush()?;

        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        let input = input.trim();

        // Handle special commands
        match input {
            "quit" | "exit" => {
                println!("\n👋 Goodbye!");
                break;
            }
            "clear" => {
                println!("\n🗑️  Conversation history cleared");
                messages = vec![Message::system(
                    "You are a helpful AI assistant with access to tools. \
                         When you need to explore the file system or read files, use the available tools. \
                         Always explain what you're doing before using a tool. \
                         After getting tool results, provide a clear summary of what you found.",
                )];
                continue;
            }
            "" => {
                println!("❌ Query cannot be empty");
                continue;
            }
            _ => {}
        }

        // Add user message to conversation history
        messages.push(Message::user(input));

        // Run the agent for this turn
        if let Err(e) = run_agent(&client, &mut messages).await {
            println!("\n❌ Error: {}", e);
            // Remove the last user message if there was an error
            messages.pop();
        }
    }

    Ok(())
}
