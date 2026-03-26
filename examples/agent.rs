use kongfu::{
    ContentBlock, ListDirectory, Message, Provider, ReadFile, RequestOptions, ToolChoice,
    ToolRegistry, ToolResultContent, Zai,
};
use std::io::Write;

// ============================================================================
// Agent implementation
// ============================================================================

/// Run the agent loop with the new tool system
async fn run_agent(
    zai: &Zai,
    user_query: &str,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let tool_registry = ToolRegistry::new().add(ListDirectory).add(ReadFile);
    let tools = tool_registry.to_tools();

    let options = RequestOptions {
        tool_choice: Some(ToolChoice::Auto),
    };

    let mut messages = vec![
        Message::system(
            "You are a helpful AI assistant with access to tools. \
             When you need to explore the file system or read files, use the available tools. \
             Always explain what you're doing before using a tool. \
             After getting tool results, provide a clear summary of what you found.",
        ),
        Message::user(user_query),
    ];

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
        let response = zai.generate(&messages, Some(&tools), &options).await?;

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

                    if let Ok(params) = serde_json::to_string_pretty(&tool_use.input) {
                        println!("   Parameters: {}", params);
                    }

                    // Execute the tool using the registry
                    let result = tool_registry
                        .execute(&tool_use.name, serde_json::json!(tool_use.input))
                        .await;

                    // Prepare the tool result message
                    let tool_result = match result {
                        Ok(output) => {
                            println!("   ✅ Success");
                            let output_str = serde_json::to_string_pretty(&output)
                                .unwrap_or_else(|_| output.to_string());
                            if output_str.len() <= 200 {
                                println!("   Result: {}", output_str);
                            } else {
                                println!("   Result: ({} bytes)", output_str.len());
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

        // If we have text and no tool calls, we're done
        if !final_text.is_empty() {
            println!("\n🤖 Assistant: {}", final_text);
            println!("\n✅ Agent completed successfully");
            break;
        }

        // If we got here, something unexpected happened
        println!("\n⚠️  Empty response, stopping");
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

    // Initialize Zai provider
    let zai = Zai::builder()
        .base_url("https://api.z.ai/api/coding/paas/v4")
        .model(std::env::var("ZAI_MODEL").unwrap_or_else(|_| "glm-4.7".to_string()))
        .build()
        .expect("Failed to create Zai provider. Check ZAI_API_KEY environment variable.");

    println!(
        "✓ Connected to Zai provider (model: {})\n",
        zai.config().model
    );

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
    println!();

    // Get user input
    print!("👤 Your query: ");
    std::io::stdout().flush()?;

    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;
    let input = input.trim();

    if input.is_empty() {
        println!("❌ Query cannot be empty");
        return Ok(());
    }

    // Run the agent
    run_agent(&zai, input).await?;

    Ok(())
}
