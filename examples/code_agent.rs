use kongfu::tooling::{EditFile, ListDirectory, ReadFile, ToolRegistry};
use kongfu::{
    ContentBlock, Message, RequestOptions, StreamingProvider, ToolChoice, ToolResultContent,
    ToolUseBlock, provider::Zai,
};
use std::collections::HashMap;
use std::io::Write;

/// Run the code assistant agent loop with streaming support
async fn run_code_agent(zai: &Zai, user_query: &str) -> Result<(), Box<dyn std::error::Error>> {
    use futures::StreamExt;

    // Register all available tools for code editing
    let tool_registry = ToolRegistry::new()
        .add(ListDirectory)
        .add(ReadFile)
        .add(EditFile);
    let tools = tool_registry.to_tools();

    let options = RequestOptions {
        tool_choice: Some(ToolChoice::Auto),
    };

    let mut messages = vec![
        Message::system(
            "You are an expert code assistant with access to file system tools. \
             Your capabilities include:\n\
             - 📂 Exploring directory structures with list_directory\n\
             - 📖 Reading file contents with read_file\n\
             - ✏️  Editing files by line ranges with edit_file\n\n\
             **Workflow Guidelines:**\n\
             1. Always read a file before editing to understand its current state\n\
             2. Use list_directory to explore the codebase structure\n\
             3. When using edit_file, always specify the exact line range (1-indexed, inclusive)\n\
             4. Provide clear explanations before and after making changes\n\
             5. Show diff-like summaries when editing code\n\n\
             **Best Practices:**\n\
             - Start by exploring the codebase structure\n\
             - Read related files to understand context\n\
             - Make targeted, well-explained edits\n\
             - Verify changes by reading modified files\n\
             - Always maintain code quality and consistency",
        ),
        Message::user(user_query),
    ];

    let mut step = 0;
    let max_steps = 15; // Code editing may require more steps

    loop {
        if step >= max_steps {
            println!("\n⚠️  Maximum steps ({}) reached, stopping.", max_steps);
            break;
        }

        step += 1;
        println!("\n--- Step {} ---", step);

        // Call the LLM with streaming
        let mut stream = zai
            .stream_generate(&messages, Some(&tools), &options)
            .await?;

        let mut current_message_blocks = Vec::new();
        let mut has_tool_calls = false;
        let mut tool_calls_to_execute = Vec::new();

        // Track thinking and content for display
        let mut thinking_cnt = 0;
        let mut resp_cnt = 0;
        let mut thinking_content = String::new();
        let mut response_content = String::new();
        let mut final_response = None;

        // Process the stream
        while let Some(update_result) = stream.next().await {
            match update_result {
                Ok(kongfu::StreamingUpdate::Thinking(chunk)) => {
                    thinking_cnt += 1;
                    thinking_content.push_str(&chunk);
                    if thinking_cnt == 1 {
                        print!("\n\x1b[90m<Thinking>:\x1b[0m\n{}", chunk);
                    } else {
                        print!("{}", chunk);
                    }
                    std::io::stdout().flush().unwrap();
                }
                Ok(kongfu::StreamingUpdate::Content(chunk)) => {
                    resp_cnt += 1;
                    response_content.push_str(&chunk);
                    if thinking_cnt > 0 && resp_cnt == 1 {
                        print!("\n\x1b[90m</End of Thinking>\x1b[0m\n{}", chunk);
                    } else {
                        print!("{}", chunk);
                    }
                    std::io::stdout().flush().unwrap();
                }
                Ok(kongfu::StreamingUpdate::ToolCall(tool_call)) => {
                    has_tool_calls = true;
                    println!("\n🔧 Tool Call: {}", tool_call.function.name);

                    // Parse the arguments
                    let args_map: HashMap<String, serde_json::Value> =
                        serde_json::from_str(&tool_call.function.arguments)
                            .map_err(|e| format!("Failed to parse tool arguments: {}", e))?;

                    if let Some(params) = serde_json::to_string_pretty(&args_map).ok() {
                        println!("   Parameters: {}", params);
                    }

                    tool_calls_to_execute.push((
                        tool_call.id.clone(),
                        tool_call.function.name.clone(),
                        args_map.clone(),
                    ));

                    // Create a ContentBlock for this tool call
                    let tool_use_block = ContentBlock::ToolUse(ToolUseBlock::new(
                        tool_call.id.clone(),
                        tool_call.function.name.clone(),
                        args_map,
                    ));
                    current_message_blocks.push(tool_use_block);
                }
                Ok(kongfu::StreamingUpdate::Done(response)) => {
                    final_response = Some(response);
                    println!("\n\n[Stream complete]");
                }
                Err(e) => {
                    eprintln!("\nStream error: {}", e);
                }
            }
        }

        // Execute tool calls if any
        if has_tool_calls {
            // Add all tool_use blocks as separate assistant messages
            for block in &current_message_blocks {
                if let ContentBlock::ToolUse(_) = block {
                    messages.push(Message::assistant(block.clone()));
                }
            }

            // Execute each tool call and add results
            for (tool_id, tool_name, tool_args) in &tool_calls_to_execute {
                // Execute the tool using the registry
                let result = tool_registry
                    .execute(tool_name, serde_json::json!(tool_args))
                    .await;

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
                            tool_id,
                            Some(ToolResultContent::Text(output_str)),
                            Some(false),
                        )
                    }
                    Err(error) => {
                        println!("   ❌ Error: {}", error);
                        ContentBlock::tool_result(
                            tool_id,
                            Some(ToolResultContent::Text(error)),
                            Some(true),
                        )
                    }
                };

                // Add tool result message
                messages.push(Message::tool(tool_result));
            }

            // Continue to next iteration
            continue;
        }

        // Get the final response
        if let Some(response) = final_response {
            // Update usage statistics if available
            if let Some(usage) = &response.usage {
                println!(
                    "   [Tokens: {} prompt + {} completion = {} total ({} cached)]",
                    usage.prompt_tokens,
                    usage.completion_tokens,
                    usage.total_tokens,
                    usage.cached_tokens
                );
            }

            // Add assistant response to history
            for block in &response.content {
                messages.push(Message::assistant(block.clone()));
            }

            if let Some(reason) = &response.finish_reason {
                println!("   [Finished: {}]", reason);
            }

            // Ask user if they want to continue
            print!("\n💬 Continue? (Enter next question or 'q' to quit): ");
            std::io::stdout().flush()?;

            let mut next_input = String::new();
            std::io::stdin().read_line(&mut next_input)?;
            let next_input = next_input.trim();

            // Check if user wants to quit
            if next_input.eq_ignore_ascii_case("q") || next_input.eq_ignore_ascii_case("quit") {
                println!("\n👋 Goodbye!");
                break;
            }

            // Add new user message and continue the loop
            messages.push(Message::user(next_input.to_string()));
            continue;
        }

        // If we got here, something unexpected happened
        println!("\n⚠️  Empty response, stopping");
        break;
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║         🤖 AI Code Assistant with Editing                 ║");
    println!("║                                                            ║");
    println!("║  An intelligent coding assistant that can:                 ║");
    println!("║    • 📂 Explore codebase structure                         ║");
    println!("║    • 📖 Read and analyze source files                      ║");
    println!("║    • ✏️  Edit code with precision (line-based)              ║");
    println!("║    • 🔄 Iterate with streaming responses                   ║");
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

    // Example queries focused on code editing
    let example_queries = vec![
        "Show me the structure of the src directory",
        "Read the Cargo.toml file and tell me the dependencies",
        "Add a new function to examples/agent.rs that prints 'Hello, World!'",
        "Edit the main function in examples/code_agent.rs to add better error handling",
        "What Rust source files are in the src/tooling directory?",
        "Read and improve the comments in src/tooling/mod.rs",
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

    // Run the code agent
    run_code_agent(&zai, input).await?;

    Ok(())
}
