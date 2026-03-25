use kongfu::{
    ContentBlock, FunctionDefinition, Message, RequestOptions, StreamingProvider, Tool, ToolChoice,
    ToolResultContent, ToolUseBlock, Zai,
};
use serde_json::json;
use std::collections::HashMap;
use std::io::Write;

/// Define available tools for the agent
fn get_tools() -> Vec<Tool> {
    vec![
        Tool::Function(FunctionDefinition {
            name: "list_directory".to_string(),
            description:
                "List files and directories in a given path. Use this to explore the file system."
                    .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The directory path to list (default: current directory '.')"
                    }
                }
            }),
        }),
        Tool::Function(FunctionDefinition {
            name: "read_file".to_string(),
            description: "Read the contents of a file. Use this to examine file contents."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The path to the file to read"
                    }
                },
                "required": ["path"]
            }),
        }),
    ]
}

/// Execute a tool call and return the result
fn execute_tool(name: &str, input: &HashMap<String, serde_json::Value>) -> Result<String, String> {
    match name {
        "list_directory" => {
            let path = input.get("path").and_then(|v| v.as_str()).unwrap_or(".");

            std::fs::read_dir(path)
                .map(|entries| {
                    let mut result = String::new();
                    for entry in entries.filter_map(|e| e.ok()) {
                        let file_name = entry.file_name().to_string_lossy().to_string();
                        let file_type = if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                            "DIR"
                        } else {
                            "FILE"
                        };
                        result.push_str(&format!("  [{}] {}\n", file_type, file_name));
                    }
                    if result.is_empty() {
                        result = "(empty directory)".to_string();
                    }
                    result
                })
                .map_err(|e| format!("Failed to list directory: {}", e))
        }
        "read_file" => {
            let path = input
                .get("path")
                .and_then(|v| v.as_str())
                .ok_or("Missing 'path' parameter")?;

            std::fs::read_to_string(path)
                .map_err(|e| format!("Failed to read file '{}': {}", path, e))
        }
        _ => Err(format!("Unknown tool: {}", name)),
    }
}

/// Run the streaming agent loop
async fn run_streaming_agent(
    zai: &Zai,
    user_query: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    use futures::StreamExt;

    let tools = get_tools();
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
    let max_steps = 10; // Prevent infinite loops

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
                let result = execute_tool(tool_name, tool_args);

                let tool_result = match result {
                    Ok(output) => {
                        println!("   ✅ Success");
                        if output.len() <= 200 {
                            println!("   Result: {}", output);
                        } else {
                            println!("   Result: ({} bytes)", output.len());
                        }
                        ContentBlock::tool_result(
                            tool_id,
                            Some(ToolResultContent::Text(output)),
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

            println!("\n✅ Streaming agent completed successfully");
            break;
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
    println!("║         🤖 Streaming Tool-Using Agent                      ║");
    println!("║                                                            ║");
    println!("║  This agent uses STREAMING to explore files and dirs:      ║");
    println!("║    • list_directory - List files in a directory            ║");
    println!("║    • read_file - Read the contents of a file               ║");
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

    // Run the streaming agent
    run_streaming_agent(&zai, input).await?;

    Ok(())
}
