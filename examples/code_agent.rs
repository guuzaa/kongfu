use futures::StreamExt;
use kongfu::provider::Zai;
use kongfu::tools::{EditFile, ListDirectory, ReadFile};
use kongfu::{Agent, AgentEvent, Provider, StreamingAgent};
use std::io::Write;
use std::pin::pin;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║         🤖 AI Code Assistant with Editing                 ║");
    println!("║                                                            ║");
    println!("║  An intelligent coding assistant that can:                 ║");
    println!("║    • 📂 Explore codebase structure                         ║");
    println!("║    • 📖 Read and analyze source files                      ║");
    println!("║    • ✏️  Edit code with precision (line-based)              ║");
    println!("║    • 🔄 Iterate with multi-turn conversations              ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!();

    // Initialize Zai provider
    let zai = Zai::builder()
        .base_url("https://api.z.ai/api/coding/paas/v4")
        .model(std::env::var("ZAI_MODEL").unwrap_or_else(|_| "glm-4.7".to_string()))
        .build()
        .expect("Failed to create Zai provider. Check ZAI_API_KEY environment variable.");

    println!(
        "✓ Connected to {} provider (model: {})\n",
        zai.name(),
        zai.config().model
    );

    // Create a code assistant agent with the new builder pattern
    let agent = Agent::builder(zai)
        .system_prompt(
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
        )
        .tool(ListDirectory)
        .tool(ReadFile)
        .tool(EditFile)
        .max_steps(15) // Code editing may require more steps
        .memory_limit(100)
        .build();

    let mut agent = StreamingAgent::from(agent);

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
    println!("   Type 'quit', 'exit', or 'clear' to control the session\n");

    let mut input_buf = String::new();

    // Multi-turn conversation loop
    loop {
        // Prompt for user input
        if input_buf.is_empty() {
            print!("👤 Your query: ");
        } else {
            print!("\n💬 Continue (next question or 'q' to quit): ");
        }
        std::io::stdout().flush()?;

        input_buf.clear();
        std::io::stdin().read_line(&mut input_buf)?;
        let input = input_buf.trim().to_string();

        // Handle special commands
        match input.as_str() {
            "quit" | "exit" | "q" => {
                println!("\n👋 Goodbye!");
                break;
            }
            "clear" => {
                println!("\n🗑️  Conversation history cleared");
                agent.clear().await?;
            }
            "" => {
                println!("❌ Query cannot be empty");
            }
            _ => {
                // Run the agent with streaming for this turn
                println!("\n🤖 Assistant:");
                match agent.run(&input).await {
                    Ok(stream) => {
                        // Pin the stream
                        let mut stream = pin!(stream);

                        // Process streaming events
                        while let Some(event_result) = stream.as_mut().next().await {
                            match event_result {
                                Ok(AgentEvent::Thinking(chunk)) => {
                                    print!("\x1b[90m{}\x1b[0m", chunk);
                                    std::io::stdout().flush()?;
                                }
                                Ok(AgentEvent::Content(chunk)) => {
                                    print!("{}", chunk);
                                    std::io::stdout().flush()?;
                                }
                                Ok(AgentEvent::ToolCall(name)) => {
                                    println!("\n\n🔧 Tool Call: {}", name);
                                }
                                Ok(AgentEvent::Done(response)) => {
                                    if let Some(usage) = response.usage {
                                        println!(
                                            "\n\n📊 Tokens: {} prompt + {} completion = {} total ({} cached)",
                                            usage.prompt_tokens,
                                            usage.completion_tokens,
                                            usage.total_tokens,
                                            usage.cached_tokens
                                        );
                                    }
                                    println!("\n🔄 Steps taken: {}", response.steps_taken);
                                    if let Some(reason) = response.finish_reason {
                                        println!("   [Finished: {}]", reason);
                                    }
                                }
                                Err(e) => {
                                    println!("\n❌ Error: {}", e);
                                }
                                _ => {}
                            }
                        }
                    }
                    Err(e) => {
                        println!("\n❌ Error: {}", e);
                    }
                }
            }
        }
    }

    Ok(())
}
