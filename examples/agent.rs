use kongfu::provider::Ollama;
use kongfu::tools::{ListDirectory, ReadFile};
use kongfu::{Agent, Provider};
use std::io::Write;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║     🤖 Agent with Type-Safe Tool System                    ║");
    println!("║                                                            ║");
    println!("║  Demonstrates the new Agent abstraction:                   ║");
    println!("║    • Clean builder pattern                                 ║");
    println!("║    • Type-safe parameters with validation                  ║");
    println!("║    • Auto-generated JSON schemas                           ║");
    println!("║    • Concurrent tool execution                             ║");
    println!("║    • Built-in memory management                            ║");
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

    // Create an agent with the new builder pattern
    let mut agent = Agent::builder(client)
        .system_prompt(
            "You are a helpful AI assistant with access to tools. \
             When you need to explore the file system or read files, use the available tools. \
             Always explain what you're doing before using a tool. \
             After getting tool results, provide a clear summary of what you found.",
        )
        .tool(ListDirectory)
        .tool(ReadFile)
        .max_steps(10)
        .memory_limit(50)
        .build();

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
                agent.clear().await?;
                continue;
            }
            "" => {
                println!("❌ Query cannot be empty");
                continue;
            }
            _ => {}
        }

        // Run the agent for this turn
        println!("\n🤖 Agent:");
        match agent.run(input).await {
            Ok(response) => {
                println!("\n{}", response.text);
                println!(
                    "\n📊 Stats: {} steps, {} tokens",
                    response.steps_taken,
                    response.usage.map_or(0, |u| u.total_tokens)
                );
            }
            Err(e) => {
                println!("\n❌ Error: {}", e);
            }
        }
    }

    Ok(())
}
