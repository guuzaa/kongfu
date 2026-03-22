use kongfu::{Message, Provider, RequestOptions, Role, Zai};
use std::io::{self, Write};

fn print_welcome() {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║                  🤖 Multi-Round Chatbot                    ║");
    println!("║                                                            ║");
    println!("║  Commands:                                                 ║");
    println!("║    /quit   - Exit the chatbot                              ║");
    println!("║    /clear  - Clear conversation history                    ║");
    println!("║    /stats  - Show token usage statistics                   ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!();
}

fn print_usage(total_tokens: usize, total_cost: f64) {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("📊 Session Statistics:");
    println!("   Total Tokens: {}", total_tokens);
    println!("   Estimated Cost: ${:.6}", total_cost);
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    print_welcome();

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

    let options = RequestOptions {
        stream: false,
        tool_choice: None,
    };

    let mut messages = vec![Message::system(
        "You are a helpful, friendly, and knowledgeable AI assistant. \
         Provide clear and concise answers. Be conversational and engaging.",
    )];

    let mut total_tokens = 0usize;
    let cost_per_1k_tokens = 0.0001;

    loop {
        // Get user input
        print!("👤 You: ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;

        let input = input.trim();

        // Handle commands
        match input {
            "/quit" | "/exit" | "/q" => {
                println!("\n👋 Goodbye! Have a great day!\n");
                print_usage(
                    total_tokens,
                    total_tokens as f64 * cost_per_1k_tokens / 1000.0,
                );
                break;
            }
            "/clear" => {
                messages = vec![Message::system(
                    "You are a helpful, friendly, and knowledgeable AI assistant. \
                     Provide clear and concise answers. Be conversational and engaging.",
                )];
                println!("✓ Conversation history cleared\n");
                continue;
            }
            "/stats" => {
                print_usage(
                    total_tokens,
                    total_tokens as f64 * cost_per_1k_tokens / 1000.0,
                );
                continue;
            }
            "" => continue,
            _ => {}
        }

        // Add user message to history
        messages.push(Message::user(input));

        // Generate response
        match zai.generate(&messages, &options).await {
            Ok(response) => {
                // Update usage statistics
                if let Some(usage) = &response.usage {
                    total_tokens += usage.total_tokens;
                    println!(
                        "   [Tokens: {} prompt + {} completion = {} total ({} cached)]",
                        usage.prompt_tokens,
                        usage.completion_tokens,
                        usage.total_tokens,
                        usage.cached_tokens
                    );
                }

                // Extract and display text content
                let text = response.content.as_text().unwrap_or("".to_string());
                println!("🤖 Assistant: {}", text);
                println!();

                let message = Message::new(Role::Assistant, response.content);
                // Add assistant response to history
                messages.push(message);
            }
            Err(e) => {
                eprintln!("❌ Error: {}", e);
                eprintln!("   Please check your connection and API key.\n");
            }
        }
    }

    Ok(())
}
