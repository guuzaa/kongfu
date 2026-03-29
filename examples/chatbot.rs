use kongfu::{Message, RequestOptions, Role, StreamingProvider, StreamingUpdate, provider::Zai};
use std::io::{self, Write};

fn print_welcome() {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║          🤖 Multi-Round Chatbot (Streaming)               ║");
    println!("║                                                            ║");
    println!("║  Commands:                                                 ║");
    println!("║    /quit   - Exit the chatbot                              ║");
    println!("║    /clear  - Clear conversation history                    ║");
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

    let options = RequestOptions { tool_choice: None };

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
            "" => continue,
            _ => {}
        }

        // Add user message to history
        messages.push(Message::user(input));

        // Generate streaming response
        match zai.stream_generate(&messages, None, &options).await {
            Ok(mut stream) => {
                use futures::StreamExt;

                print!("🤖 Assistant: ");
                io::stdout().flush()?;
                let mut thinking_cnt = 0;
                let mut resp_cnt = 0;

                let mut thinking_content = String::new();
                let mut response_content = String::new();
                let mut final_response = None;

                while let Some(update_result) = stream.next().await {
                    match update_result {
                        Ok(StreamingUpdate::Thinking(chunk)) => {
                            thinking_cnt += 1;
                            thinking_content.push_str(&chunk);
                            if thinking_cnt == 1 {
                                print!("\x1b[90m<Thinking>:\x1b[0m\n{}", chunk);
                            } else {
                                print!("{}", chunk);
                            }
                            io::stdout().flush().unwrap();
                        }
                        Ok(StreamingUpdate::Content(chunk)) => {
                            resp_cnt += 1;
                            response_content.push_str(&chunk);
                            if thinking_cnt > 0 && resp_cnt == 1 {
                                print!("\n\x1b[90m</End of Thinking>:\x1b[0m\n{}", chunk);
                            } else {
                                print!("{}", chunk);
                            }
                            io::stdout().flush().unwrap();
                        }
                        Ok(StreamingUpdate::Done(response)) => {
                            final_response = Some(response);
                        }
                        Err(e) => {
                            eprintln!("\n❌ Stream error: {}", e);
                        }
                        _ => {}
                    }
                }

                println!(); // End the line after streaming

                // Add assistant response to history
                if let Some(response) = final_response {
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

                    response
                        .content
                        .into_iter()
                        .map(|block| Message::assistant(block))
                        .for_each(|msg| messages.push(msg));

                    if let Some(reason) = &response.finish_reason {
                        println!("   [Finished: {}]", reason);
                    }
                }
                println!();
            }
            Err(e) => {
                eprintln!("❌ Error: {}", e);
                eprintln!("   Please check your connection and API key.\n");
            }
        }
    }

    Ok(())
}
