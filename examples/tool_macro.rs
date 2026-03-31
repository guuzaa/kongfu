//! Demonstration of the #[tool] proc-macro ergonomic improvements
//!
//! This example shows defining tools with the #[tool] macro.

use kongfu::tool;

/// A simple greeting tool that says hello to the user
///
/// This tool generates a personalized greeting message.
#[tool]
async fn greet_user(
    /// The name of the user to greet
    name: String,
    /// Whether to include an exclamation mark
    enthusiastic: Option<bool>,
) -> Result<String, Box<dyn std::error::Error>> {
    let greeting = if enthusiastic.unwrap_or(false) {
        format!("Hello, {}! Great to see you!", name)
    } else {
        format!("Hello, {}.", name)
    };
    Ok(greeting)
}

/// Calculate the sum of two numbers
#[tool]
async fn add_numbers(
    /// The first number
    a: i64,
    /// The second number
    b: i64,
) -> Result<String, Box<dyn std::error::Error>> {
    let sum = a + b;
    Ok(format!("The sum of {} and {} is {}", a, b, sum))
}

/// Concatenate multiple strings with a separator
#[tool]
async fn join_strings(
    /// The strings to join
    strings: Vec<String>,
    /// The separator to use (defaults to ", ")
    separator: Option<String>,
) -> Result<String, Box<dyn std::error::Error>> {
    let sep = separator.unwrap_or_else(|| ", ".to_string());
    let result = strings.join(&sep);
    Ok(result)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Tool Macro Demo ===\n");

    // The macro generates:
    // 1. GreetUserParams struct
    // 2. GreetUser struct implementing ToolHandler
    //
    // Let's use them:

    use kongfu::tools::ToolHandler;

    let tool = GreetUser;
    let params = GreetUserParams {
        name: "Alice".to_string(),
        enthusiastic: Some(true),
    };

    let result = tool.execute(params).await?;
    println!("Greeting: {}\n", result);

    // Test add_numbers
    let add_tool = AddNumbers;
    let add_params = AddNumbersParams { a: 42, b: 58 };
    let add_result = add_tool.execute(add_params).await?;
    println!("Math: {}\n", add_result);

    // Test join_strings
    let join_tool = JoinStrings;
    let join_params = JoinStringsParams {
        strings: vec![
            "apple".to_string(),
            "banana".to_string(),
            "cherry".to_string(),
        ],
        separator: Some(" | ".to_string()),
    };
    let join_result = join_tool.execute(join_params).await?;
    println!("Join: {}\n", join_result);

    // You can also convert to FunctionDefinition for LLM APIs
    use kongfu::tools::ToFunctionDefinition;
    let def = GreetUser.to_function_definition();
    println!("Tool definition:\n{}", serde_json::to_string_pretty(&def)?);

    Ok(())
}
