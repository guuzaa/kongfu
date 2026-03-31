use kongfu::tools::{ToFunctionDefinition, ToolHandler};
use kongfu_macros::tool;

/// A simple echo tool
#[tool]
async fn echo(message: String) -> Result<String, Box<dyn std::error::Error>> {
    Ok(message)
}

#[test]
fn test_single_parameter_tool() {
    let tool = Echo;

    assert_eq!(tool.name(), "echo");
    assert!(tool.description().contains("echo tool"));

    // Check that the schema is generated correctly
    let schema = tool.schema();
    assert_eq!(schema["type"], "object");
    assert!(schema["properties"]["message"].is_object());
    assert_eq!(schema["properties"]["message"]["type"], "string");
    assert!(
        schema["required"]
            .as_array()
            .unwrap()
            .contains(&"message".into())
    );
}

#[tokio::test]
async fn test_single_parameter_execution() {
    let tool = Echo;
    let params = EchoParams {
        message: "Hello, world!".to_string(),
    };

    let result = tool.execute(params).await.unwrap();
    assert_eq!(result, "Hello, world!");
}

/// A tool that processes multiple parameters
#[tool]
async fn multi_param(
    text: String,
    count: i32,
    flag: bool,
    rate: Option<f64>,
) -> Result<String, Box<dyn std::error::Error>> {
    let rate = rate.unwrap_or(1.0);
    Ok(format!(
        "{} x{} (flag: {}, rate: {})",
        text, count, flag, rate
    ))
}

#[test]
fn test_multiple_parameters_tool() {
    let tool = MultiParam;

    assert_eq!(tool.name(), "multi_param");

    // Check schema generation for all types
    let schema = tool.schema();
    assert_eq!(schema["properties"]["text"]["type"], "string");
    assert_eq!(schema["properties"]["count"]["type"], "integer");
    assert_eq!(schema["properties"]["flag"]["type"], "boolean");
    assert_eq!(schema["properties"]["rate"]["type"], "number");

    // Check required vs optional
    let required = schema["required"].as_array().unwrap();
    assert!(required.contains(&"text".into()));
    assert!(required.contains(&"count".into()));
    assert!(required.contains(&"flag".into()));
    assert!(!required.contains(&"rate".into())); // rate is optional
}

#[tokio::test]
async fn test_multiple_parameters_execution() {
    let tool = MultiParam;

    // Test with all parameters
    let params = MultiParamParams {
        text: "test".to_string(),
        count: 5,
        flag: true,
        rate: Some(2.5),
    };
    let result = tool.execute(params).await.unwrap();
    assert_eq!(result, "test x5 (flag: true, rate: 2.5)");

    // Test with optional parameter omitted
    let params = MultiParamParams {
        text: "test".to_string(),
        count: 3,
        flag: false,
        rate: None,
    };
    let result = tool.execute(params).await.unwrap();
    assert_eq!(result, "test x3 (flag: false, rate: 1)");
}

/// Sum a list of numbers
#[tool]
async fn sum_numbers(numbers: Vec<i64>) -> Result<String, Box<dyn std::error::Error>> {
    let sum: i64 = numbers.iter().sum();
    Ok(format!("Sum: {}", sum))
}

#[tokio::test]
async fn test_vec_parameter() {
    let tool = SumNumbers;
    let params = SumNumbersParams {
        numbers: vec![1, 2, 3, 4, 5],
    };

    let result = tool.execute(params).await.unwrap();
    assert_eq!(result, "Sum: 15");
}

/// This tool should have a custom name
#[tool(name = "custom_calculator")]
async fn add(a: i64, b: i64) -> Result<String, Box<dyn std::error::Error>> {
    Ok(format!("Result: {}", a + b))
}

#[test]
fn test_custom_tool_name() {
    let tool = Add;
    assert_eq!(tool.name(), "custom_calculator");
}

#[test]
fn test_to_function_definition() {
    let tool = Echo;
    let def = tool.to_function_definition();

    assert_eq!(def.name, "echo");
    assert!(def.description.contains("echo"));
    assert_eq!(def.parameters["type"], "object");
    assert!(def.parameters["properties"]["message"].is_object());
}

#[test]
fn test_multi_param_to_function_definition() {
    let tool = MultiParam;
    let def = tool.to_function_definition();

    assert_eq!(def.name, "multi_param");

    // Verify all parameters are in the definition
    assert!(def.parameters["properties"]["text"].is_object());
    assert!(def.parameters["properties"]["count"].is_object());
    assert!(def.parameters["properties"]["flag"].is_object());
    assert!(def.parameters["properties"]["rate"].is_object());

    // Verify parameter types
    assert_eq!(def.parameters["properties"]["text"]["type"], "string");
    assert_eq!(def.parameters["properties"]["count"]["type"], "integer");
    assert_eq!(def.parameters["properties"]["flag"]["type"], "boolean");
    assert_eq!(def.parameters["properties"]["rate"]["type"], "number");
}

#[test]
fn test_params_deserialization() {
    use kongfu::ToolParams;
    use serde_json::json;

    // Test deserializing from JSON
    let json_value = json!({
        "message": "Hello from JSON"
    });

    let params = EchoParams::from_value(json_value).unwrap();
    assert_eq!(params.message, "Hello from JSON");
}

#[test]
fn test_multi_params_deserialization() {
    use kongfu::ToolParams;
    use serde_json::json;

    // Test with all parameters
    let json_value = json!({
        "text": "test",
        "count": 10,
        "flag": true,
        "rate": 3.14
    });

    let params = MultiParamParams::from_value(json_value).unwrap();
    assert_eq!(params.text, "test");
    assert_eq!(params.count, 10);
    assert_eq!(params.flag, true);
    assert_eq!(params.rate, Some(3.14));

    // Test with optional parameter omitted
    let json_value = json!({
        "text": "test",
        "count": 5,
        "flag": false
    });

    let params = MultiParamParams::from_value(json_value).unwrap();
    assert_eq!(params.rate, None);
}

/// A tool that returns an error
#[tool]
async fn failing_tool(should_fail: bool) -> Result<String, Box<dyn std::error::Error>> {
    if should_fail {
        Err("Intentional failure".into())
    } else {
        Ok("Success".to_string())
    }
}

#[tokio::test]
async fn test_error_handling() {
    let tool = FailingTool;

    // Test success case
    let params = FailingToolParams { should_fail: false };
    let result = tool.execute(params).await.unwrap();
    assert_eq!(result, "Success");

    // Test error case
    let params = FailingToolParams { should_fail: true };
    let result = tool.execute(params).await;
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().to_string(), "Intentional failure");
}

/// A tool with a more complex implementation
#[tool]
async fn complex_tool(input: String, count: i32) -> Result<String, Box<dyn std::error::Error>> {
    if count < 0 {
        return Err("Count must be non-negative".into());
    }

    let mut result = String::new();
    for i in 0..count {
        result.push_str(&format!("{}[{}] ", input, i));
    }

    Ok(result)
}

#[tokio::test]
async fn test_complex_tool() {
    let tool = ComplexTool;

    let params = ComplexToolParams {
        input: "test".to_string(),
        count: 3,
    };

    let result = tool.execute(params).await.unwrap();
    assert_eq!(result, "test[0] test[1] test[2] ");

    // Test error case
    let params = ComplexToolParams {
        input: "test".to_string(),
        count: -1,
    };

    let result = tool.execute(params).await;
    assert!(result.is_err());
}
