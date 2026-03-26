//! Tool system for type-safe tool definitions and execution
//!
//! This module provides traits and utilities for defining tools that can be used
//! by LLM agents. The tool system ensures type safety and automatic JSON schema generation.

mod edit_file;
mod list_directory;
mod read_file;

// Re-export common tools for convenience
pub use edit_file::{EditFile, EditFileParams};
pub use list_directory::{ListDirectory, ListDirectoryParams};
pub use read_file::{ReadFile, ReadFileParams};

use async_trait::async_trait;
use serde::Serialize;
use serde_json::Value;

// ============================================================================
// Tool System Traits
// ============================================================================

/// Trait for tool parameter structures
///
/// This trait should be derived using `#[derive(ToolParams)]` from kongfu_macros
///
/// # Example
///
/// ```rust,ignore
/// use kongfu_macros::ToolParams;
/// use serde::Deserialize;
///
/// #[derive(ToolParams, Deserialize)]
/// struct MyToolParams {
///     /// Parameter description (automatically extracted)
///     name: String,
///     count: Option<i32>,
/// }
/// ```
pub trait ToolParams: Send + Sync + Sized {
    /// Get the JSON schema for these parameters
    fn schema() -> Value;

    /// Parse from JSON value
    fn from_value(value: Value) -> std::result::Result<Self, String>;
}

/// Handler for a specific tool with type-safe parameters
///
/// # Example
///
/// ```rust,ignore
/// use kongfu::ToolHandler;
/// use async_trait::async_trait;
///
/// struct MyTool;
///
/// #[async_trait::async_trait]
/// impl ToolHandler for MyTool {
///     type Params = MyToolParams;
///     type Output = String;
///
///     fn name(&self) -> &str {
///         "my_tool"
///     }
///
///     fn description(&self) -> &str {
///         "Does something"
///     }
///
///     async fn execute(&self, params: Self::Params)
///         -> std::result::Result<Self::Output, Box<dyn std::error::Error>>
///     {
///         Ok(format!("Received: {}", params.name))
///     }
/// }
/// ```
#[async_trait]
pub trait ToolHandler: Send + Sync {
    /// Parameter type for this tool
    type Params: ToolParams;

    /// Output type for this tool
    type Output: Send + Sync + Serialize;

    /// Tool name
    fn name(&self) -> &str;

    /// Tool description
    fn description(&self) -> &str;

    /// Execute the tool with given parameters
    async fn execute(
        &self,
        params: Self::Params,
    ) -> std::result::Result<Self::Output, Box<dyn std::error::Error>>;

    /// Get JSON schema for this tool's parameters
    fn schema(&self) -> Value {
        Self::Params::schema()
    }

    /// Execute with raw JSON value (convenience method)
    async fn execute_raw(
        &self,
        params: Value,
    ) -> std::result::Result<Value, Box<dyn std::error::Error>> {
        let typed_params = Self::Params::from_value(params).map_err(|e| {
            Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, e))
                as Box<dyn std::error::Error>
        })?;

        let result = self.execute(typed_params).await?;

        serde_json::to_value(result).map_err(|e| e.into())
    }
}

/// Helper trait to convert ToolHandler to FunctionDefinition
pub trait ToFunctionDefinition {
    fn to_function_definition(&self) -> crate::provider::FunctionDefinition;
}

impl<H> ToFunctionDefinition for H
where
    H: ToolHandler,
{
    fn to_function_definition(&self) -> crate::provider::FunctionDefinition {
        crate::provider::FunctionDefinition {
            name: self.name().to_string(),
            description: self.description().to_string(),
            parameters: self.schema(),
        }
    }
}

// ============================================================================
// Tool Registry
// ============================================================================

/// A registry for managing tools
///
/// # Example
///
/// ```rust,ignore
/// use kongfu::{ToolRegistry, ToolHandler};
///
/// let tools = ToolRegistry::new()
///     .add(ListDirectory)
///     .add(ReadFile)
///     .add(WriteFile);
///
/// // Convert to LLM API format
/// let definitions = tools.to_definitions();
/// ```
#[derive(Default)]
pub struct ToolRegistry {
    tools: Vec<Box<dyn ToolHandlerWrapper>>,
}

// Wrapper trait to allow dynamic dispatch of different ToolHandler types
#[async_trait]
trait ToolHandlerWrapper: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn schema(&self) -> Value;
    fn to_definition(&self) -> crate::provider::FunctionDefinition;
    async fn execute_raw(
        &self,
        params: Value,
    ) -> std::result::Result<Value, Box<dyn std::error::Error>>;
}

#[async_trait]
impl<H> ToolHandlerWrapper for H
where
    H: ToolHandler + 'static,
{
    fn name(&self) -> &str {
        ToolHandler::name(self)
    }

    fn description(&self) -> &str {
        ToolHandler::description(self)
    }

    fn schema(&self) -> Value {
        ToolHandler::schema(self)
    }

    fn to_definition(&self) -> crate::provider::FunctionDefinition {
        self.to_function_definition()
    }

    async fn execute_raw(
        &self,
        params: Value,
    ) -> std::result::Result<Value, Box<dyn std::error::Error>> {
        ToolHandler::execute_raw(self, params).await
    }
}

impl ToolRegistry {
    /// Create a new empty tool registry
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a tool to the registry
    pub fn add<H>(mut self, tool: H) -> Self
    where
        H: ToolHandler + 'static,
    {
        let boxed: Box<dyn ToolHandlerWrapper> = Box::new(tool);
        self.tools.push(boxed);
        self
    }

    /// Convert all tools to FunctionDefinition format for LLM API
    pub fn to_definitions(&self) -> Vec<crate::provider::FunctionDefinition> {
        self.tools.iter().map(|t| t.to_definition()).collect()
    }

    /// Convert to provider Tool enum format
    pub fn to_tools(&self) -> Vec<crate::provider::Tool> {
        self.to_definitions()
            .into_iter()
            .map(crate::provider::Tool::Function)
            .collect()
    }

    /// Execute a tool by name
    pub async fn execute(&self, name: &str, params: Value) -> std::result::Result<Value, String> {
        self.tools
            .iter()
            .find(|t| t.name() == name)
            .ok_or_else(|| format!("Tool not found: {}", name))?
            .execute_raw(params)
            .await
            .map_err(|e| format!("Tool execution failed: {}", e))
    }

    /// List all tool names
    pub fn list(&self) -> Vec<&str> {
        self.tools.iter().map(|t| t.name()).collect()
    }

    /// Get tool count
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Check if registry is empty
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestTool;

    #[async_trait::async_trait]
    impl ToolHandler for TestTool {
        type Params = TestParams;
        type Output = String;

        fn name(&self) -> &str {
            "test_tool"
        }

        fn description(&self) -> &str {
            "A test tool"
        }

        async fn execute(
            &self,
            params: Self::Params,
        ) -> std::result::Result<Self::Output, Box<dyn std::error::Error>> {
            Ok(format!("Got: {}", params.input))
        }
    }

    #[derive(kongfu_macros::ToolParams, serde::Deserialize)]
    struct TestParams {
        input: String,
    }

    #[tokio::test]
    async fn test_tool_handler() {
        let tool = TestTool;
        assert_eq!(ToolHandler::name(&tool), "test_tool");
        assert_eq!(ToolHandler::description(&tool), "A test tool");

        let schema = ToolHandler::schema(&tool);
        assert!(schema.is_object());
        assert_eq!(schema["type"], "object");
    }

    #[test]
    fn test_to_function_definition() {
        let def = TestTool.to_function_definition();

        assert_eq!(def.name, "test_tool");
        assert_eq!(def.description, "A test tool");

        assert!(def.parameters.is_object());
        let expected = r#"{"properties":{"input":{"description":"The input parameter","type":"string"}},"required":["input"],"type":"object"}"#;
        let parameters = def.parameters.to_string();
        assert_eq!(expected, parameters);
    }

    #[derive(kongfu_macros::ToolParams, serde::Deserialize)]
    struct TestParamsWithDoc {
        /// A test string parameter
        name: String,
        /// An optional number
        count: Option<i32>,
    }

    #[test]
    fn test_tool_params_schema_generation() {
        let schema = TestParamsWithDoc::schema();
        assert!(schema.is_object());
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"].is_object());
    }

    #[test]
    fn test_tool_params_from_value() {
        let json = r#"{"name": "test", "count": 42}"#;
        let value: serde_json::Value = serde_json::from_str(json).unwrap();

        let params = TestParamsWithDoc::from_value(value).unwrap();
        assert_eq!(params.name, "test");
        assert_eq!(params.count, Some(42));
    }

    #[test]
    fn test_tool_params_from_value_with_optional() {
        let json = r#"{"name": "test"}"#;
        let value: serde_json::Value = serde_json::from_str(json).unwrap();

        let params = TestParamsWithDoc::from_value(value).unwrap();
        assert_eq!(params.name, "test");
        assert_eq!(params.count, None);
    }
}
