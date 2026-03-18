use crate::error::{KongfuError, Result};
use async_trait::async_trait;
#[cfg(test)]
use mockall::automock;
use serde_json::Value;
use std::collections::HashMap;

#[cfg_attr(test, automock)]
#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    async fn execute(&self, params: Value) -> Result<Value>;
}

#[derive(Default)]
pub struct ToolRegistry {
    tools: HashMap<String, Box<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(mut self, tool: Box<dyn Tool>) -> Self {
        let name = tool.name().to_string();
        self.tools.insert(name, tool);
        self
    }

    pub fn get(&self, name: &str) -> Option<&dyn Tool> {
        self.tools.get(name).map(|b| b.as_ref())
    }

    pub async fn execute(&self, name: &str, params: Value) -> Result<Value> {
        let tool = self
            .get(name)
            .ok_or_else(|| KongfuError::ToolNotFound(name.to_string()))?;

        tool.execute(params)
            .await
            .map_err(|e| KongfuError::ToolExecutionFailed(e.to_string()))
    }

    pub fn list_tools(&self) -> Vec<(String, String)> {
        self.tools
            .iter()
            .map(|(name, tool)| (name.clone(), tool.description().to_string()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_tool_registry() {
        let mut tool1 = MockTool::new();
        tool1.expect_name().return_const("test_tool".into());
        tool1
            .expect_description()
            .return_const("A test tool".into());
        tool1
            .expect_execute()
            .returning(|_| Ok(serde_json::json!({"result": "mocked"})));

        let registry = ToolRegistry::default().register(Box::new(tool1));
        let result = registry.execute("test_tool", serde_json::json!({})).await;
        assert!(result.is_ok());

        let result = registry
            .get("test_tool")
            .unwrap()
            .execute(serde_json::json!({}))
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_tool_registry_multiple_tools() {
        let tools_num = 5;
        let mut tools = Vec::with_capacity(tools_num);
        for i in 0..tools_num {
            let mut tool = MockTool::new();
            tool.expect_name().return_const(format!("test_tool{i}"));
            tool.expect_description()
                .return_const(format!("A test tool#{i}"));
            tool.expect_execute()
                .returning(|_| Ok(serde_json::json!({"result": "mocked"})));
            tools.push(tool);
        }

        let mut registry = ToolRegistry::default();
        for tool in tools {
            registry = registry.register(Box::new(tool));
        }

        let result = registry.execute("test_tool1", serde_json::json!({})).await;
        assert!(result.is_ok());

        let tools_summary = registry.list_tools();
        assert_eq!(tools_summary.len(), tools_num);
    }
}
