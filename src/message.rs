use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

impl From<Role> for &str {
    fn from(value: Role) -> Self {
        match value {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TextBlock {
    pub text: String,
}

impl TextBlock {
    pub fn new(text: impl Into<String>) -> Self {
        Self { text: text.into() }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ThinkingBlock {
    pub thinking: String,
}

impl ThinkingBlock {
    pub fn new(thinking: impl Into<String>) -> Self {
        Self {
            thinking: thinking.into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolUseBlock {
    pub id: String,
    pub name: String,
    pub input: HashMap<String, Value>,
}

impl ToolUseBlock {
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        input: HashMap<String, Value>,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            input,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolResultBlock {
    pub tool_use_id: String,
    pub content: Option<ToolResultContent>,
    pub is_error: Option<bool>,
}

impl ToolResultBlock {
    pub fn new(
        tool_use_id: impl Into<String>,
        content: Option<ToolResultContent>,
        is_error: Option<bool>,
    ) -> Self {
        Self {
            tool_use_id: tool_use_id.into(),
            content,
            is_error,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum ToolResultContent {
    Text(String),
    Objects(Vec<HashMap<String, Value>>),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text(TextBlock),
    Thinking(ThinkingBlock),
    ToolUse(ToolUseBlock),
    ToolResult(ToolResultBlock),
}

impl ContentBlock {
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text(TextBlock::new(text))
    }

    pub fn thinking(thinking: impl Into<String>) -> Self {
        Self::Thinking(ThinkingBlock::new(thinking))
    }

    pub fn tool_use(
        id: impl Into<String>,
        name: impl Into<String>,
        input: HashMap<String, Value>,
    ) -> Self {
        Self::ToolUse(ToolUseBlock::new(id, name, input))
    }

    pub fn tool_result(
        tool_use_id: impl Into<String>,
        content: Option<ToolResultContent>,
        is_error: Option<bool>,
    ) -> Self {
        Self::ToolResult(ToolResultBlock::new(tool_use_id, content, is_error))
    }

    pub fn as_text(&self) -> Option<String> {
        match self {
            ContentBlock::Text(block) => Some(block.text.clone()),
            ContentBlock::Thinking(block) => Some(block.thinking.clone()),
            ContentBlock::ToolResult(result) => match &result.content {
                Some(ToolResultContent::Text(s)) => Some(s.clone()),
                Some(ToolResultContent::Objects(objs)) => {
                    Some(serde_json::to_string(objs).unwrap_or_default())
                }
                None => Some(String::new()),
            },
            ContentBlock::ToolUse(_) => None,
        }
    }
}

impl From<String> for ContentBlock {
    fn from(text: String) -> Self {
        Self::Text(TextBlock::new(text))
    }
}

impl From<&str> for ContentBlock {
    fn from(text: &str) -> Self {
        Self::Text(TextBlock::new(text))
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: Vec<ContentBlock>,
}

impl serde::Serialize for Message {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeMap;

        let mut map = serializer.serialize_map(None)?;
        map.serialize_entry("role", &self.role)?;

        // Handle different content configurations
        match self.content.as_slice() {
            [ContentBlock::ToolUse(tool_use)] => {
                // Single tool_use block - use tool_calls format
                let function_call = serde_json::json!({
                    "name": tool_use.name,
                    "arguments": serde_json::to_string(&tool_use.input).unwrap_or_default()
                });
                map.serialize_entry(
                    "tool_calls",
                    &vec![serde_json::json!({
                        "id": tool_use.id,
                        "type": "function",
                        "function": function_call
                    })],
                )?;
            }
            [ContentBlock::ToolResult(result)] => {
                // Single tool_result block - use tool_call_id format
                map.serialize_entry("tool_call_id", &result.tool_use_id)?;
                let content_str = match &result.content {
                    Some(ToolResultContent::Text(s)) => s.as_str(),
                    Some(ToolResultContent::Objects(_)) | None => "",
                };
                map.serialize_entry("content", content_str)?;
            }
            blocks => {
                // Empty, single text/thinking, or multiple blocks
                let content_text = blocks
                    .iter()
                    .filter_map(|block| block.as_text())
                    .collect::<Vec<_>>()
                    .join("\n");
                map.serialize_entry("content", &content_text)?;
            }
        }

        map.end()
    }
}

impl Message {
    pub fn new(role: Role, content: Vec<ContentBlock>) -> Self {
        Self { role, content }
    }

    pub fn system(content: impl Into<ContentBlock>) -> Self {
        Self::new(Role::System, vec![content.into()])
    }

    pub fn user(content: impl Into<ContentBlock>) -> Self {
        Self::new(Role::User, vec![content.into()])
    }

    pub fn assistant(content: impl Into<ContentBlock>) -> Self {
        Self::new(Role::Assistant, vec![content.into()])
    }

    pub fn tool(content: impl Into<ContentBlock>) -> Self {
        Self::new(Role::Tool, vec![content.into()])
    }

    pub fn contents(role: Role, contents: Vec<impl Into<ContentBlock>>) -> Self {
        Self::new(role, contents.into_iter().map(|c| c.into()).collect())
    }

    /// Add a block to the same message
    pub fn push(mut self, block: impl Into<ContentBlock>) -> Self {
        self.content.push(block.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use crate::message::*;
    use std::collections::HashMap;

    #[test]
    fn test_message_ctor() {
        // Single block - backward compatible
        let msg = Message::system("content");
        assert_eq!(msg.content.len(), 1);
        assert!(matches!(msg.content[0], ContentBlock::Text(_)));

        let msg = Message::assistant("assistant content");
        assert_eq!(msg.content.len(), 1);

        let msg = Message::user("user content");
        assert_eq!(msg.content.len(), 1);

        let msg = Message::tool("tool content");
        assert_eq!(msg.content.len(), 1);
    }

    #[test]
    fn test_message_push() {
        // Builder pattern for adding blocks
        let msg = Message::user("First")
            .push(ContentBlock::thinking("Second"))
            .push(ContentBlock::text("Third"));
        assert_eq!(msg.content.len(), 3);
    }

    #[test]
    fn test_content_block_text() {
        let block = ContentBlock::text("Hello, world!");
        assert!(matches!(block, ContentBlock::Text(_)));

        let block_from_string: ContentBlock = "Hello".into();
        assert!(matches!(block_from_string, ContentBlock::Text(_)));
    }

    #[test]
    fn test_content_block_thinking() {
        let block = ContentBlock::thinking("thinking process");
        assert!(matches!(block, ContentBlock::Thinking(_)));
        if let ContentBlock::Thinking(tb) = block {
            assert_eq!(tb.thinking, "thinking process");
        }
    }

    #[test]
    fn test_content_block_tool_use() {
        let mut input = HashMap::new();
        input.insert("param1".to_string(), serde_json::json!("value1"));
        input.insert("param2".to_string(), serde_json::json!(42));

        let block = ContentBlock::tool_use("tool_123", "my_tool", input.clone());
        assert!(matches!(block, ContentBlock::ToolUse(_)));
        if let ContentBlock::ToolUse(tub) = block {
            assert_eq!(tub.id, "tool_123");
            assert_eq!(tub.name, "my_tool");
            assert_eq!(tub.input.len(), 2);
        }
    }

    #[test]
    fn test_content_block_tool_result() {
        let content = ToolResultContent::Text("Result text".to_string());
        let block = ContentBlock::tool_result("tool_123", Some(content), Some(false));
        assert!(matches!(block, ContentBlock::ToolResult(_)));
        if let ContentBlock::ToolResult(trb) = block {
            assert_eq!(trb.tool_use_id, "tool_123");
            assert!(trb.content.is_some());
            assert_eq!(trb.is_error, Some(false));
        }
    }

    #[test]
    fn test_message_insert() {
        // insert now replaced by push
        let msg = Message::system("Initial content").push(ContentBlock::text("Additional content"));
        assert_eq!(msg.content.len(), 2);
        if let ContentBlock::Text(block) = &msg.content[1] {
            assert_eq!(block.text, "Additional content");
        }
    }

    #[test]
    fn test_content_block_as_text() {
        let text_block = ContentBlock::text("Hello, world!");
        assert_eq!(text_block.as_text(), Some("Hello, world!".to_string()));

        let thinking_block = ContentBlock::thinking("thinking");
        assert_eq!(thinking_block.as_text(), Some("thinking".to_string()));
    }

    #[test]
    fn test_content_block_serialization_tool_use() {
        let mut input = HashMap::new();
        input.insert("param1".to_string(), serde_json::json!("value1"));
        input.insert("param2".to_string(), serde_json::json!(42));

        let block = ContentBlock::tool_use("call_123", "my_tool", input.clone());
        let json = serde_json::to_string(&block).unwrap();
        assert!(json.contains("\"type\":\"tool_use\""));
        assert!(json.contains("call_123"));
        assert!(json.contains("my_tool"));

        // Test deserialization
        let deserialized: ContentBlock = serde_json::from_str(&json).unwrap();
        assert_eq!(block, deserialized);
    }

    #[test]
    fn test_message_serialization_single_text() {
        // Single text block should serialize to string
        let msg = Message::user("Hello, world!");
        let json = serde_json::to_string(&msg).unwrap();

        // Should be {"role":"user","content":"Hello, world!"}
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();

        assert_eq!(value["role"], "user");
        assert!(value["content"].is_string());
        assert_eq!(value["content"], "Hello, world!");
    }

    #[test]
    fn test_message_serialization_multi_block() {
        let msg = Message::user("Hello").push(ContentBlock::thinking("Thinking"));
        let json = serde_json::to_string(&msg).unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();

        assert_eq!(value["role"], "user");
        assert!(value["content"].is_string());
        let content = value["content"].as_str().unwrap();
        let lines = content.lines().collect::<Vec<_>>();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0], "Hello");
        assert_eq!(lines[1], "Thinking");
    }
}
