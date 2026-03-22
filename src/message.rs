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
            ContentBlock::Thinking(block) => {
                let thinking = format!("<think>{}</think>", &block.thinking);
                Some(thinking)
            }
            _ => None,
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
    pub content: ContentBlock,
}

impl serde::Serialize for Message {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeMap;

        let mut map = serializer.serialize_map(Some(2))?;
        map.serialize_entry("role", &self.role)?;

        match &self.content {
            ContentBlock::Text(TextBlock { text }) => {
                map.serialize_entry("content", text)?;
            }
            ContentBlock::Thinking(ThinkingBlock { thinking }) => {
                map.serialize_entry("content", thinking)?;
            }
            ContentBlock::ToolUse(block) => {
                map.serialize_entry("content", block)?;
            }
            ContentBlock::ToolResult(result) => {
                map.serialize_entry("content", result)?;
            }
        }

        map.end()
    }
}

impl Message {
    pub fn new(role: Role, content: ContentBlock) -> Self {
        Self { role, content }
    }

    pub fn system(content: impl Into<ContentBlock>) -> Self {
        Self::new(Role::System, content.into())
    }

    pub fn user(content: impl Into<ContentBlock>) -> Self {
        Self::new(Role::User, content.into())
    }

    pub fn assistant(content: impl Into<ContentBlock>) -> Self {
        Self::new(Role::Assistant, content.into())
    }

    pub fn tool(content: impl Into<ContentBlock>) -> Self {
        Self::new(Role::Tool, content.into())
    }

    pub fn with_blocks(role: Role, blocks: Vec<ContentBlock>) -> Self {
        // Only take the first block
        let content = blocks
            .into_iter()
            .next()
            .unwrap_or_else(|| ContentBlock::text(""));
        Self::new(role, content)
    }

    pub fn with_block(mut self, block: ContentBlock) -> Self {
        // Replace current block with new one
        self.content = block;
        self
    }
}

#[cfg(test)]
mod tests {
    use crate::message::*;
    use std::collections::HashMap;

    #[test]
    fn test_message_ctor() {
        let msg = Message::system("content");
        assert!(matches!(msg.content, ContentBlock::Text(_)));

        let msg = Message::assistant("assistant content");
        assert!(matches!(msg.content, ContentBlock::Text(_)));

        let msg = Message::user("user content");
        assert!(matches!(msg.content, ContentBlock::Text(_)));

        let msg = Message::tool("tool content");
        assert!(matches!(msg.content, ContentBlock::Text(_)));
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
    fn test_message_with_multiple_blocks() {
        // with_blocks now only takes the first block
        let blocks = vec![
            ContentBlock::thinking("First text"),
            ContentBlock::text("Second text"),
        ];
        let msg = Message::with_blocks(Role::User, blocks);
        // Should only have the first block (thinking)
        assert!(matches!(msg.content, ContentBlock::Thinking(_)));
    }

    #[test]
    fn test_message_with_block() {
        // with_block now replaces the content
        let msg =
            Message::system("Initial content").with_block(ContentBlock::text("Additional content"));
        assert!(matches!(msg.content, ContentBlock::Text(_)));
        if let ContentBlock::Text(block) = &msg.content {
            assert_eq!(block.text, "Additional content");
        }
    }

    #[test]
    fn test_content_block_as_text() {
        let text_block = ContentBlock::text("Hello, world!");
        assert_eq!(text_block.as_text(), Some("Hello, world!".to_string()));

        let thinking_block = ContentBlock::thinking("thinking");
        assert_eq!(thinking_block.as_text(), None);
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
    fn test_message_serialization_non_text_block() {
        // Non-text block should serialize to object
        let mut input = HashMap::new();
        input.insert("param".to_string(), serde_json::json!("value"));

        let msg = Message::with_blocks(
            Role::Assistant,
            vec![ContentBlock::tool_use("tool_1", "my_tool", input)],
        );
        let json = serde_json::to_string(&msg).unwrap();

        let value: serde_json::Value = serde_json::from_str(&json).unwrap();

        assert_eq!(value["role"], "assistant");
        // Single non-text block serializes to an object with type field
        assert!(value["content"].is_object());
    }
}
