use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub id: String,
    pub role: Role,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

impl Message {
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            role,
            content: content.into(),
            metadata: None,
        }
    }

    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = Some(metadata);
        self
    }

    pub fn system(content: impl Into<String>) -> Self {
        Self::new(Role::System, content)
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::new(Role::User, content)
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new(Role::Assistant, content)
    }

    pub fn tool(content: impl Into<String>) -> Self {
        Self::new(Role::Tool, content)
    }
}

#[cfg(test)]
mod tests {
    use crate::message::Message;

    #[test]
    fn test_message_ctor() {
        let msg = Message::system("content");
        assert_eq!(msg.content, "content");
        assert!(!msg.id.is_empty());

        let msg = Message::assistant("assistant content");
        assert_eq!(msg.content, "assistant content");

        let msg = Message::user("user content");
        assert_eq!(msg.content, "user content");

        let msg = Message::user("tool content");
        assert_eq!(msg.content, "tool content");
    }
}
