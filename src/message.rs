use serde::{Deserialize, Serialize};

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

impl Message {
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
        }
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

        let msg = Message::assistant("assistant content");
        assert_eq!(msg.content, "assistant content");

        let msg = Message::user("user content");
        assert_eq!(msg.content, "user content");

        let msg = Message::user("tool content");
        assert_eq!(msg.content, "tool content");
    }
}
