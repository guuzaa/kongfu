use crate::error::Result;
use crate::message::Message;
use async_trait::async_trait;
use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::RwLock;

#[async_trait]
pub trait Memory: Send + Sync {
    async fn add(&mut self, message: Message) -> Result<()>;
    async fn get_all(&self) -> Result<Vec<Message>>;
    async fn get_recent(&self, count: usize) -> Result<Vec<Message>>;
    async fn clear(&mut self) -> Result<()>;
    // TODO async fn load() -> Self;
    // TODO async fn save(&self) -> Result<()>;
}

#[derive(Debug, Clone, Default)]
pub struct MemoryStore {
    messages: Arc<RwLock<VecDeque<Message>>>,
    max_size: Option<usize>,
}

impl MemoryStore {
    pub fn new() -> Self {
        Self {
            ..Default::default()
        }
    }

    pub fn with_max_size(max_size: usize) -> Self {
        Self {
            messages: Arc::new(RwLock::new(VecDeque::new())),
            max_size: Some(max_size),
        }
    }
}

#[async_trait]
impl Memory for MemoryStore {
    async fn add(&mut self, message: Message) -> Result<()> {
        let mut messages = self.messages.write().await;
        messages.push_back(message);

        if let Some(max_size) = self.max_size {
            if messages.len() > max_size {
                let excess = messages.len() - max_size;
                for _ in 0..excess {
                    messages.pop_front();
                }
            }
        }

        Ok(())
    }

    async fn get_all(&self) -> Result<Vec<Message>> {
        let messages = self.messages.read().await;
        Ok(messages.iter().cloned().collect())
    }

    async fn get_recent(&self, count: usize) -> Result<Vec<Message>> {
        let messages = self.messages.read().await;
        let start = messages.len().saturating_sub(count);
        Ok(messages.iter().skip(start).cloned().collect())
    }

    async fn clear(&mut self) -> Result<()> {
        let mut messages = self.messages.write().await;
        messages.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        memory::{Memory, MemoryStore},
        message::Message,
    };

    #[tokio::test]
    async fn test_memorystore_add() {
        let mut store = MemoryStore::default();
        store.add(Message::system("system prompt")).await.unwrap();
        store.add(Message::user("user query")).await.unwrap();
        store.add(Message::user("tool calling")).await.unwrap();

        let messages = store.get_all().await.unwrap();
        assert_eq!(3, messages.len());
        let recent_messages = store.get_recent(1).await.unwrap();
        assert_eq!(1, recent_messages.len());
        assert_eq!(recent_messages[0].content, "tool calling");

        store.clear().await.unwrap();
        let messages = store.get_all().await.unwrap();
        assert!(messages.is_empty());
    }
}
