use std::path::PathBuf;
use std::sync::Arc;

use comfy_core::NodeRegistry;
use tokio::sync::Mutex;

use crate::history::HistoryStore;
use crate::queue::PromptQueue;

/// Shared application state accessible from all route handlers.
#[derive(Clone)]
pub struct AppState {
    pub registry: Arc<NodeRegistry>,
    pub queue: Arc<Mutex<PromptQueue>>,
    pub history: Arc<Mutex<HistoryStore>>,
    pub base_path: PathBuf,
}

impl AppState {
    pub fn new(registry: NodeRegistry, base_path: PathBuf) -> Self {
        Self {
            registry: Arc::new(registry),
            queue: Arc::new(Mutex::new(PromptQueue::new())),
            history: Arc::new(Mutex::new(HistoryStore::new(200))),
            base_path,
        }
    }
}
