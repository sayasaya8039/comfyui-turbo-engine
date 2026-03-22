use std::path::PathBuf;
use std::sync::Arc;

use comfy_core::NodeRegistry;
use comfy_python::FolderPaths;
use tokio::sync::{broadcast, Mutex};

use crate::history::HistoryStore;
use crate::queue::PromptQueue;

/// Shared application state accessible from all route handlers.
#[derive(Clone)]
pub struct AppState {
    pub registry: Arc<NodeRegistry>,
    pub queue: Arc<Mutex<PromptQueue>>,
    pub history: Arc<Mutex<HistoryStore>>,
    pub base_path: PathBuf,
    pub folder_paths: Arc<FolderPaths>,
    /// Broadcast channel for sending progress notifications to all WebSocket clients.
    pub progress_tx: broadcast::Sender<String>,
}

impl AppState {
    pub fn new(registry: NodeRegistry, base_path: PathBuf) -> Self {
        let (progress_tx, _rx) = broadcast::channel(256);
        let folder_paths = FolderPaths::new(&base_path);
        Self {
            registry: Arc::new(registry),
            queue: Arc::new(Mutex::new(PromptQueue::new())),
            history: Arc::new(Mutex::new(HistoryStore::new(200))),
            folder_paths: Arc::new(folder_paths),
            base_path,
            progress_tx,
        }
    }
}
