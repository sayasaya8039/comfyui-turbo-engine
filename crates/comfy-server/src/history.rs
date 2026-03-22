use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// A single history entry recording the result of a prompt execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryEntry {
    pub prompt_id: String,
    pub prompt: serde_json::Value,
    pub outputs: HashMap<String, serde_json::Value>,
    pub status: HistoryStatus,
}

/// Status of a completed prompt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryStatus {
    pub status_str: String,
    pub completed: bool,
}

/// In-memory store for prompt execution history.
///
/// Entries are stored in insertion order. When `max_size` is exceeded,
/// the oldest entry is evicted.
pub struct HistoryStore {
    entries: Vec<HistoryEntry>,
    max_size: usize,
}

impl HistoryStore {
    pub fn new(max_size: usize) -> Self {
        Self {
            entries: Vec::new(),
            max_size,
        }
    }

    /// Add a new history entry, evicting the oldest if at capacity.
    pub fn add(&mut self, entry: HistoryEntry) {
        if self.entries.len() >= self.max_size {
            self.entries.remove(0);
        }
        self.entries.push(entry);
    }

    /// Get a single entry by prompt_id.
    pub fn get(&self, prompt_id: &str) -> Option<&HistoryEntry> {
        self.entries.iter().find(|e| e.prompt_id == prompt_id)
    }

    /// List all entries (most recent last).
    pub fn list(&self) -> &[HistoryEntry] {
        &self.entries
    }

    /// Clear all history.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Delete specific entries by prompt_id.
    pub fn delete(&mut self, prompt_ids: &[String]) {
        self.entries.retain(|e| !prompt_ids.contains(&e.prompt_id));
    }
}

impl Default for HistoryStore {
    fn default() -> Self {
        Self::new(200)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entry(id: &str) -> HistoryEntry {
        HistoryEntry {
            prompt_id: id.to_string(),
            prompt: serde_json::json!({}),
            outputs: HashMap::new(),
            status: HistoryStatus {
                status_str: "success".to_string(),
                completed: true,
            },
        }
    }

    #[test]
    fn test_add_and_get() {
        let mut store = HistoryStore::new(10);
        store.add(make_entry("a"));
        assert!(store.get("a").is_some());
        assert!(store.get("b").is_none());
    }

    #[test]
    fn test_max_size_eviction() {
        let mut store = HistoryStore::new(2);
        store.add(make_entry("a"));
        store.add(make_entry("b"));
        store.add(make_entry("c"));

        assert!(store.get("a").is_none()); // evicted
        assert!(store.get("b").is_some());
        assert!(store.get("c").is_some());
        assert_eq!(store.list().len(), 2);
    }

    #[test]
    fn test_clear_and_delete() {
        let mut store = HistoryStore::new(10);
        store.add(make_entry("a"));
        store.add(make_entry("b"));
        store.add(make_entry("c"));

        store.delete(&["b".into()]);
        assert_eq!(store.list().len(), 2);
        assert!(store.get("b").is_none());

        store.clear();
        assert!(store.list().is_empty());
    }
}
