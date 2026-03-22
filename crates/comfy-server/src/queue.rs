use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

/// An item in the prompt queue.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueItem {
    pub number: f64,
    pub prompt_id: String,
    pub prompt: serde_json::Value,
    pub client_id: Option<String>,
}

/// Priority queue for prompt execution.
///
/// Items marked as "front" are inserted at the head of the queue.
/// A separate `running` slot tracks the currently executing item.
pub struct PromptQueue {
    pending: VecDeque<QueueItem>,
    running: Option<QueueItem>,
    counter: f64,
}

impl PromptQueue {
    pub fn new() -> Self {
        Self {
            pending: VecDeque::new(),
            running: None,
            counter: 0.0,
        }
    }

    /// Enqueue a new prompt. If `front` is true, insert at the head.
    /// Returns the assigned queue number.
    pub fn enqueue(
        &mut self,
        prompt_id: String,
        prompt: serde_json::Value,
        client_id: Option<String>,
        front: bool,
    ) -> f64 {
        self.counter += 1.0;
        let number = self.counter;
        let item = QueueItem {
            number,
            prompt_id,
            prompt,
            client_id,
        };
        if front {
            self.pending.push_front(item);
        } else {
            self.pending.push_back(item);
        }
        number
    }

    /// Dequeue the next pending item and mark it as running.
    pub fn dequeue(&mut self) -> Option<QueueItem> {
        let item = self.pending.pop_front();
        if let Some(ref i) = item {
            self.running = Some(i.clone());
        }
        item
    }

    /// Mark the running item as finished.
    pub fn finish(&mut self) {
        self.running = None;
    }

    /// Clear all pending items. Optionally returns the cleared items.
    pub fn clear(&mut self) -> Vec<QueueItem> {
        self.pending.drain(..).collect()
    }

    /// Delete specific items by prompt_id from the pending queue.
    pub fn delete(&mut self, prompt_ids: &[String]) {
        self.pending.retain(|item| !prompt_ids.contains(&item.prompt_id));
    }

    /// Number of items remaining (pending + running).
    pub fn remaining(&self) -> usize {
        self.pending.len() + if self.running.is_some() { 1 } else { 0 }
    }

    /// Get a snapshot of the running queue as serializable tuples.
    pub fn running_items(&self) -> Vec<(f64, String)> {
        match &self.running {
            Some(item) => vec![(item.number, item.prompt_id.clone())],
            None => vec![],
        }
    }

    /// Get a snapshot of the pending queue as serializable tuples.
    pub fn pending_items(&self) -> Vec<(f64, String)> {
        self.pending
            .iter()
            .map(|item| (item.number, item.prompt_id.clone()))
            .collect()
    }
}

impl Default for PromptQueue {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enqueue_dequeue() {
        let mut q = PromptQueue::new();
        let n = q.enqueue("a".into(), serde_json::json!({}), None, false);
        assert_eq!(n, 1.0);
        assert_eq!(q.remaining(), 1);

        let item = q.dequeue().unwrap();
        assert_eq!(item.prompt_id, "a");
        assert_eq!(q.remaining(), 1); // still running
        q.finish();
        assert_eq!(q.remaining(), 0);
    }

    #[test]
    fn test_front_priority() {
        let mut q = PromptQueue::new();
        q.enqueue("back".into(), serde_json::json!({}), None, false);
        q.enqueue("front".into(), serde_json::json!({}), None, true);

        let item = q.dequeue().unwrap();
        assert_eq!(item.prompt_id, "front");
    }

    #[test]
    fn test_clear_and_delete() {
        let mut q = PromptQueue::new();
        q.enqueue("a".into(), serde_json::json!({}), None, false);
        q.enqueue("b".into(), serde_json::json!({}), None, false);
        q.enqueue("c".into(), serde_json::json!({}), None, false);

        q.delete(&["b".into()]);
        assert_eq!(q.pending_items().len(), 2);

        let cleared = q.clear();
        assert_eq!(cleared.len(), 2);
        assert_eq!(q.remaining(), 0);
    }

    #[test]
    fn test_running_items() {
        let mut q = PromptQueue::new();
        assert!(q.running_items().is_empty());
        q.enqueue("x".into(), serde_json::json!({}), None, false);
        q.dequeue();
        assert_eq!(q.running_items().len(), 1);
        assert_eq!(q.running_items()[0].1, "x");
    }
}
