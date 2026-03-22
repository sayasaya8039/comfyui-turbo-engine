use std::collections::{HashMap, VecDeque};

use crate::node::NodeOutputs;

/// LRU cache for node execution outputs.
///
/// When the cache exceeds `max_entries`, the least recently used
/// entry is evicted.
pub struct TensorCache {
    max_entries: usize,
    /// Map from cache key to stored outputs.
    map: HashMap<String, NodeOutputs>,
    /// LRU order: front = oldest (least recently used), back = newest.
    order: VecDeque<String>,
}

impl TensorCache {
    pub fn new(max_entries: usize) -> Self {
        Self {
            max_entries,
            map: HashMap::new(),
            order: VecDeque::new(),
        }
    }

    /// Retrieve a cached entry, promoting it to most-recently-used.
    pub fn get(&mut self, key: &str) -> Option<&NodeOutputs> {
        if self.map.contains_key(key) {
            // Move to back (most recently used)
            self.order.retain(|k| k != key);
            self.order.push_back(key.to_string());
            self.map.get(key)
        } else {
            None
        }
    }

    /// Insert an entry into the cache, evicting the LRU entry if at capacity.
    pub fn put(&mut self, key: impl Into<String>, value: NodeOutputs) {
        let key = key.into();

        // If already present, remove old position
        if self.map.contains_key(&key) {
            self.order.retain(|k| k != &key);
        } else if self.map.len() >= self.max_entries {
            // Evict LRU (front of deque)
            if let Some(evicted) = self.order.pop_front() {
                self.map.remove(&evicted);
            }
        }

        self.map.insert(key.clone(), value);
        self.order.push_back(key);
    }

    /// Clear all cached entries.
    pub fn clear(&mut self) {
        self.map.clear();
        self.order.clear();
    }

    /// Number of entries currently in the cache.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::NodeValue;

    fn make_output(value: f64) -> NodeOutputs {
        let mut out = NodeOutputs::new();
        out.set("output_0", NodeValue::Float(value));
        out
    }

    #[test]
    fn test_cache_put_get() {
        let mut cache = TensorCache::new(10);
        cache.put("node_1", make_output(1.0));
        cache.put("node_2", make_output(2.0));

        let out = cache.get("node_1").unwrap();
        match out.get("output_0") {
            Some(NodeValue::Float(v)) => assert_eq!(*v, 1.0),
            other => panic!("Expected Float(1.0), got {other:?}"),
        }

        let out = cache.get("node_2").unwrap();
        match out.get("output_0") {
            Some(NodeValue::Float(v)) => assert_eq!(*v, 2.0),
            other => panic!("Expected Float(2.0), got {other:?}"),
        }

        assert!(cache.get("nonexistent").is_none());
    }

    #[test]
    fn test_cache_eviction() {
        let mut cache = TensorCache::new(2);
        cache.put("A", make_output(1.0));
        cache.put("B", make_output(2.0));
        assert_eq!(cache.len(), 2);

        // Adding C should evict A (LRU)
        cache.put("C", make_output(3.0));
        assert_eq!(cache.len(), 2);
        assert!(cache.get("A").is_none(), "A should have been evicted");
        assert!(cache.get("B").is_some());
        assert!(cache.get("C").is_some());
    }

    #[test]
    fn test_cache_lru_promotion() {
        let mut cache = TensorCache::new(2);
        cache.put("A", make_output(1.0));
        cache.put("B", make_output(2.0));

        // Access A to promote it
        cache.get("A");

        // Now adding C should evict B (A was promoted)
        cache.put("C", make_output(3.0));
        assert!(cache.get("A").is_some(), "A should still be present");
        assert!(cache.get("B").is_none(), "B should have been evicted");
        assert!(cache.get("C").is_some());
    }

    #[test]
    fn test_cache_overwrite() {
        let mut cache = TensorCache::new(10);
        cache.put("A", make_output(1.0));
        cache.put("A", make_output(99.0));

        assert_eq!(cache.len(), 1);
        let out = cache.get("A").unwrap();
        match out.get("output_0") {
            Some(NodeValue::Float(v)) => assert_eq!(*v, 99.0),
            other => panic!("Expected Float(99.0), got {other:?}"),
        }
    }

    #[test]
    fn test_cache_clear() {
        let mut cache = TensorCache::new(10);
        cache.put("A", make_output(1.0));
        cache.put("B", make_output(2.0));
        assert_eq!(cache.len(), 2);

        cache.clear();
        assert!(cache.is_empty());
        assert!(cache.get("A").is_none());
    }
}
