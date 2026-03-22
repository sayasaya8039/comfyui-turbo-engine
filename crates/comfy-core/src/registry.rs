use std::collections::HashMap;

use crate::node::Node;

/// Registry that maps node type names to factory functions.
pub struct NodeRegistry {
    factories: HashMap<String, Box<dyn Fn() -> Box<dyn Node> + Send + Sync>>,
}

impl NodeRegistry {
    pub fn new() -> Self {
        Self {
            factories: HashMap::new(),
        }
    }

    /// Register a node factory under the given name.
    pub fn register<F>(&mut self, name: impl Into<String>, factory: F)
    where
        F: Fn() -> Box<dyn Node> + Send + Sync + 'static,
    {
        self.factories.insert(name.into(), Box::new(factory));
    }

    /// Create a new instance of a node by its registered name.
    pub fn create(&self, name: &str) -> Option<Box<dyn Node>> {
        self.factories.get(name).map(|f| f())
    }

    /// List all registered node names (sorted for determinism).
    pub fn list(&self) -> Vec<String> {
        let mut names: Vec<String> = self.factories.keys().cloned().collect();
        names.sort();
        names
    }

    /// Check if a node type is registered.
    pub fn contains(&self, name: &str) -> bool {
        self.factories.contains_key(name)
    }
}

impl Default for NodeRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ComfyResult;
    use crate::node::{NodeInputs, NodeMetadata, NodeOutputs, NodeValue};

    /// A test node for registry tests.
    struct ConstNode {
        value: f64,
    }

    impl Node for ConstNode {
        fn execute(&self, _inputs: &NodeInputs) -> ComfyResult<NodeOutputs> {
            let mut outputs = NodeOutputs::new();
            outputs.set("output_0", NodeValue::Float(self.value));
            Ok(outputs)
        }

        fn metadata(&self) -> NodeMetadata {
            NodeMetadata {
                name: "ConstNode".to_string(),
                display_name: "Constant".to_string(),
                category: "utils".to_string(),
                description: "Produces a constant value".to_string(),
                output_node: false,
            }
        }
    }

    #[test]
    fn test_register_and_create() {
        let mut registry = NodeRegistry::new();
        registry.register("ConstNode", || Box::new(ConstNode { value: 42.0 }));

        assert!(registry.contains("ConstNode"));
        assert!(!registry.contains("NonExistent"));

        let node = registry.create("ConstNode").unwrap();
        let inputs = NodeInputs::new();
        let outputs = node.execute(&inputs).unwrap();

        match outputs.get("output_0") {
            Some(NodeValue::Float(v)) => assert_eq!(*v, 42.0),
            other => panic!("Expected Float(42.0), got {other:?}"),
        }
    }

    #[test]
    fn test_create_nonexistent() {
        let registry = NodeRegistry::new();
        assert!(registry.create("NonExistent").is_none());
    }

    #[test]
    fn test_list_nodes() {
        let mut registry = NodeRegistry::new();
        registry.register("Zebra", || Box::new(ConstNode { value: 0.0 }));
        registry.register("Alpha", || Box::new(ConstNode { value: 0.0 }));
        registry.register("Middle", || Box::new(ConstNode { value: 0.0 }));

        let names = registry.list();
        assert_eq!(names, vec!["Alpha", "Middle", "Zebra"]);
    }

    #[test]
    fn test_register_overwrite() {
        let mut registry = NodeRegistry::new();
        registry.register("Node", || Box::new(ConstNode { value: 1.0 }));
        registry.register("Node", || Box::new(ConstNode { value: 2.0 }));

        let node = registry.create("Node").unwrap();
        let outputs = node.execute(&NodeInputs::new()).unwrap();
        match outputs.get("output_0") {
            Some(NodeValue::Float(v)) => assert_eq!(*v, 2.0),
            other => panic!("Expected Float(2.0), got {other:?}"),
        }
    }
}
