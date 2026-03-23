use std::collections::HashMap;

use crate::cache::TensorCache;
use crate::dag::Dag;
use crate::error::{ComfyError, ComfyResult};
use crate::node::{output_key, NodeInputs, NodeOutputs, NodeValue};
use crate::registry::NodeRegistry;
use crate::workflow::{InputValue, Workflow};

/// Executor that runs a workflow through the DAG in topological order.
pub struct Executor {
    registry: NodeRegistry,
    cache: TensorCache,
}

impl Executor {
    pub fn new(registry: NodeRegistry, cache_size: usize) -> Self {
        Self {
            registry,
            cache: TensorCache::new(cache_size),
        }
    }

    /// Execute a workflow from its JSON representation.
    ///
    /// Returns a map from node_id to that node's outputs.
    pub fn execute_workflow(
        &mut self,
        json: &str,
    ) -> ComfyResult<HashMap<String, NodeOutputs>> {
        let workflow = Workflow::from_json(json)?;
        let dag = Dag::from_workflow(&workflow)?;
        let sorted = dag.topological_sort()?;

        let mut all_outputs: HashMap<String, NodeOutputs> = HashMap::new();

        for node_id in &sorted {
            // Check cache first
            if let Some(cached) = self.cache.get(node_id) {
                all_outputs.insert(node_id.clone(), cached.clone());
                continue;
            }

            let workflow_node = workflow.nodes.get(node_id).ok_or_else(|| {
                ComfyError::NodeNotFound(node_id.clone())
            })?;

            // Create the node instance from registry
            let node_impl = self.registry.create(&workflow_node.class_type).ok_or_else(|| {
                ComfyError::NodeNotFound(format!(
                    "class_type '{}' not registered",
                    workflow_node.class_type
                ))
            })?;

            // Build inputs by resolving links from previous outputs
            let mut inputs = NodeInputs::new();
            for (input_name, input_val) in &workflow_node.inputs {
                let resolved = self.resolve_input(input_val, &all_outputs)?;
                inputs.set(input_name.clone(), resolved);
            }

            // Execute the node
            let outputs = node_impl.execute(&inputs).map_err(|e| {
                ComfyError::ExecutionError {
                    node: node_id.clone(),
                    message: e.to_string(),
                }
            })?;

            // Cache and store
            self.cache.put(node_id.clone(), outputs.clone());
            all_outputs.insert(node_id.clone(), outputs);
        }

        Ok(all_outputs)
    }

    /// Clear the execution cache.
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    fn resolve_input(
        &self,
        input: &InputValue,
        all_outputs: &HashMap<String, NodeOutputs>,
    ) -> ComfyResult<NodeValue> {
        match input {
            InputValue::Link(source_id, output_index) => {
                let source_outputs = all_outputs.get(source_id).ok_or_else(|| {
                    ComfyError::ExecutionError {
                        node: source_id.clone(),
                        message: "output not yet computed".to_string(),
                    }
                })?;
                let key = output_key(*output_index);
                source_outputs
                    .get(&key)
                    .cloned()
                    .ok_or_else(|| ComfyError::ExecutionError {
                        node: source_id.clone(),
                        message: format!("no output at index {output_index}"),
                    })
            }
            InputValue::Float(v) => Ok(NodeValue::Float(*v)),
            InputValue::Int(v) => Ok(NodeValue::Int(*v)),
            InputValue::String(s) => Ok(NodeValue::String(s.clone())),
            InputValue::Bool(b) => Ok(NodeValue::Bool(*b)),
            InputValue::None => Ok(NodeValue::None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::{Node, NodeMetadata};

    /// A source node that outputs a constant float value from its "value" input.
    struct SourceNode;

    impl Node for SourceNode {
        fn execute(&self, inputs: &NodeInputs) -> ComfyResult<NodeOutputs> {
            let value = inputs.get_float("value")?;
            let mut outputs = NodeOutputs::new();
            outputs.set("output_0", NodeValue::Float(value));
            Ok(outputs)
        }

        fn metadata(&self) -> NodeMetadata {
            NodeMetadata {
                name: "SourceNode".to_string(),
                display_name: "Source".to_string(),
                category: "utils".to_string(),
                description: "Outputs a constant value".to_string(),
                output_node: false,
            }
        }
    }

    /// A node that doubles its float input.
    struct DoubleNode;

    impl Node for DoubleNode {
        fn execute(&self, inputs: &NodeInputs) -> ComfyResult<NodeOutputs> {
            let value = inputs.get_float("x")?;
            let mut outputs = NodeOutputs::new();
            outputs.set("output_0", NodeValue::Float(value * 2.0));
            Ok(outputs)
        }

        fn metadata(&self) -> NodeMetadata {
            NodeMetadata {
                name: "DoubleNode".to_string(),
                display_name: "Double".to_string(),
                category: "math".to_string(),
                description: "Doubles the input value".to_string(),
                output_node: false,
            }
        }
    }

    /// A node that adds two float inputs.
    struct AddNode;

    impl Node for AddNode {
        fn execute(&self, inputs: &NodeInputs) -> ComfyResult<NodeOutputs> {
            let a = inputs.get_float("a")?;
            let b = inputs.get_float("b")?;
            let mut outputs = NodeOutputs::new();
            outputs.set("output_0", NodeValue::Float(a + b));
            Ok(outputs)
        }

        fn metadata(&self) -> NodeMetadata {
            NodeMetadata {
                name: "AddNode".to_string(),
                display_name: "Add".to_string(),
                category: "math".to_string(),
                description: "Adds two values".to_string(),
                output_node: true,
            }
        }
    }

    fn build_registry() -> NodeRegistry {
        let mut registry = NodeRegistry::new();
        registry.register("SourceNode", || Box::new(SourceNode));
        registry.register("DoubleNode", || Box::new(DoubleNode));
        registry.register("AddNode", || Box::new(AddNode));
        registry
    }

    #[test]
    fn test_executor_simple_chain() {
        // SourceNode(value=5.0) -> DoubleNode -> result should be 10.0
        let json = r#"{
            "1": {
                "class_type": "SourceNode",
                "inputs": { "value": 5.0 }
            },
            "2": {
                "class_type": "DoubleNode",
                "inputs": { "x": ["1", 0] }
            }
        }"#;

        let registry = build_registry();
        let mut executor = Executor::new(registry, 100);
        let results = executor.execute_workflow(json).unwrap();

        // Check SourceNode output
        let source_out = &results["1"];
        match source_out.get("output_0") {
            Some(NodeValue::Float(v)) => assert_eq!(*v, 5.0),
            other => panic!("Expected Float(5.0), got {other:?}"),
        }

        // Check DoubleNode output
        let double_out = &results["2"];
        match double_out.get("output_0") {
            Some(NodeValue::Float(v)) => assert_eq!(*v, 10.0),
            other => panic!("Expected Float(10.0), got {other:?}"),
        }
    }

    #[test]
    fn test_executor_diamond() {
        // Source(3.0) -> Double -> Add
        // Source(3.0) ----------> Add
        // Add(6.0 + 3.0) = 9.0
        let json = r#"{
            "1": {
                "class_type": "SourceNode",
                "inputs": { "value": 3.0 }
            },
            "2": {
                "class_type": "DoubleNode",
                "inputs": { "x": ["1", 0] }
            },
            "3": {
                "class_type": "AddNode",
                "inputs": { "a": ["2", 0], "b": ["1", 0] }
            }
        }"#;

        let registry = build_registry();
        let mut executor = Executor::new(registry, 100);
        let results = executor.execute_workflow(json).unwrap();

        let add_out = &results["3"];
        match add_out.get("output_0") {
            Some(NodeValue::Float(v)) => assert_eq!(*v, 9.0),
            other => panic!("Expected Float(9.0), got {other:?}"),
        }
    }

    #[test]
    fn test_executor_unregistered_node() {
        let json = r#"{
            "1": {
                "class_type": "NonExistentNode",
                "inputs": {}
            }
        }"#;

        let registry = NodeRegistry::new();
        let mut executor = Executor::new(registry, 100);
        let result = executor.execute_workflow(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_executor_cycle_error() {
        let json = r#"{
            "1": { "class_type": "DoubleNode", "inputs": { "x": ["2", 0] } },
            "2": { "class_type": "DoubleNode", "inputs": { "x": ["1", 0] } }
        }"#;

        let registry = build_registry();
        let mut executor = Executor::new(registry, 100);
        let result = executor.execute_workflow(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_executor_cache_clear() {
        let json = r#"{
            "1": { "class_type": "SourceNode", "inputs": { "value": 1.0 } }
        }"#;

        let registry = build_registry();
        let mut executor = Executor::new(registry, 100);

        let results = executor.execute_workflow(json).unwrap();
        assert!(results.contains_key("1"));

        executor.clear_cache();

        // Should still work after clearing cache
        let results2 = executor.execute_workflow(json).unwrap();
        assert!(results2.contains_key("1"));
    }
}
