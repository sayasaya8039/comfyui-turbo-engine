use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::error::{ComfyError, ComfyResult};
use crate::tensor::Tensor;

/// Pre-allocated output key strings to avoid repeated allocations.
/// Covers output indices 0-15 which handles 99%+ of workflows.
static OUTPUT_KEYS: &[&str] = &[
    "output_0", "output_1", "output_2", "output_3",
    "output_4", "output_5", "output_6", "output_7",
    "output_8", "output_9", "output_10", "output_11",
    "output_12", "output_13", "output_14", "output_15",
];

/// Get a pre-allocated output key string for the given index.
/// Falls back to format!() for indices > 15.
pub fn output_key(index: usize) -> String {
    if index < OUTPUT_KEYS.len() {
        OUTPUT_KEYS[index].to_string()
    } else {
        format!("output_{index}")
    }
}

/// A value that can be passed between nodes.
#[derive(Debug, Clone)]
pub enum NodeValue {
    Tensor(Tensor),
    Float(f64),
    Int(i64),
    String(String),
    Bool(bool),
    List(Vec<NodeValue>),
    None,
}

impl NodeValue {
    /// Returns a human-readable type name for error messages.
    pub fn type_name(&self) -> &'static str {
        match self {
            NodeValue::Tensor(_) => "Tensor",
            NodeValue::Float(_) => "Float",
            NodeValue::Int(_) => "Int",
            NodeValue::String(_) => "String",
            NodeValue::Bool(_) => "Bool",
            NodeValue::List(_) => "List",
            NodeValue::None => "None",
        }
    }
}

/// Input values for a node execution.
#[derive(Debug, Clone, Default)]
pub struct NodeInputs {
    values: HashMap<String, NodeValue>,
}

impl NodeInputs {
    pub fn new() -> Self {
        Self {
            values: HashMap::new(),
        }
    }

    pub fn set(&mut self, name: impl Into<String>, value: NodeValue) {
        self.values.insert(name.into(), value);
    }

    pub fn get(&self, name: &str) -> Option<&NodeValue> {
        self.values.get(name)
    }

    pub fn get_tensor(&self, name: &str) -> ComfyResult<&Tensor> {
        match self.values.get(name) {
            Some(NodeValue::Tensor(t)) => Ok(t),
            Some(other) => Err(ComfyError::TypeMismatch {
                expected: "Tensor".to_string(),
                got: other.type_name().to_string(),
            }),
            None => Err(ComfyError::InvalidInput {
                node: String::new(),
                message: format!("missing input '{name}'"),
            }),
        }
    }

    pub fn get_float(&self, name: &str) -> ComfyResult<f64> {
        match self.values.get(name) {
            Some(NodeValue::Float(v)) => Ok(*v),
            Some(NodeValue::Int(v)) => Ok(*v as f64),
            Some(other) => Err(ComfyError::TypeMismatch {
                expected: "Float".to_string(),
                got: other.type_name().to_string(),
            }),
            None => Err(ComfyError::InvalidInput {
                node: String::new(),
                message: format!("missing input '{name}'"),
            }),
        }
    }

    pub fn get_int(&self, name: &str) -> ComfyResult<i64> {
        match self.values.get(name) {
            Some(NodeValue::Int(v)) => Ok(*v),
            Some(other) => Err(ComfyError::TypeMismatch {
                expected: "Int".to_string(),
                got: other.type_name().to_string(),
            }),
            None => Err(ComfyError::InvalidInput {
                node: String::new(),
                message: format!("missing input '{name}'"),
            }),
        }
    }

    pub fn get_string(&self, name: &str) -> ComfyResult<&str> {
        match self.values.get(name) {
            Some(NodeValue::String(s)) => Ok(s.as_str()),
            Some(other) => Err(ComfyError::TypeMismatch {
                expected: "String".to_string(),
                got: other.type_name().to_string(),
            }),
            None => Err(ComfyError::InvalidInput {
                node: String::new(),
                message: format!("missing input '{name}'"),
            }),
        }
    }
}

/// Output values produced by a node execution.
#[derive(Debug, Clone, Default)]
pub struct NodeOutputs {
    pub(crate) values: HashMap<String, NodeValue>,
}

impl NodeOutputs {
    pub fn new() -> Self {
        Self {
            values: HashMap::new(),
        }
    }

    pub fn set(&mut self, name: impl Into<String>, value: NodeValue) {
        self.values.insert(name.into(), value);
    }

    pub fn get(&self, name: &str) -> Option<&NodeValue> {
        self.values.get(name)
    }

    /// Get the value at a given output index (by insertion order is unreliable,
    /// so we use a convention: output_0, output_1, etc.).
    pub fn get_by_index(&self, index: usize) -> Option<&NodeValue> {
        let key = output_key(index);
        self.values.get(&key)
    }
}

/// Metadata describing a node type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeMetadata {
    pub name: String,
    pub display_name: String,
    pub category: String,
    pub description: String,
    pub output_node: bool,
}

/// The core trait that all ComfyUI nodes must implement.
pub trait Node: Send + Sync {
    fn execute(&self, inputs: &NodeInputs) -> ComfyResult<NodeOutputs>;
    fn metadata(&self) -> NodeMetadata;

    /// Returns the preferred device for this node's execution.
    ///
    /// Defaults to CPU. Nodes that benefit from GPU acceleration
    /// (e.g. KSampler, VAEDecode) should override this to return
    /// `Device::Gpu(0)` or similar.
    fn device_hint(&self) -> crate::device::Device {
        crate::device::Device::Cpu
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A simple test node that adds 1.0 to its float input.
    pub(crate) struct AddOneNode;

    impl Node for AddOneNode {
        fn execute(&self, inputs: &NodeInputs) -> ComfyResult<NodeOutputs> {
            let value = inputs.get_float("value")?;
            let mut outputs = NodeOutputs::new();
            outputs.set("output_0", NodeValue::Float(value + 1.0));
            Ok(outputs)
        }

        fn metadata(&self) -> NodeMetadata {
            NodeMetadata {
                name: "AddOne".to_string(),
                display_name: "Add One".to_string(),
                category: "math".to_string(),
                description: "Adds 1.0 to the input value".to_string(),
                output_node: false,
            }
        }
    }

    #[test]
    fn test_node_execute() {
        let node = AddOneNode;
        let mut inputs = NodeInputs::new();
        inputs.set("value", NodeValue::Float(5.0));

        let outputs = node.execute(&inputs).unwrap();
        match outputs.get("output_0") {
            Some(NodeValue::Float(v)) => assert_eq!(*v, 6.0),
            other => panic!("Expected Float(6.0), got {other:?}"),
        }
    }

    #[test]
    fn test_node_metadata() {
        let node = AddOneNode;
        let meta = node.metadata();
        assert_eq!(meta.name, "AddOne");
        assert_eq!(meta.category, "math");
        assert!(!meta.output_node);
    }

    #[test]
    fn test_node_value_types() {
        let values = vec![
            (NodeValue::Float(1.0), "Float"),
            (NodeValue::Int(42), "Int"),
            (NodeValue::String("hello".to_string()), "String"),
            (NodeValue::Bool(true), "Bool"),
            (NodeValue::None, "None"),
            (NodeValue::List(vec![NodeValue::Int(1)]), "List"),
        ];
        for (val, expected_name) in values {
            assert_eq!(val.type_name(), expected_name);
        }
    }

    #[test]
    fn test_node_inputs_get_tensor_type_mismatch() {
        let mut inputs = NodeInputs::new();
        inputs.set("x", NodeValue::Float(1.0));
        let result = inputs.get_tensor("x");
        assert!(result.is_err());
    }

    #[test]
    fn test_node_inputs_get_float_from_int() {
        let mut inputs = NodeInputs::new();
        inputs.set("x", NodeValue::Int(5));
        // get_float should accept Int and convert
        let v = inputs.get_float("x").unwrap();
        assert_eq!(v, 5.0);
    }

    #[test]
    fn test_node_inputs_missing_key() {
        let inputs = NodeInputs::new();
        assert!(inputs.get_float("missing").is_err());
        assert!(inputs.get_int("missing").is_err());
        assert!(inputs.get_string("missing").is_err());
        assert!(inputs.get_tensor("missing").is_err());
    }

    #[test]
    fn test_node_outputs_by_index() {
        let mut outputs = NodeOutputs::new();
        outputs.set("output_0", NodeValue::Float(1.0));
        outputs.set("output_1", NodeValue::Float(2.0));
        match outputs.get_by_index(0) {
            Some(NodeValue::Float(v)) => assert_eq!(*v, 1.0),
            other => panic!("Expected Float(1.0), got {other:?}"),
        }
        match outputs.get_by_index(1) {
            Some(NodeValue::Float(v)) => assert_eq!(*v, 2.0),
            other => panic!("Expected Float(2.0), got {other:?}"),
        }
        assert!(outputs.get_by_index(2).is_none());
    }

    #[test]
    fn test_node_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Box<dyn Node>>();
    }

    #[test]
    fn test_default_device_hint_is_cpu() {
        let node = AddOneNode;
        assert_eq!(node.device_hint(), crate::device::Device::Cpu);
    }
}
