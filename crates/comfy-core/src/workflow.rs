use std::collections::HashMap;

use serde_json::Value;

use crate::error::{ComfyError, ComfyResult};

/// A parsed workflow ready for DAG construction and execution.
#[derive(Debug, Clone)]
pub struct Workflow {
    pub nodes: HashMap<String, WorkflowNode>,
}

/// A single node in the workflow.
#[derive(Debug, Clone)]
pub struct WorkflowNode {
    pub id: String,
    pub class_type: String,
    pub inputs: HashMap<String, InputValue>,
}

/// An input value: either a link to another node's output or a literal.
#[derive(Debug, Clone)]
pub enum InputValue {
    /// Link to another node: (source_node_id, output_index).
    Link(String, usize),
    Float(f64),
    Int(i64),
    String(String),
    Bool(bool),
    None,
}

impl Workflow {
    /// Parse a ComfyUI-format JSON workflow.
    ///
    /// Expected format:
    /// ```json
    /// {
    ///   "1": { "class_type": "KSampler", "inputs": { "seed": 42, "model": ["2", 0] } },
    ///   "2": { "class_type": "CheckpointLoader", "inputs": { "ckpt_name": "v1.safetensors" } }
    /// }
    /// ```
    ///
    /// Links are JSON arrays of [node_id_string_or_number, output_index].
    pub fn from_json(json: &str) -> ComfyResult<Self> {
        let root: Value = serde_json::from_str(json)?;
        let obj = root
            .as_object()
            .ok_or_else(|| ComfyError::TensorError("workflow must be a JSON object".to_string()))?;

        let mut nodes = HashMap::new();

        for (node_id, node_val) in obj {
            let node_obj = node_val.as_object().ok_or_else(|| {
                ComfyError::InvalidInput {
                    node: node_id.clone(),
                    message: "node must be a JSON object".to_string(),
                }
            })?;

            let class_type = node_obj
                .get("class_type")
                .and_then(|v| v.as_str())
                .ok_or_else(|| ComfyError::InvalidInput {
                    node: node_id.clone(),
                    message: "missing or invalid class_type".to_string(),
                })?
                .to_string();

            let mut inputs = HashMap::new();

            if let Some(inputs_val) = node_obj.get("inputs") {
                if let Some(inputs_obj) = inputs_val.as_object() {
                    for (input_name, input_val) in inputs_obj {
                        let parsed = Self::parse_input_value(input_val);
                        inputs.insert(input_name.clone(), parsed);
                    }
                }
            }

            nodes.insert(
                node_id.clone(),
                WorkflowNode {
                    id: node_id.clone(),
                    class_type,
                    inputs,
                },
            );
        }

        Ok(Workflow { nodes })
    }

    fn parse_input_value(val: &Value) -> InputValue {
        // Check if it's a link: [node_id, output_index]
        if let Some(arr) = val.as_array() {
            if arr.len() == 2 {
                let source_id = if let Some(s) = arr[0].as_str() {
                    Some(s.to_string())
                } else if let Some(n) = arr[0].as_u64() {
                    Some(n.to_string())
                } else {
                    None
                };

                let output_index = arr[1].as_u64().map(|n| n as usize);

                if let (Some(id), Some(idx)) = (source_id, output_index) {
                    return InputValue::Link(id, idx);
                }
            }
        }

        // Literal values
        if let Some(b) = val.as_bool() {
            return InputValue::Bool(b);
        }
        if let Some(n) = val.as_i64() {
            // Check if it could also be a float (has decimal part)
            if val.is_f64() {
                return InputValue::Float(val.as_f64().unwrap());
            }
            return InputValue::Int(n);
        }
        if let Some(f) = val.as_f64() {
            return InputValue::Float(f);
        }
        if let Some(s) = val.as_str() {
            return InputValue::String(s.to_string());
        }
        if val.is_null() {
            return InputValue::None;
        }

        InputValue::None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_workflow() {
        let json = r#"{
            "1": {
                "class_type": "CheckpointLoader",
                "inputs": {
                    "ckpt_name": "v1-5.safetensors"
                }
            },
            "2": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": 42,
                    "model": ["1", 0]
                }
            }
        }"#;

        let workflow = Workflow::from_json(json).unwrap();
        assert_eq!(workflow.nodes.len(), 2);

        let node1 = &workflow.nodes["1"];
        assert_eq!(node1.class_type, "CheckpointLoader");
        assert_eq!(node1.id, "1");

        let node2 = &workflow.nodes["2"];
        assert_eq!(node2.class_type, "KSampler");
    }

    #[test]
    fn test_parse_link() {
        let json = r#"{
            "1": {
                "class_type": "Source",
                "inputs": {}
            },
            "2": {
                "class_type": "Process",
                "inputs": {
                    "data": ["1", 0],
                    "config": ["1", 1]
                }
            }
        }"#;

        let workflow = Workflow::from_json(json).unwrap();
        let node2 = &workflow.nodes["2"];

        match &node2.inputs["data"] {
            InputValue::Link(id, idx) => {
                assert_eq!(id, "1");
                assert_eq!(*idx, 0);
            }
            other => panic!("Expected Link, got {other:?}"),
        }

        match &node2.inputs["config"] {
            InputValue::Link(id, idx) => {
                assert_eq!(id, "1");
                assert_eq!(*idx, 1);
            }
            other => panic!("Expected Link, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_link_numeric_id() {
        // ComfyUI sometimes uses numeric IDs in links
        let json = r#"{
            "1": {
                "class_type": "Source",
                "inputs": {}
            },
            "2": {
                "class_type": "Process",
                "inputs": {
                    "data": [1, 0]
                }
            }
        }"#;

        let workflow = Workflow::from_json(json).unwrap();
        let node2 = &workflow.nodes["2"];

        match &node2.inputs["data"] {
            InputValue::Link(id, idx) => {
                assert_eq!(id, "1");
                assert_eq!(*idx, 0);
            }
            other => panic!("Expected Link, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_literal_values() {
        let json = r#"{
            "1": {
                "class_type": "Config",
                "inputs": {
                    "seed": 42,
                    "cfg_scale": 7.5,
                    "name": "test",
                    "enabled": true,
                    "nothing": null
                }
            }
        }"#;

        let workflow = Workflow::from_json(json).unwrap();
        let node = &workflow.nodes["1"];

        match &node.inputs["seed"] {
            InputValue::Int(v) => assert_eq!(*v, 42),
            other => panic!("Expected Int(42), got {other:?}"),
        }
        match &node.inputs["cfg_scale"] {
            InputValue::Float(v) => assert!((v - 7.5).abs() < f64::EPSILON),
            other => panic!("Expected Float(7.5), got {other:?}"),
        }
        match &node.inputs["name"] {
            InputValue::String(s) => assert_eq!(s, "test"),
            other => panic!("Expected String, got {other:?}"),
        }
        match &node.inputs["enabled"] {
            InputValue::Bool(b) => assert!(*b),
            other => panic!("Expected Bool(true), got {other:?}"),
        }
        match &node.inputs["nothing"] {
            InputValue::None => {}
            other => panic!("Expected None, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_invalid_json() {
        let result = Workflow::from_json("not json");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_missing_class_type() {
        let json = r#"{ "1": { "inputs": {} } }"#;
        let result = Workflow::from_json(json);
        assert!(result.is_err());
    }
}
