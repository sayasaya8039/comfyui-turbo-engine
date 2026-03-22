//! Data types exchanged between the host and WASM plugin nodes.
//!
//! These structures are serialised to JSON when crossing the WASM boundary.
//! [`WasmTensor`] provides lossless conversion to/from [`comfy_core::Tensor`].

use comfy_core::{ComfyResult, Tensor};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// WasmTensor
// ---------------------------------------------------------------------------

/// A tensor representation that can be serialised across the WASM boundary.
///
/// All data is stored as `Vec<f32>` regardless of the original dtype
/// (the host converts on the boundary).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WasmTensor {
    pub data: Vec<f32>,
    pub shape: Vec<u32>,
    pub dtype: String,
}

impl WasmTensor {
    /// Convert a [`Tensor`] (F32 only) into a [`WasmTensor`].
    pub fn from_tensor(tensor: &Tensor) -> ComfyResult<Self> {
        let data = tensor.as_slice_f32()?.to_vec();
        let shape = tensor.shape().iter().map(|&s| s as u32).collect();
        Ok(Self {
            data,
            shape,
            dtype: format!("{:?}", tensor.dtype()),
        })
    }

    /// Convert back into a [`Tensor`].
    pub fn to_tensor(&self) -> ComfyResult<Tensor> {
        let shape: Vec<usize> = self.shape.iter().map(|&s| s as usize).collect();
        Tensor::from_vec_f32(self.data.clone(), shape)
    }

    /// Total number of elements implied by the shape.
    pub fn numel(&self) -> usize {
        self.shape.iter().map(|&s| s as usize).product()
    }
}

// ---------------------------------------------------------------------------
// Node metadata & port definitions
// ---------------------------------------------------------------------------

/// Describes a single input or output port of a WASM node.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WasmPortDef {
    pub name: String,
    pub data_type: String,
    pub required: bool,
}

/// Metadata for a WASM-hosted node that the host registers.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WasmNodeMetadata {
    pub name: String,
    pub category: String,
    pub description: String,
    pub inputs: Vec<WasmPortDef>,
    pub outputs: Vec<WasmPortDef>,
}

// ---------------------------------------------------------------------------
// Request / Response
// ---------------------------------------------------------------------------

/// A request sent from the host to a WASM node for execution.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WasmNodeRequest {
    pub node_name: String,
    pub inputs: std::collections::HashMap<String, serde_json::Value>,
}

/// The response a WASM node sends back after execution.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WasmNodeResponse {
    pub outputs: std::collections::HashMap<String, serde_json::Value>,
    pub error: Option<String>,
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_roundtrip() {
        let original = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let wasm_tensor = WasmTensor::from_tensor(&original).unwrap();
        let recovered = wasm_tensor.to_tensor().unwrap();

        assert_eq!(recovered.shape(), original.shape());
        assert_eq!(
            recovered.as_slice_f32().unwrap(),
            original.as_slice_f32().unwrap()
        );
    }

    #[test]
    fn test_tensor_size_check() {
        let wt = WasmTensor {
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            shape: vec![2, 3],
            dtype: "F32".to_string(),
        };
        assert_eq!(wt.numel(), 6);
        assert_eq!(wt.data.len(), wt.numel());

        // Mismatched size should fail on to_tensor
        let bad = WasmTensor {
            data: vec![1.0, 2.0],
            shape: vec![3, 3],
            dtype: "F32".to_string(),
        };
        assert!(bad.to_tensor().is_err());
    }

    #[test]
    fn test_metadata_serde() {
        let meta = WasmNodeMetadata {
            name: "BlurNode".to_string(),
            category: "image/filter".to_string(),
            description: "Gaussian blur".to_string(),
            inputs: vec![WasmPortDef {
                name: "image".to_string(),
                data_type: "TENSOR".to_string(),
                required: true,
            }],
            outputs: vec![WasmPortDef {
                name: "blurred".to_string(),
                data_type: "TENSOR".to_string(),
                required: true,
            }],
        };

        let json = serde_json::to_string(&meta).unwrap();
        let deserialized: WasmNodeMetadata = serde_json::from_str(&json).unwrap();
        assert_eq!(meta, deserialized);
    }

    #[test]
    fn test_request_response_serde() {
        let req = WasmNodeRequest {
            node_name: "BlurNode".to_string(),
            inputs: {
                let mut m = std::collections::HashMap::new();
                m.insert("radius".to_string(), serde_json::json!(5));
                m
            },
        };

        let req_json = serde_json::to_string(&req).unwrap();
        let req_de: WasmNodeRequest = serde_json::from_str(&req_json).unwrap();
        assert_eq!(req, req_de);

        let resp = WasmNodeResponse {
            outputs: {
                let mut m = std::collections::HashMap::new();
                m.insert("status".to_string(), serde_json::json!("ok"));
                m
            },
            error: None,
        };

        let resp_json = serde_json::to_string(&resp).unwrap();
        let resp_de: WasmNodeResponse = serde_json::from_str(&resp_json).unwrap();
        assert_eq!(resp, resp_de);
    }
}
