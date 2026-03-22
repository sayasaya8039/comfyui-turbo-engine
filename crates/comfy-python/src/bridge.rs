use comfy_core::{ComfyError, ComfyResult, DType, Tensor};
use serde::{Deserialize, Serialize};

/// Metadata describing a tensor's shape and dtype for serialization.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TensorMeta {
    pub shape: Vec<usize>,
    pub dtype: String,
    pub byte_size: usize,
}

impl TensorMeta {
    /// Build metadata from an existing tensor.
    pub fn from_tensor(tensor: &Tensor) -> Self {
        let dtype_str = match tensor.dtype() {
            DType::F32 => "float32",
            DType::F16 => "float16",
            DType::BF16 => "bfloat16",
            DType::I8 => "int8",
            DType::I32 => "int32",
            DType::U8 => "uint8",
        };
        Self {
            shape: tensor.shape().to_vec(),
            dtype: dtype_str.to_string(),
            byte_size: tensor.byte_size(),
        }
    }

    /// Convert the dtype string back to a `DType`.
    pub fn to_dtype(&self) -> ComfyResult<DType> {
        match self.dtype.as_str() {
            "float32" | "f32" => Ok(DType::F32),
            "float16" | "f16" => Ok(DType::F16),
            "bfloat16" | "bf16" => Ok(DType::BF16),
            "int8" | "i8" => Ok(DType::I8),
            "int32" | "i32" => Ok(DType::I32),
            "uint8" | "u8" => Ok(DType::U8),
            other => Err(ComfyError::TensorError(format!(
                "unknown dtype: {other}"
            ))),
        }
    }
}

/// Serialize a tensor into raw bytes and its metadata.
pub fn tensor_to_bytes(tensor: &Tensor) -> (Vec<u8>, TensorMeta) {
    let meta = TensorMeta::from_tensor(tensor);
    let bytes = tensor.as_bytes().to_vec();
    (bytes, meta)
}

/// Reconstruct a tensor from raw bytes and metadata.
pub fn tensor_from_bytes(bytes: Vec<u8>, meta: &TensorMeta) -> ComfyResult<Tensor> {
    let dtype = meta.to_dtype()?;
    Tensor::from_raw(bytes, meta.shape.clone(), dtype)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_meta_roundtrip() {
        let tensor = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let meta = TensorMeta::from_tensor(&tensor);

        assert_eq!(meta.shape, vec![2, 2]);
        assert_eq!(meta.dtype, "float32");
        assert_eq!(meta.byte_size, 16); // 4 elements * 4 bytes

        // Serialize and deserialize the metadata
        let json = serde_json::to_string(&meta).unwrap();
        let meta2: TensorMeta = serde_json::from_str(&json).unwrap();
        assert_eq!(meta, meta2);
    }

    #[test]
    fn test_bytes_roundtrip() {
        let original = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let (bytes, meta) = tensor_to_bytes(&original);

        let restored = tensor_from_bytes(bytes, &meta).unwrap();
        assert_eq!(restored.shape(), original.shape());
        assert_eq!(restored.dtype(), original.dtype());
        assert_eq!(restored.as_bytes(), original.as_bytes());
    }

    #[test]
    fn test_unknown_dtype_error() {
        let meta = TensorMeta {
            shape: vec![2, 2],
            dtype: "complex128".to_string(),
            byte_size: 64,
        };
        let result = meta.to_dtype();
        assert!(result.is_err());
    }
}
