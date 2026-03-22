use comfy_core::{ComfyError, ComfyResult, DType, Tensor};

/// Convert a comfy-core Tensor (F32) into a Vec<f32> and its shape.
///
/// Returns an error if the tensor's dtype is not F32.
pub fn tensor_to_f32_vec(tensor: &Tensor) -> ComfyResult<(Vec<f32>, Vec<usize>)> {
    if tensor.dtype() != DType::F32 {
        return Err(ComfyError::InferenceError(format!(
            "tensor_to_f32_vec requires F32 dtype, got {:?}",
            tensor.dtype()
        )));
    }

    let slice = tensor.as_slice_f32()?;
    let data = slice.to_vec();
    let shape = tensor.shape().to_vec();
    Ok((data, shape))
}

/// Create a comfy-core Tensor from an f32 slice and shape.
///
/// The number of elements in `data` must match the product of dimensions in `shape`.
/// Panics via `ComfyResult` propagation is left to the caller; this function
/// returns a `Tensor` directly since `from_vec_f32` validates internally.
pub fn f32_slice_to_tensor(data: &[f32], shape: Vec<usize>) -> ComfyResult<Tensor> {
    Tensor::from_vec_f32(data.to_vec(), shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_roundtrip() {
        let original_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let original_shape = vec![2, 3];

        // Create a Tensor from the original data
        let tensor = Tensor::from_vec_f32(original_data.clone(), original_shape.clone()).unwrap();

        // Convert Tensor -> (Vec<f32>, Vec<usize>)
        let (extracted_data, extracted_shape) = tensor_to_f32_vec(&tensor).unwrap();
        assert_eq!(extracted_data, original_data);
        assert_eq!(extracted_shape, original_shape);

        // Convert back (Vec<f32>, Vec<usize>) -> Tensor
        let reconstructed = f32_slice_to_tensor(&extracted_data, extracted_shape).unwrap();
        assert_eq!(reconstructed.shape(), original_shape.as_slice());
        assert_eq!(reconstructed.dtype(), DType::F32);
        assert_eq!(reconstructed.numel(), 6);

        // Verify the actual float values match
        let final_slice = reconstructed.as_slice_f32().unwrap();
        assert_eq!(final_slice, original_data.as_slice());
    }

    #[test]
    fn test_non_f32_returns_error() {
        // Create an I8 tensor
        let data = vec![1u8, 2, 3, 4];
        let tensor = Tensor::from_raw(data, vec![4], DType::I8).unwrap();

        let result = tensor_to_f32_vec(&tensor);
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("F32"),
            "expected error to mention F32, got: {err_msg}"
        );
    }

    #[test]
    fn test_f32_slice_to_tensor_shape_mismatch() {
        let data = vec![1.0f32, 2.0, 3.0];
        let result = f32_slice_to_tensor(&data, vec![2, 3]);
        assert!(result.is_err(), "shape [2,3] requires 6 elements but got 3");
    }

    #[test]
    fn test_single_element_tensor() {
        let tensor = Tensor::from_vec_f32(vec![42.0], vec![1]).unwrap();
        let (data, shape) = tensor_to_f32_vec(&tensor).unwrap();
        assert_eq!(data, vec![42.0]);
        assert_eq!(shape, vec![1]);
    }

    #[test]
    fn test_u8_tensor_conversion_fails() {
        let tensor = Tensor::zeros(vec![4], DType::U8);
        let result = tensor_to_f32_vec(&tensor);
        assert!(result.is_err());
    }
}
