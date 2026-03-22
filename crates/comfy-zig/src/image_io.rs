use comfy_core::{ComfyError, ComfyResult, DType, Tensor};

/// Decode an image file into an [H, W, 3] U8 tensor.
///
/// Uses the `image` crate as the pure-Rust fallback implementation.
#[cfg(feature = "fallback")]
pub fn decode_image(path: &str) -> ComfyResult<Tensor> {
    use image::io::Reader as ImageReader;

    let img = ImageReader::open(path)
        .map_err(|e| ComfyError::IoError(e))?
        .decode()
        .map_err(|e| ComfyError::TensorError(format!("image decode failed: {e}")))?;

    let rgb = img.to_rgb8();
    let (width, height) = rgb.dimensions();
    let data = rgb.into_raw();

    Tensor::from_raw(data, vec![height as usize, width as usize, 3], DType::U8)
}

/// Decode stub when fallback feature is disabled.
#[cfg(not(feature = "fallback"))]
pub fn decode_image(_path: &str) -> ComfyResult<Tensor> {
    Err(ComfyError::TensorError(
        "image decoding requires 'fallback' or 'zig-native' feature".to_string(),
    ))
}

/// Encode an [H, W, 3] U8 tensor as a PNG file.
#[cfg(feature = "fallback")]
pub fn encode_png(tensor: &Tensor, path: &str) -> ComfyResult<()> {
    if tensor.dtype() != DType::U8 {
        return Err(ComfyError::TensorError(format!(
            "encode_png requires U8 tensor, got {:?}",
            tensor.dtype()
        )));
    }

    let shape = tensor.shape();
    if shape.len() != 3 || shape[2] != 3 {
        return Err(ComfyError::TensorError(format!(
            "encode_png requires [H, W, 3] shape, got {:?}",
            shape
        )));
    }

    let height = shape[0] as u32;
    let width = shape[1] as u32;
    let data = tensor.as_bytes().to_vec();

    image::save_buffer(path, &data, width, height, image::ColorType::Rgb8)
        .map_err(|e| ComfyError::TensorError(format!("PNG encode failed: {e}")))?;

    Ok(())
}

/// Encode stub when fallback feature is disabled.
#[cfg(not(feature = "fallback"))]
pub fn encode_png(_tensor: &Tensor, _path: &str) -> ComfyResult<()> {
    Err(ComfyError::TensorError(
        "image encoding requires 'fallback' or 'zig-native' feature".to_string(),
    ))
}

/// Normalize a U8 tensor [0, 255] to an F32 tensor [0.0, 1.0].
///
/// Every byte value `v` maps to `v as f32 / 255.0`.
/// Shape is preserved.
pub fn normalize_u8_to_f32(tensor: &Tensor) -> ComfyResult<Tensor> {
    if tensor.dtype() != DType::U8 {
        return Err(ComfyError::TensorError(format!(
            "normalize_u8_to_f32 requires U8 tensor, got {:?}",
            tensor.dtype()
        )));
    }

    let bytes = tensor.as_bytes();
    let floats: Vec<f32> = bytes.iter().map(|&b| b as f32 / 255.0).collect();

    Tensor::from_vec_f32(floats, tensor.shape().to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_u8_to_f32() {
        // Create a U8 tensor with known values: 0, 128, 255
        let data = vec![0u8, 128, 255];
        let tensor = Tensor::from_raw(data, vec![3], DType::U8).unwrap();

        let result = normalize_u8_to_f32(&tensor).unwrap();
        assert_eq!(result.dtype(), DType::F32);
        assert_eq!(result.shape(), &[3]);

        let slice = result.as_slice_f32().unwrap();
        assert!((slice[0] - 0.0).abs() < 1e-6, "0 -> 0.0, got {}", slice[0]);
        assert!(
            (slice[1] - 128.0 / 255.0).abs() < 1e-6,
            "128 -> ~0.502, got {}",
            slice[1]
        );
        assert!(
            (slice[2] - 1.0).abs() < 1e-6,
            "255 -> 1.0, got {}",
            slice[2]
        );
    }

    #[test]
    fn test_normalize_wrong_dtype() {
        let tensor = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let result = normalize_u8_to_f32(&tensor);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("U8"), "Error should mention U8: {msg}");
    }

    #[test]
    fn test_decode_nonexistent_file() {
        let result = decode_image("/nonexistent/path/image.png");
        assert!(result.is_err());
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        // Create a 2x2 RGB tensor
        #[rustfmt::skip]
        let data = vec![
            255, 0,   0,    // red
            0,   255, 0,    // green
            0,   0,   255,  // blue
            128, 128, 128,  // gray
        ];
        let tensor = Tensor::from_raw(data.clone(), vec![2, 2, 3], DType::U8).unwrap();

        let tmp_path = std::env::temp_dir().join("comfy_zig_test_roundtrip.png");
        let tmp_str = tmp_path.to_str().unwrap();

        // Encode
        encode_png(&tensor, tmp_str).unwrap();

        // Decode
        let decoded = decode_image(tmp_str).unwrap();
        assert_eq!(decoded.shape(), &[2, 2, 3]);
        assert_eq!(decoded.dtype(), DType::U8);
        assert_eq!(decoded.as_bytes(), &data[..]);

        // Cleanup
        let _ = std::fs::remove_file(&tmp_path);
    }
}
