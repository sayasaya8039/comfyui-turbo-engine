use comfy_core::{ComfyError, ComfyResult, DType, Tensor};

/// A quantized tensor storing INT8 values with affine quantization parameters.
///
/// The original float value can be reconstructed as:
///   `f32_value = (i8_value - zero_point) * scale`
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    /// Quantized INT8 data.
    pub data: Vec<i8>,
    /// Original tensor shape.
    pub shape: Vec<usize>,
    /// Scale factor: maps quantized range to float range.
    pub scale: f32,
    /// Zero point offset in the quantized domain.
    pub zero_point: i8,
}

/// Quantize an F32 tensor to INT8 using affine min-max quantization.
///
/// The quantization maps the float range `[min, max]` to `[-128, 127]`:
///   `quantized = clamp(round(value / scale + zero_point), -128, 127)`
///   `scale = (max - min) / 255`
///   `zero_point = clamp(round(-128 - min / scale), -128, 127)`
///
/// Dequantization: `value = (quantized - zero_point) * scale`
pub fn quantize_to_int8(tensor: &Tensor) -> ComfyResult<QuantizedTensor> {
    let slice = tensor.as_slice_f32().map_err(|_| {
        ComfyError::TensorError(format!(
            "quantize_to_int8 requires F32 tensor, got {:?}",
            tensor.dtype()
        ))
    })?;

    if slice.is_empty() {
        return Ok(QuantizedTensor {
            data: Vec::new(),
            shape: tensor.shape().to_vec(),
            scale: 1.0,
            zero_point: 0,
        });
    }

    let min_val = slice.iter().copied().fold(f32::INFINITY, f32::min);
    let max_val = slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    // Compute scale: maps the full float range to 256 quantization levels.
    let scale = if (max_val - min_val).abs() < f32::EPSILON {
        1.0
    } else {
        (max_val - min_val) / 255.0
    };

    // Zero point: the quantized value that represents float 0.0.
    // zero_point = round(-128 - min / scale), clamped to i8 range.
    let zero_point_f = -128.0 - min_val / scale;
    let zero_point = zero_point_f.round().clamp(-128.0, 127.0) as i8;

    let data: Vec<i8> = slice
        .iter()
        .map(|&v| {
            let q = v / scale + zero_point as f32;
            q.round().clamp(-128.0, 127.0) as i8
        })
        .collect();

    Ok(QuantizedTensor {
        data,
        shape: tensor.shape().to_vec(),
        scale,
        zero_point,
    })
}

/// Dequantize an INT8 quantized tensor back to F32.
///
/// Reconstructs float values using: `f32_value = (i8_value - zero_point) * scale`
pub fn dequantize_to_f32(qt: &QuantizedTensor) -> ComfyResult<Tensor> {
    let values: Vec<f32> = qt
        .data
        .iter()
        .map(|&q| (q as f32 - qt.zero_point as f32) * qt.scale)
        .collect();

    Tensor::from_vec_f32(values, qt.shape.clone())
}

/// Quantize an F32 tensor to F16 representation (stored as raw bytes).
///
/// Uses IEEE 754 half-precision format via bit manipulation:
///   - Sign: 1 bit, Exponent: 5 bits, Mantissa: 10 bits
///   - Handles special cases: NaN, Inf, subnormals
pub fn quantize_to_f16(tensor: &Tensor) -> ComfyResult<Tensor> {
    let slice = tensor.as_slice_f32().map_err(|_| {
        ComfyError::TensorError(format!(
            "quantize_to_f16 requires F32 tensor, got {:?}",
            tensor.dtype()
        ))
    })?;

    let mut f16_bytes = Vec::with_capacity(slice.len() * 2);
    for &val in slice {
        let half = f32_to_f16_bits(val);
        f16_bytes.extend_from_slice(&half.to_le_bytes());
    }

    Tensor::from_raw(f16_bytes, tensor.shape().to_vec(), DType::F16)
}

/// Convert a single f32 value to f16 bits (IEEE 754 half-precision).
fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x007F_FFFF;

    if exponent == 255 {
        // NaN or Inf
        if mantissa != 0 {
            // NaN: preserve some mantissa bits
            return sign | 0x7C00 | ((mantissa >> 13) as u16).max(1);
        }
        // Infinity
        return sign | 0x7C00;
    }

    // Re-bias exponent from f32 bias (127) to f16 bias (15)
    let new_exp = exponent - 127 + 15;

    if new_exp >= 31 {
        // Overflow -> Infinity
        return sign | 0x7C00;
    }

    if new_exp <= 0 {
        // Subnormal or zero in f16
        if new_exp < -10 {
            // Too small, flush to zero
            return sign;
        }
        // Subnormal: shift mantissa with implicit leading 1
        let full_mantissa = mantissa | 0x0080_0000;
        let shift = 1 - new_exp;
        let half_mantissa = (full_mantissa >> (13 + shift)) as u16;
        return sign | half_mantissa;
    }

    sign | ((new_exp as u16) << 10) | ((mantissa >> 13) as u16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_int8_roundtrip() {
        let original = vec![0.0f32, 0.5, 1.0, -1.0, -0.5, 0.25, 0.75, -0.75];
        let tensor = Tensor::from_vec_f32(original.clone(), vec![2, 4]).unwrap();

        let quantized = quantize_to_int8(&tensor).unwrap();
        assert_eq!(quantized.shape, vec![2, 4]);
        assert_eq!(quantized.data.len(), 8);

        let dequantized = dequantize_to_f32(&quantized).unwrap();
        let result = dequantized.as_slice_f32().unwrap();

        // Roundtrip error must be within quantization tolerance
        for (i, (&orig, &recon)) in original.iter().zip(result.iter()).enumerate() {
            let error = (orig - recon).abs();
            assert!(
                error < 0.01,
                "element {i}: original={orig}, reconstructed={recon}, error={error}"
            );
        }
    }

    #[test]
    fn test_quantization_scale_range() {
        let values: Vec<f32> = (0..256).map(|i| i as f32 / 255.0).collect();
        let tensor = Tensor::from_vec_f32(values, vec![256]).unwrap();

        let quantized = quantize_to_int8(&tensor).unwrap();
        assert!(quantized.scale > 0.0, "scale must be positive");
        assert!(
            quantized.scale < 1.0,
            "scale should be < 1.0 for [0,1] range, got {}",
            quantized.scale
        );
    }

    #[test]
    fn test_f16_quantization() {
        let values = vec![0.0f32, 1.0, -1.0, 0.5, 65504.0]; // 65504 is f16 max
        let tensor = Tensor::from_vec_f32(values, vec![5]).unwrap();

        let f16_tensor = quantize_to_f16(&tensor).unwrap();
        assert_eq!(f16_tensor.dtype(), DType::F16);
        assert_eq!(f16_tensor.shape(), &[5]);
        // F16 is 2 bytes per element
        assert_eq!(f16_tensor.byte_size(), 10);

        // Verify known f16 bit patterns
        let bytes = f16_tensor.as_bytes();
        // 0.0 in f16 = 0x0000
        assert_eq!(bytes[0], 0x00);
        assert_eq!(bytes[1], 0x00);
        // 1.0 in f16 = 0x3C00
        assert_eq!(bytes[2], 0x00);
        assert_eq!(bytes[3], 0x3C);
    }
}
