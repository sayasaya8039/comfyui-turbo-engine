//! Dispatch layer: uses Zig SIMD kernels when `zig-native` feature is enabled,
//! otherwise falls back to pure-Rust implementations.

use comfy_core::{ComfyResult, Tensor};

/// SiLU activation with automatic SIMD dispatch.
pub fn silu_dispatch(x: &Tensor) -> ComfyResult<Tensor> {
    #[cfg(feature = "zig-native")]
    {
        let data = x.as_slice_f32()?;
        let mut output = vec![0.0f32; data.len()];
        unsafe {
            crate::ffi::comfy_zig_silu(data.as_ptr(), output.as_mut_ptr(), data.len());
        }
        Tensor::from_vec_f32(output, x.shape().to_vec())
    }
    #[cfg(not(feature = "zig-native"))]
    {
        crate::kernels::silu(x)
    }
}

/// Softmax with automatic SIMD dispatch.
pub fn softmax_dispatch(x: &Tensor) -> ComfyResult<Tensor> {
    #[cfg(feature = "zig-native")]
    {
        let data = x.as_slice_f32()?;
        let last_dim = *x.shape().last().unwrap_or(&1);
        let num_rows = data.len() / last_dim;
        let mut output = vec![0.0f32; data.len()];
        for row in 0..num_rows {
            let offset = row * last_dim;
            unsafe {
                crate::ffi::comfy_zig_softmax(
                    data[offset..].as_ptr(),
                    output[offset..].as_mut_ptr(),
                    last_dim,
                );
            }
        }
        Tensor::from_vec_f32(output, x.shape().to_vec())
    }
    #[cfg(not(feature = "zig-native"))]
    {
        crate::kernels::softmax(x)
    }
}

/// Layer normalization with automatic SIMD dispatch.
pub fn layer_norm_dispatch(x: &Tensor, eps: f32) -> ComfyResult<Tensor> {
    #[cfg(feature = "zig-native")]
    {
        let data = x.as_slice_f32()?;
        let last_dim = *x.shape().last().unwrap_or(&1);
        let num_rows = data.len() / last_dim;
        let mut output = vec![0.0f32; data.len()];
        for row in 0..num_rows {
            let offset = row * last_dim;
            unsafe {
                crate::ffi::comfy_zig_layer_norm(
                    data[offset..].as_ptr(),
                    output[offset..].as_mut_ptr(),
                    last_dim,
                    eps,
                );
            }
        }
        Tensor::from_vec_f32(output, x.shape().to_vec())
    }
    #[cfg(not(feature = "zig-native"))]
    {
        crate::kernels::layer_norm(x, eps)
    }
}

/// GELU activation with automatic SIMD dispatch.
pub fn gelu_dispatch(x: &Tensor) -> ComfyResult<Tensor> {
    #[cfg(feature = "zig-native")]
    {
        let data = x.as_slice_f32()?;
        let mut output = vec![0.0f32; data.len()];
        unsafe {
            crate::ffi::comfy_zig_gelu(data.as_ptr(), output.as_mut_ptr(), data.len());
        }
        Tensor::from_vec_f32(output, x.shape().to_vec())
    }
    #[cfg(not(feature = "zig-native"))]
    {
        crate::kernels::gelu(x)
    }
}

/// Group normalization with automatic SIMD dispatch.
pub fn group_norm_dispatch(x: &Tensor, num_groups: usize, eps: f32) -> ComfyResult<Tensor> {
    #[cfg(feature = "zig-native")]
    {
        let shape = x.shape();
        if shape.len() < 3 {
            return crate::kernels::group_norm(x, num_groups, eps);
        }
        let batch = shape[0];
        let channels = shape[1];
        let spatial: usize = shape[2..].iter().product();
        let data = x.as_slice_f32()?;
        let mut output = vec![0.0f32; data.len()];
        unsafe {
            crate::ffi::comfy_zig_group_norm(
                data.as_ptr(),
                output.as_mut_ptr(),
                batch,
                channels,
                spatial,
                num_groups,
                eps,
            );
        }
        Tensor::from_vec_f32(output, shape.to_vec())
    }
    #[cfg(not(feature = "zig-native"))]
    {
        crate::kernels::group_norm(x, num_groups, eps)
    }
}

/// Fused multiply-add with automatic SIMD dispatch.
pub fn fused_multiply_add_dispatch(a: &Tensor, b: &Tensor, scale: f32) -> ComfyResult<Tensor> {
    #[cfg(feature = "zig-native")]
    {
        let a_data = a.as_slice_f32()?;
        let b_data = b.as_slice_f32()?;
        if a_data.len() != b_data.len() {
            return crate::simd::fused_multiply_add(a, b, scale);
        }
        let mut output = vec![0.0f32; a_data.len()];
        unsafe {
            crate::ffi::comfy_zig_fused_multiply_add(
                a_data.as_ptr(),
                b_data.as_ptr(),
                output.as_mut_ptr(),
                a_data.len(),
                scale,
            );
        }
        Tensor::from_vec_f32(output, a.shape().to_vec())
    }
    #[cfg(not(feature = "zig-native"))]
    {
        crate::simd::fused_multiply_add(a, b, scale)
    }
}

/// Elementwise clamp with automatic SIMD dispatch.
pub fn clamp_dispatch(x: &Tensor, min_val: f32, max_val: f32) -> ComfyResult<Tensor> {
    #[cfg(feature = "zig-native")]
    {
        let data = x.as_slice_f32()?;
        let mut output = vec![0.0f32; data.len()];
        unsafe {
            crate::ffi::comfy_zig_clamp(
                data.as_ptr(),
                output.as_mut_ptr(),
                data.len(),
                min_val,
                max_val,
            );
        }
        Tensor::from_vec_f32_fast(output, x.shape().to_vec())
    }
    #[cfg(not(feature = "zig-native"))]
    {
        crate::kernels::clamp(x, min_val, max_val)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_silu_dispatch() {
        let x = Tensor::from_vec_f32(vec![0.0, 1.0, -1.0], vec![3]).unwrap();
        let result = silu_dispatch(&x).unwrap();
        let data = result.as_slice_f32().unwrap();
        assert!(data[0].abs() < 1e-6);
        assert!((data[1] - 0.7311).abs() < 0.01);
    }

    #[test]
    fn test_gelu_dispatch() {
        let x = Tensor::from_vec_f32(vec![0.0, 1.0, -1.0], vec![3]).unwrap();
        let result = gelu_dispatch(&x).unwrap();
        let data = result.as_slice_f32().unwrap();
        assert!(data[0].abs() < 1e-6);
        assert!((data[1] - 0.8412).abs() < 0.01);
    }

    #[test]
    fn test_softmax_dispatch() {
        let x = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]).unwrap();
        let result = softmax_dispatch(&x).unwrap();
        let data = result.as_slice_f32().unwrap();
        let sum: f32 = data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_layer_norm_dispatch() {
        let x = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let result = layer_norm_dispatch(&x, 1e-5).unwrap();
        let data = result.as_slice_f32().unwrap();
        let mean: f32 = data[0..2].iter().sum::<f32>() / 2.0;
        assert!(mean.abs() < 1e-4);
    }

    #[test]
    fn test_fused_multiply_add_dispatch() {
        let a = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let b = Tensor::from_vec_f32(vec![10.0, 20.0, 30.0], vec![3]).unwrap();
        let result = fused_multiply_add_dispatch(&a, &b, 2.0).unwrap();
        let data = result.as_slice_f32().unwrap();
        assert!((data[0] - 12.0).abs() < 1e-5);
        assert!((data[1] - 24.0).abs() < 1e-5);
        assert!((data[2] - 36.0).abs() < 1e-5);
    }

    #[test]
    fn test_clamp_dispatch() {
        let x = Tensor::from_vec_f32(vec![-2.0, 0.5, 1.5, 3.0], vec![4]).unwrap();
        let result = clamp_dispatch(&x, 0.0, 1.0).unwrap();
        let data = result.as_slice_f32().unwrap();
        assert!((data[0] - 0.0).abs() < 1e-6);
        assert!((data[1] - 0.5).abs() < 1e-6);
        assert!((data[2] - 1.0).abs() < 1e-6);
        assert!((data[3] - 1.0).abs() < 1e-6);
    }
}
