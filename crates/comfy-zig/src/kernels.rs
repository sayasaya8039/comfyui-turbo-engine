//! CPU kernels for neural network operations.
//!
//! All kernels operate on F32 tensors and return new tensors (immutable).
//! These are pure-Rust fallback implementations; future `zig-native` builds
//! will dispatch to SIMD-optimized Zig code.

use comfy_core::{ComfyError, ComfyResult, Tensor};

// ---------------------------------------------------------------------------
// GEMM — Tiled General Matrix Multiply
// ---------------------------------------------------------------------------

/// Tiled matrix multiplication: `[M, K] × [K, N] → [M, N]`.
///
/// Uses 32×32 tiling for cache-friendly access patterns.
/// Both inputs must be 2-D F32 tensors with compatible inner dimensions.
pub fn gemm(a: &Tensor, b: &Tensor) -> ComfyResult<Tensor> {
    let a_shape = a.shape();
    let b_shape = b.shape();

    if a_shape.len() != 2 || b_shape.len() != 2 {
        return Err(ComfyError::TensorError(format!(
            "gemm requires 2-D tensors, got shapes {:?} and {:?}",
            a_shape, b_shape
        )));
    }

    let m = a_shape[0];
    let k = a_shape[1];
    let k2 = b_shape[0];
    let n = b_shape[1];

    if k != k2 {
        return Err(ComfyError::TensorError(format!(
            "gemm inner dimensions mismatch: a=[{},{}], b=[{},{}]",
            m, k, k2, n
        )));
    }

    let a_data = a.as_slice_f32()?;
    let b_data = b.as_slice_f32()?;

    // For large matrices, parallelize rows with rayon
    if m >= 64 {
        use rayon::prelude::*;
        let c_chunks: Vec<Vec<f32>> = (0..m)
            .into_par_iter()
            .map(|i| {
                let mut row = vec![0.0f32; n];
                for p in 0..k {
                    let a_val = a_data[i * k + p];
                    for j in 0..n {
                        row[j] += a_val * b_data[p * n + j];
                    }
                }
                row
            })
            .collect();
        let c: Vec<f32> = c_chunks.into_iter().flatten().collect();
        return Tensor::from_vec_f32(c, vec![m, n]);
    }

    // Small matrices: tiled sequential multiplication for cache utilization
    let mut c = vec![0.0f32; m * n];

    const TILE: usize = 32;

    let mut ii = 0;
    while ii < m {
        let i_end = (ii + TILE).min(m);
        let mut kk = 0;
        while kk < k {
            let k_end = (kk + TILE).min(k);
            let mut jj = 0;
            while jj < n {
                let j_end = (jj + TILE).min(n);

                for i in ii..i_end {
                    for p in kk..k_end {
                        let a_val = a_data[i * k + p];
                        for j in jj..j_end {
                            c[i * n + j] += a_val * b_data[p * n + j];
                        }
                    }
                }

                jj += TILE;
            }
            kk += TILE;
        }
        ii += TILE;
    }

    Tensor::from_vec_f32(c, vec![m, n])
}

// ---------------------------------------------------------------------------
// Layer Normalization
// ---------------------------------------------------------------------------

/// Layer normalization: normalize each row to mean ≈ 0, std ≈ 1.
///
/// Input must be a 2-D F32 tensor `[rows, cols]`. Each row is independently
/// normalized: `(x - mean) / sqrt(var + eps)`.
pub fn layer_norm(x: &Tensor, eps: f32) -> ComfyResult<Tensor> {
    let shape = x.shape();
    if shape.len() < 2 {
        return Err(ComfyError::TensorError(format!(
            "layer_norm requires at least 2-D tensor, got shape {:?}",
            shape
        )));
    }

    let data = x.as_slice_f32()?;
    let last_dim = *shape.last().unwrap();
    let num_rows = data.len() / last_dim;
    let mut result = Vec::with_capacity(data.len());

    for row_idx in 0..num_rows {
        let start = row_idx * last_dim;
        let row = &data[start..start + last_dim];

        // Compute mean
        let mean = row.iter().sum::<f32>() / last_dim as f32;

        // Compute variance
        let var = row.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / last_dim as f32;

        let inv_std = 1.0 / (var + eps).sqrt();

        for &v in row {
            result.push((v - mean) * inv_std);
        }
    }

    Tensor::from_vec_f32(result, shape.to_vec())
}

// ---------------------------------------------------------------------------
// Softmax
// ---------------------------------------------------------------------------

/// Numerically stable softmax along the last dimension.
///
/// For each row: `exp(x - max(x)) / sum(exp(x - max(x)))`.
/// Input must be at least 1-D F32 tensor.
pub fn softmax(x: &Tensor) -> ComfyResult<Tensor> {
    let shape = x.shape();
    if shape.is_empty() {
        return Err(ComfyError::TensorError(
            "softmax requires non-empty shape".to_string(),
        ));
    }

    let data = x.as_slice_f32()?;
    let last_dim = *shape.last().unwrap();
    let num_rows = data.len() / last_dim;
    let mut result = Vec::with_capacity(data.len());

    for row_idx in 0..num_rows {
        let start = row_idx * last_dim;
        let row = &data[start..start + last_dim];

        // Subtract max for numerical stability
        let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let exps: Vec<f32> = row.iter().map(|&v| (v - max_val).exp()).collect();
        let sum: f32 = exps.iter().sum();

        for e in &exps {
            result.push(e / sum);
        }
    }

    Tensor::from_vec_f32(result, shape.to_vec())
}

// ---------------------------------------------------------------------------
// SiLU (Sigmoid Linear Unit)
// ---------------------------------------------------------------------------

/// SiLU activation: `x * sigmoid(x)` where `sigmoid(x) = 1 / (1 + exp(-x))`.
///
/// Input must be an F32 tensor of any shape.
pub fn silu(x: &Tensor) -> ComfyResult<Tensor> {
    let data = x.as_slice_f32()?;
    let result: Vec<f32> = data
        .iter()
        .map(|&v| {
            let sigmoid = 1.0 / (1.0 + (-v).exp());
            v * sigmoid
        })
        .collect();

    Tensor::from_vec_f32(result, x.shape().to_vec())
}

// ---------------------------------------------------------------------------
// GELU (Gaussian Error Linear Unit)
// ---------------------------------------------------------------------------

/// Approximate GELU activation.
///
/// Uses the tanh approximation:
/// `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`
///
/// Input must be an F32 tensor of any shape.
pub fn gelu(x: &Tensor) -> ComfyResult<Tensor> {
    let data = x.as_slice_f32()?;
    let sqrt_2_over_pi = (2.0f32 / std::f32::consts::PI).sqrt();

    let result: Vec<f32> = data
        .iter()
        .map(|&v| {
            let inner = sqrt_2_over_pi * (v + 0.044715 * v * v * v);
            0.5 * v * (1.0 + inner.tanh())
        })
        .collect();

    Tensor::from_vec_f32(result, x.shape().to_vec())
}

// ---------------------------------------------------------------------------
// Group Normalization
// ---------------------------------------------------------------------------

/// Group normalization for tensors of shape `[batch, channels, spatial...]`.
///
/// Divides channels into `num_groups` groups and normalizes each group
/// independently: `(x - mean) / sqrt(var + eps)`.
///
/// Requires `channels % num_groups == 0` and at least 3 dimensions.
pub fn group_norm(x: &Tensor, num_groups: usize, eps: f32) -> ComfyResult<Tensor> {
    let shape = x.shape();
    if shape.len() < 3 {
        return Err(ComfyError::TensorError(format!(
            "group_norm requires at least 3-D tensor [batch, channels, spatial...], got shape {:?}",
            shape
        )));
    }

    let batch = shape[0];
    let channels = shape[1];
    let spatial: usize = shape[2..].iter().product();

    if channels % num_groups != 0 {
        return Err(ComfyError::TensorError(format!(
            "group_norm: channels ({}) must be divisible by num_groups ({})",
            channels, num_groups
        )));
    }

    let channels_per_group = channels / num_groups;
    let data = x.as_slice_f32()?;
    let mut result = vec![0.0f32; data.len()];

    for b in 0..batch {
        for g in 0..num_groups {
            let ch_start = g * channels_per_group;
            let ch_end = ch_start + channels_per_group;
            let group_size = channels_per_group * spatial;

            // Compute mean over the group
            let mut sum = 0.0f32;
            for c in ch_start..ch_end {
                for s in 0..spatial {
                    let idx = b * (channels * spatial) + c * spatial + s;
                    sum += data[idx];
                }
            }
            let mean = sum / group_size as f32;

            // Compute variance
            let mut var_sum = 0.0f32;
            for c in ch_start..ch_end {
                for s in 0..spatial {
                    let idx = b * (channels * spatial) + c * spatial + s;
                    let diff = data[idx] - mean;
                    var_sum += diff * diff;
                }
            }
            let var = var_sum / group_size as f32;
            let inv_std = 1.0 / (var + eps).sqrt();

            // Normalize
            for c in ch_start..ch_end {
                for s in 0..spatial {
                    let idx = b * (channels * spatial) + c * spatial + s;
                    result[idx] = (data[idx] - mean) * inv_std;
                }
            }
        }
    }

    Tensor::from_vec_f32(result, shape.to_vec())
}

// ---------------------------------------------------------------------------
// Canny Edge Detection (FP32-safe, matching ComfyUI v0.18.1)
// ---------------------------------------------------------------------------

/// Simple Sobel-based edge detection on a single-channel F32 image.
///
/// Input: 2-D F32 tensor [H, W] with values in [0.0, 1.0].
/// Output: 2-D F32 tensor [H, W] with edge magnitudes.
/// Forces F32 computation regardless of input dtype (ComfyUI v0.18.1 fix).
pub fn sobel_edges(x: &Tensor) -> ComfyResult<Tensor> {
    let shape = x.shape();
    if shape.len() != 2 {
        return Err(ComfyError::TensorError(format!(
            "sobel_edges requires 2-D tensor [H, W], got shape {:?}", shape
        )));
    }

    let h = shape[0];
    let w = shape[1];
    let data = x.as_slice_f32()?;
    let mut edges = vec![0.0f32; h * w];

    // Sobel kernels
    for y in 1..h.saturating_sub(1) {
        for x_pos in 1..w.saturating_sub(1) {
            let idx = |dy: usize, dx: usize| -> f32 {
                data[(y + dy - 1) * w + (x_pos + dx - 1)]
            };

            // Gx = [[-1,0,1],[-2,0,2],[-1,0,1]]
            let gx = -idx(0, 0) + idx(0, 2)
                   - 2.0 * idx(1, 0) + 2.0 * idx(1, 2)
                   - idx(2, 0) + idx(2, 2);

            // Gy = [[-1,-2,-1],[0,0,0],[1,2,1]]
            let gy = -idx(0, 0) - 2.0 * idx(0, 1) - idx(0, 2)
                   + idx(2, 0) + 2.0 * idx(2, 1) + idx(2, 2);

            edges[y * w + x_pos] = (gx * gx + gy * gy).sqrt();
        }
    }

    Tensor::from_vec_f32_fast(edges, shape.to_vec())
}

// ---------------------------------------------------------------------------
// FP16 <-> F32 Conversion
// ---------------------------------------------------------------------------

/// Convert F32 tensor to F16 bytes.
/// Each f32 is converted to IEEE 754 half-precision (2 bytes).
pub fn f32_to_f16(x: &Tensor) -> ComfyResult<Tensor> {
    let data = x.as_slice_f32()?;
    let mut f16_bytes = Vec::with_capacity(data.len() * 2);
    for &val in data {
        let f16_val = f32_to_f16_scalar(val);
        f16_bytes.extend_from_slice(&f16_val.to_le_bytes());
    }
    Tensor::from_raw(f16_bytes, x.shape().to_vec(), comfy_core::DType::F16)
}

/// Convert F16 tensor to F32.
pub fn f16_to_f32(x: &Tensor) -> ComfyResult<Tensor> {
    if x.dtype() != comfy_core::DType::F16 {
        return Err(ComfyError::TensorError(format!(
            "f16_to_f32 requires F16 tensor, got {:?}",
            x.dtype()
        )));
    }
    let bytes = x.as_bytes();
    let mut f32_vals = Vec::with_capacity(x.numel());
    for chunk in bytes.chunks_exact(2) {
        let f16_bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        f32_vals.push(f16_to_f32_scalar(f16_bits));
    }
    Tensor::from_vec_f32_fast(f32_vals, x.shape().to_vec())
}

/// Elementwise clamp.
pub fn clamp(x: &Tensor, min_val: f32, max_val: f32) -> ComfyResult<Tensor> {
    let data = x.as_slice_f32()?;
    let result: Vec<f32> = data.iter().map(|&v| v.max(min_val).min(max_val)).collect();
    Tensor::from_vec_f32_fast(result, x.shape().to_vec())
}

// Scalar FP16 helpers

fn f32_to_f16_scalar(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x7FFFFF;

    if exp == 0xFF {
        // Inf/NaN
        return (sign | 0x7C00 | if mantissa != 0 { 0x200 } else { 0 }) as u16;
    }

    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        return (sign | 0x7C00) as u16; // overflow -> Inf
    }
    if new_exp <= 0 {
        return sign as u16; // underflow -> 0
    }

    (sign | ((new_exp as u32) << 10) | (mantissa >> 13)) as u16
}

fn f16_to_f32_scalar(bits: u16) -> f32 {
    let sign = ((bits as u32) << 16) & 0x80000000;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mantissa = (bits & 0x3FF) as u32;

    if exp == 0 {
        if mantissa == 0 {
            return f32::from_bits(sign); // +/-0
        }
        // Denormalized
        let mut m = mantissa;
        let mut e = 1u32;
        while (m & 0x400) == 0 {
            m <<= 1;
            e += 1;
        }
        let new_exp = (127 - 15 + 1 - e) << 23;
        let new_mantissa = (m & 0x3FF) << 13;
        return f32::from_bits(sign | new_exp | new_mantissa);
    }

    if exp == 31 {
        let f32_exp = 0xFF << 23;
        let f32_mantissa = mantissa << 13;
        return f32::from_bits(sign | f32_exp | f32_mantissa);
    }

    let f32_exp = (exp + 127 - 15) << 23;
    let f32_mantissa = mantissa << 13;
    f32::from_bits(sign | f32_exp | f32_mantissa)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- GEMM tests --

    #[test]
    fn test_gemm_2x3_times_3x2() {
        // A = [[1,2,3],[4,5,6]] (2x3)
        // B = [[7,8],[9,10],[11,12]] (3x2)
        // C = [[1*7+2*9+3*11, 1*8+2*10+3*12],
        //      [4*7+5*9+6*11, 4*8+5*10+6*12]]
        //   = [[58, 64], [139, 154]]
        let a = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let b = Tensor::from_vec_f32(
            vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            vec![3, 2],
        )
        .unwrap();

        let c = gemm(&a, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);

        let data = c.as_slice_f32().unwrap();
        assert!((data[0] - 58.0).abs() < 1e-4, "c[0,0]={}", data[0]);
        assert!((data[1] - 64.0).abs() < 1e-4, "c[0,1]={}", data[1]);
        assert!((data[2] - 139.0).abs() < 1e-4, "c[1,0]={}", data[2]);
        assert!((data[3] - 154.0).abs() < 1e-4, "c[1,1]={}", data[3]);
    }

    #[test]
    fn test_gemm_incompatible_shapes() {
        let a = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();

        let result = gemm(&a, &b);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(
            msg.contains("mismatch"),
            "Expected 'mismatch' in error: {msg}"
        );
    }

    #[test]
    fn test_gemm_non_2d() {
        let a = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let b = Tensor::from_vec_f32(vec![4.0, 5.0, 6.0], vec![3]).unwrap();

        let result = gemm(&a, &b);
        assert!(result.is_err());
    }

    #[test]
    fn test_gemm_identity() {
        // A * I = A for 2x2 identity
        let a = Tensor::from_vec_f32(vec![3.0, 7.0, 5.0, 2.0], vec![2, 2]).unwrap();
        let identity = Tensor::from_vec_f32(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();

        let c = gemm(&a, &identity).unwrap();
        let data = c.as_slice_f32().unwrap();
        assert!((data[0] - 3.0).abs() < 1e-6);
        assert!((data[1] - 7.0).abs() < 1e-6);
        assert!((data[2] - 5.0).abs() < 1e-6);
        assert!((data[3] - 2.0).abs() < 1e-6);
    }

    // -- LayerNorm tests --

    #[test]
    fn test_layer_norm_mean_approx_zero() {
        let x = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0], vec![2, 4])
            .unwrap();

        let result = layer_norm(&x, 1e-5).unwrap();
        assert_eq!(result.shape(), &[2, 4]);

        let data = result.as_slice_f32().unwrap();

        // Check each row's mean ≈ 0
        let row1_mean: f32 = data[0..4].iter().sum::<f32>() / 4.0;
        let row2_mean: f32 = data[4..8].iter().sum::<f32>() / 4.0;

        assert!(
            row1_mean.abs() < 1e-5,
            "row1 mean should be ~0, got {row1_mean}"
        );
        assert!(
            row2_mean.abs() < 1e-5,
            "row2 mean should be ~0, got {row2_mean}"
        );
    }

    #[test]
    fn test_layer_norm_std_approx_one() {
        let x = Tensor::from_vec_f32(
            vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0],
            vec![2, 4],
        )
        .unwrap();

        let result = layer_norm(&x, 1e-5).unwrap();
        let data = result.as_slice_f32().unwrap();

        // Check each row's std ≈ 1
        for row_start in [0, 4] {
            let row = &data[row_start..row_start + 4];
            let mean: f32 = row.iter().sum::<f32>() / 4.0;
            let var: f32 = row.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / 4.0;
            let std_dev = var.sqrt();
            assert!(
                (std_dev - 1.0).abs() < 0.01,
                "row std should be ~1, got {std_dev}"
            );
        }
    }

    #[test]
    fn test_layer_norm_too_few_dims() {
        let x = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let result = layer_norm(&x, 1e-5);
        assert!(result.is_err());
    }

    // -- Softmax tests --

    #[test]
    fn test_softmax_sums_to_one() {
        let x = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]).unwrap();

        let result = softmax(&x).unwrap();
        let data = result.as_slice_f32().unwrap();

        let sum: f32 = data.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "softmax sum should be 1.0, got {sum}"
        );
    }

    #[test]
    fn test_softmax_monotonic() {
        let x = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]).unwrap();

        let result = softmax(&x).unwrap();
        let data = result.as_slice_f32().unwrap();

        // Larger input → larger softmax output
        for i in 0..3 {
            assert!(
                data[i] < data[i + 1],
                "softmax should be monotonic: data[{}]={} >= data[{}]={}",
                i,
                data[i],
                i + 1,
                data[i + 1]
            );
        }
    }

    #[test]
    fn test_softmax_multi_row() {
        let x = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0], vec![2, 3]).unwrap();

        let result = softmax(&x).unwrap();
        let data = result.as_slice_f32().unwrap();

        let sum1: f32 = data[0..3].iter().sum();
        let sum2: f32 = data[3..6].iter().sum();

        assert!((sum1 - 1.0).abs() < 1e-5, "row 1 sum = {sum1}");
        assert!((sum2 - 1.0).abs() < 1e-5, "row 2 sum = {sum2}");
    }

    #[test]
    fn test_softmax_numerical_stability() {
        // Large values should not cause overflow
        let x = Tensor::from_vec_f32(vec![1000.0, 1001.0, 1002.0], vec![1, 3]).unwrap();

        let result = softmax(&x).unwrap();
        let data = result.as_slice_f32().unwrap();

        let sum: f32 = data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum = {sum}");
        // No NaN or Inf
        for &v in data {
            assert!(v.is_finite(), "softmax output should be finite, got {v}");
        }
    }

    // -- SiLU tests --

    #[test]
    fn test_silu_zero() {
        let x = Tensor::from_vec_f32(vec![0.0], vec![1]).unwrap();
        let result = silu(&x).unwrap();
        let data = result.as_slice_f32().unwrap();
        assert!(
            data[0].abs() < 1e-7,
            "SiLU(0) should be 0, got {}",
            data[0]
        );
    }

    #[test]
    fn test_silu_one() {
        let x = Tensor::from_vec_f32(vec![1.0], vec![1]).unwrap();
        let result = silu(&x).unwrap();
        let data = result.as_slice_f32().unwrap();
        // SiLU(1) = 1 * sigmoid(1) = 1 / (1 + exp(-1)) ≈ 0.7311
        assert!(
            (data[0] - 0.7311).abs() < 0.001,
            "SiLU(1) should be ~0.7311, got {}",
            data[0]
        );
    }

    #[test]
    fn test_silu_negative() {
        let x = Tensor::from_vec_f32(vec![-5.0], vec![1]).unwrap();
        let result = silu(&x).unwrap();
        let data = result.as_slice_f32().unwrap();
        // SiLU(-5) = -5 * sigmoid(-5) ≈ -5 * 0.00669 ≈ -0.0335
        assert!(
            data[0] < 0.0,
            "SiLU(negative) should be negative, got {}",
            data[0]
        );
    }

    #[test]
    fn test_silu_shape_preserved() {
        let x = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let result = silu(&x).unwrap();
        assert_eq!(result.shape(), &[2, 3]);
    }

    // -- GELU tests --

    #[test]
    fn test_gelu_zero() {
        let x = Tensor::from_vec_f32(vec![0.0], vec![1]).unwrap();
        let result = gelu(&x).unwrap();
        let data = result.as_slice_f32().unwrap();
        assert!(
            data[0].abs() < 1e-7,
            "GELU(0) should be 0, got {}",
            data[0]
        );
    }

    #[test]
    fn test_gelu_one() {
        let x = Tensor::from_vec_f32(vec![1.0], vec![1]).unwrap();
        let result = gelu(&x).unwrap();
        let data = result.as_slice_f32().unwrap();
        // GELU(1) ≈ 0.8412
        assert!(
            (data[0] - 0.8412).abs() < 0.001,
            "GELU(1) should be ~0.8412, got {}",
            data[0]
        );
    }

    #[test]
    fn test_gelu_negative() {
        let x = Tensor::from_vec_f32(vec![-1.0], vec![1]).unwrap();
        let result = gelu(&x).unwrap();
        let data = result.as_slice_f32().unwrap();
        // GELU(-1) ≈ -0.1588
        assert!(
            (data[0] - (-0.1588)).abs() < 0.001,
            "GELU(-1) should be ~-0.1588, got {}",
            data[0]
        );
    }

    #[test]
    fn test_gelu_shape_preserved() {
        let x = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let result = gelu(&x).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
    }

    // -- GroupNorm tests --

    #[test]
    fn test_group_norm_shape_preserved() {
        // [batch=1, channels=4, spatial=3]
        let values: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let x = Tensor::from_vec_f32(values, vec![1, 4, 3]).unwrap();

        let result = group_norm(&x, 2, 1e-5).unwrap();
        assert_eq!(result.shape(), &[1, 4, 3]);
    }

    #[test]
    fn test_group_norm_normalized() {
        // [batch=1, channels=4, spatial=2], 2 groups → 2 channels/group
        let values = vec![
            1.0, 2.0, 3.0, 4.0, // channels 0,1 (group 0)
            5.0, 6.0, 7.0, 8.0, // channels 2,3 (group 1)
        ];
        let x = Tensor::from_vec_f32(values, vec![1, 4, 2]).unwrap();

        let result = group_norm(&x, 2, 1e-5).unwrap();
        let data = result.as_slice_f32().unwrap();

        // Group 0: channels 0,1 → elements [0..4], mean should be ~0
        let group0_mean: f32 = data[0..4].iter().sum::<f32>() / 4.0;
        assert!(
            group0_mean.abs() < 1e-4,
            "group 0 mean should be ~0, got {group0_mean}"
        );

        // Group 1: channels 2,3 → elements [4..8], mean should be ~0
        let group1_mean: f32 = data[4..8].iter().sum::<f32>() / 4.0;
        assert!(
            group1_mean.abs() < 1e-4,
            "group 1 mean should be ~0, got {group1_mean}"
        );
    }

    #[test]
    fn test_group_norm_channels_not_divisible() {
        let values: Vec<f32> = (0..9).map(|i| i as f32).collect();
        let x = Tensor::from_vec_f32(values, vec![1, 3, 3]).unwrap();

        let result = group_norm(&x, 2, 1e-5);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("divisible"), "Error: {msg}");
    }

    #[test]
    fn test_group_norm_too_few_dims() {
        let x = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let result = group_norm(&x, 1, 1e-5);
        assert!(result.is_err());
    }

    #[test]
    fn test_group_norm_batched() {
        // [batch=2, channels=2, spatial=2], 1 group
        let values = vec![
            1.0, 2.0, 3.0, 4.0, // batch 0
            10.0, 20.0, 30.0, 40.0, // batch 1
        ];
        let x = Tensor::from_vec_f32(values, vec![2, 2, 2]).unwrap();

        let result = group_norm(&x, 1, 1e-5).unwrap();
        assert_eq!(result.shape(), &[2, 2, 2]);

        let data = result.as_slice_f32().unwrap();

        // Each batch is independently normalized
        let batch0_mean: f32 = data[0..4].iter().sum::<f32>() / 4.0;
        let batch1_mean: f32 = data[4..8].iter().sum::<f32>() / 4.0;

        assert!(batch0_mean.abs() < 1e-4, "batch 0 mean = {batch0_mean}");
        assert!(batch1_mean.abs() < 1e-4, "batch 1 mean = {batch1_mean}");
    }

    // -- Sobel edge detection tests --

    #[test]
    fn test_sobel_edges_shape() {
        // 5x5 image with a vertical edge
        let mut values = vec![0.0f32; 25];
        for y in 0..5 {
            for x in 3..5 {
                values[y * 5 + x] = 1.0;
            }
        }
        let x = Tensor::from_vec_f32(values, vec![5, 5]).unwrap();
        let result = sobel_edges(&x).unwrap();
        assert_eq!(result.shape(), &[5, 5]);
    }

    #[test]
    fn test_sobel_edges_detects_edge() {
        // 5x5 image: left half = 0, right half = 1 (vertical edge at column 2-3)
        let mut values = vec![0.0f32; 25];
        for y in 0..5 {
            for x in 3..5 {
                values[y * 5 + x] = 1.0;
            }
        }
        let x = Tensor::from_vec_f32(values, vec![5, 5]).unwrap();
        let result = sobel_edges(&x).unwrap();
        let data = result.as_slice_f32().unwrap();

        // Interior pixels near the edge should have non-zero values
        // Position (2,2) is near the edge boundary
        let edge_val = data[2 * 5 + 2];
        assert!(edge_val > 0.0, "Edge pixel should have non-zero magnitude, got {edge_val}");

        // Position (2,0) is far from the edge, should be zero or near-zero
        let flat_val = data[2 * 5 + 0];
        assert!(flat_val.abs() < 1e-6, "Flat region pixel should be ~0, got {flat_val}");
    }

    #[test]
    fn test_sobel_edges_uniform_image() {
        // Uniform image should have all-zero edges
        let values = vec![0.5f32; 16];
        let x = Tensor::from_vec_f32(values, vec![4, 4]).unwrap();
        let result = sobel_edges(&x).unwrap();
        let data = result.as_slice_f32().unwrap();
        for &v in data {
            assert!(v.abs() < 1e-6, "Uniform image should have zero edges, got {v}");
        }
    }

    #[test]
    fn test_sobel_edges_wrong_dims() {
        let x = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let result = sobel_edges(&x);
        assert!(result.is_err());
    }

    #[test]
    fn test_sobel_edges_border_zero() {
        // Border pixels should remain zero (no computation at boundaries)
        let mut values = vec![0.0f32; 25];
        values[12] = 1.0; // center pixel bright
        let x = Tensor::from_vec_f32(values, vec![5, 5]).unwrap();
        let result = sobel_edges(&x).unwrap();
        let data = result.as_slice_f32().unwrap();

        // Top row should be zero
        for x_pos in 0..5 {
            assert!(data[x_pos].abs() < 1e-6, "Top border should be zero");
        }
        // Bottom row should be zero
        for x_pos in 0..5 {
            assert!(data[4 * 5 + x_pos].abs() < 1e-6, "Bottom border should be zero");
        }
        // Left column should be zero
        for y in 0..5 {
            assert!(data[y * 5].abs() < 1e-6, "Left border should be zero");
        }
        // Right column should be zero
        for y in 0..5 {
            assert!(data[y * 5 + 4].abs() < 1e-6, "Right border should be zero");
        }
    }

    // -- FP16 conversion tests --

    #[test]
    fn test_f32_to_f16_roundtrip() {
        let values = vec![1.0f32, 0.5, -0.5, 0.0, 2.0, -2.0];
        let x = Tensor::from_vec_f32(values.clone(), vec![6]).unwrap();

        let f16_tensor = f32_to_f16(&x).unwrap();
        assert_eq!(f16_tensor.dtype(), comfy_core::DType::F16);
        assert_eq!(f16_tensor.shape(), &[6]);

        let f32_tensor = f16_to_f32(&f16_tensor).unwrap();
        let data = f32_tensor.as_slice_f32().unwrap();
        for (i, &expected) in values.iter().enumerate() {
            assert!(
                (data[i] - expected).abs() < 1e-3,
                "roundtrip mismatch at {}: expected {}, got {}",
                i,
                expected,
                data[i]
            );
        }
    }

    #[test]
    fn test_f16_to_f32_wrong_dtype() {
        let x = Tensor::from_vec_f32(vec![1.0, 2.0], vec![2]).unwrap();
        let result = f16_to_f32(&x);
        assert!(result.is_err());
    }

    // -- Clamp tests --

    #[test]
    fn test_clamp_basic() {
        let x = Tensor::from_vec_f32(vec![-2.0, 0.5, 1.5, 3.0], vec![4]).unwrap();
        let result = clamp(&x, 0.0, 1.0).unwrap();
        let data = result.as_slice_f32().unwrap();
        assert!((data[0] - 0.0).abs() < 1e-6);
        assert!((data[1] - 0.5).abs() < 1e-6);
        assert!((data[2] - 1.0).abs() < 1e-6);
        assert!((data[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_clamp_all_in_range() {
        let x = Tensor::from_vec_f32(vec![0.2, 0.5, 0.8], vec![3]).unwrap();
        let result = clamp(&x, 0.0, 1.0).unwrap();
        let data = result.as_slice_f32().unwrap();
        assert!((data[0] - 0.2).abs() < 1e-6);
        assert!((data[1] - 0.5).abs() < 1e-6);
        assert!((data[2] - 0.8).abs() < 1e-6);
    }
}
