//! Fused kernel operations — combine multiple operations into single passes
//! to eliminate intermediate tensor allocations and improve cache utilisation.
//!
//! Each fused kernel produces results numerically equivalent to running the
//! individual operations in sequence, but with fewer memory round-trips.

use comfy_core::{ComfyError, ComfyResult, Tensor};

// ---------------------------------------------------------------------------
// Fused LayerNorm + SiLU
// ---------------------------------------------------------------------------

/// Layer-normalise `x` then apply SiLU in a single pass (no intermediate tensor).
///
/// Equivalent to `silu(layer_norm(x, eps))` but avoids allocating the
/// normalised intermediate.
///
/// Input must be at least 2-D F32.
pub fn fused_layer_norm_silu(x: &Tensor, eps: f32) -> ComfyResult<Tensor> {
    let shape = x.shape();
    if shape.len() < 2 {
        return Err(ComfyError::TensorError(format!(
            "fused_layer_norm_silu requires at least 2-D tensor, got shape {:?}",
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

        // --- Layer Norm ---
        let mean = row.iter().sum::<f32>() / last_dim as f32;
        let var = row
            .iter()
            .map(|&v| (v - mean) * (v - mean))
            .sum::<f32>()
            / last_dim as f32;
        let inv_std = 1.0 / (var + eps).sqrt();

        // --- Fused: normalise + SiLU in one loop ---
        for &v in row {
            let normed = (v - mean) * inv_std;
            let sigmoid = 1.0 / (1.0 + (-normed).exp());
            result.push(normed * sigmoid);
        }
    }

    Tensor::from_vec_f32(result, shape.to_vec())
}

// ---------------------------------------------------------------------------
// Fused Softmax + Scale
// ---------------------------------------------------------------------------

/// Numerically-stable softmax followed by element-wise scaling, fused into
/// one pass per row.
///
/// Equivalent to multiplying every element of `softmax(x)` by `scale`.
///
/// Input must be at least 1-D F32.
pub fn fused_softmax_scale(x: &Tensor, scale: f32) -> ComfyResult<Tensor> {
    let shape = x.shape();
    if shape.is_empty() {
        return Err(ComfyError::TensorError(
            "fused_softmax_scale requires non-empty shape".to_string(),
        ));
    }

    let data = x.as_slice_f32()?;
    let last_dim = *shape.last().unwrap();
    let num_rows = data.len() / last_dim;
    let mut result = Vec::with_capacity(data.len());

    for row_idx in 0..num_rows {
        let start = row_idx * last_dim;
        let row = &data[start..start + last_dim];

        let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = row.iter().map(|&v| (v - max_val).exp()).collect();
        let sum: f32 = exps.iter().sum();

        // Fused: divide + scale in one loop
        for e in &exps {
            result.push((e / sum) * scale);
        }
    }

    Tensor::from_vec_f32(result, shape.to_vec())
}

// ---------------------------------------------------------------------------
// Fused GEMM + SiLU
// ---------------------------------------------------------------------------

/// Matrix multiply `a * b` then apply SiLU element-wise, without allocating
/// the intermediate GEMM result.
///
/// Both inputs must be 2-D F32 with compatible inner dimensions.
pub fn fused_gemm_silu(a: &Tensor, b: &Tensor) -> ComfyResult<Tensor> {
    let a_shape = a.shape();
    let b_shape = b.shape();

    if a_shape.len() != 2 || b_shape.len() != 2 {
        return Err(ComfyError::TensorError(format!(
            "fused_gemm_silu requires 2-D tensors, got shapes {:?} and {:?}",
            a_shape, b_shape
        )));
    }

    let m = a_shape[0];
    let k = a_shape[1];
    let k2 = b_shape[0];
    let n = b_shape[1];

    if k != k2 {
        return Err(ComfyError::TensorError(format!(
            "fused_gemm_silu inner dimensions mismatch: a=[{},{}], b=[{},{}]",
            m, k, k2, n
        )));
    }

    let a_data = a.as_slice_f32()?;
    let b_data = b.as_slice_f32()?;
    let mut c = vec![0.0f32; m * n];

    // --- GEMM (tiled) ---
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

    // --- Fused SiLU ---
    for v in &mut c {
        let sigmoid = 1.0 / (1.0 + (-*v).exp());
        *v *= sigmoid;
    }

    Tensor::from_vec_f32(c, vec![m, n])
}

// ---------------------------------------------------------------------------
// Fused GEMM + GELU
// ---------------------------------------------------------------------------

/// Matrix multiply `a * b` then apply approximate GELU element-wise,
/// without allocating the intermediate GEMM result.
///
/// Both inputs must be 2-D F32 with compatible inner dimensions.
pub fn fused_gemm_gelu(a: &Tensor, b: &Tensor) -> ComfyResult<Tensor> {
    let a_shape = a.shape();
    let b_shape = b.shape();

    if a_shape.len() != 2 || b_shape.len() != 2 {
        return Err(ComfyError::TensorError(format!(
            "fused_gemm_gelu requires 2-D tensors, got shapes {:?} and {:?}",
            a_shape, b_shape
        )));
    }

    let m = a_shape[0];
    let k = a_shape[1];
    let k2 = b_shape[0];
    let n = b_shape[1];

    if k != k2 {
        return Err(ComfyError::TensorError(format!(
            "fused_gemm_gelu inner dimensions mismatch: a=[{},{}], b=[{},{}]",
            m, k, k2, n
        )));
    }

    let a_data = a.as_slice_f32()?;
    let b_data = b.as_slice_f32()?;
    let mut c = vec![0.0f32; m * n];

    // --- GEMM (tiled) ---
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

    // --- Fused GELU ---
    let sqrt_2_over_pi = (2.0f32 / std::f32::consts::PI).sqrt();
    for v in &mut c {
        let inner = sqrt_2_over_pi * (*v + 0.044715 * *v * *v * *v);
        *v = 0.5 * *v * (1.0 + inner.tanh());
    }

    Tensor::from_vec_f32(c, vec![m, n])
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels;

    #[test]
    fn test_fused_layer_norm_silu_shape_preserved() {
        let x = Tensor::from_vec_f32(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
        )
        .unwrap();
        let result = fused_layer_norm_silu(&x, 1e-5).unwrap();
        assert_eq!(result.shape(), &[2, 3]);
    }

    #[test]
    fn test_fused_layer_norm_silu_matches_separate() {
        let x = Tensor::from_vec_f32(
            vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
            vec![2, 4],
        )
        .unwrap();

        // Separate operations
        let normed = kernels::layer_norm(&x, 1e-5).unwrap();
        let expected = kernels::silu(&normed).unwrap();

        // Fused
        let fused = fused_layer_norm_silu(&x, 1e-5).unwrap();

        let exp_data = expected.as_slice_f32().unwrap();
        let fused_data = fused.as_slice_f32().unwrap();

        assert_eq!(exp_data.len(), fused_data.len());
        for (i, (&e, &f)) in exp_data.iter().zip(fused_data.iter()).enumerate() {
            assert!(
                (e - f).abs() < 1e-5,
                "mismatch at index {i}: separate={e}, fused={f}"
            );
        }
    }

    #[test]
    fn test_fused_softmax_scale_sum() {
        let x = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]).unwrap();
        let scale = 2.5f32;
        let result = fused_softmax_scale(&x, scale).unwrap();
        let data = result.as_slice_f32().unwrap();

        let sum: f32 = data.iter().sum();
        assert!(
            (sum - scale).abs() < 1e-5,
            "softmax*scale row sum should equal scale ({scale}), got {sum}"
        );
    }

    #[test]
    fn test_fused_gemm_silu_shape_and_positive() {
        // A = I (2x2), B = [[1, 2], [3, 4]]
        // GEMM result = B, then SiLU(B)
        let a = Tensor::from_vec_f32(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let b = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();

        let result = fused_gemm_silu(&a, &b).unwrap();
        assert_eq!(result.shape(), &[2, 2]);

        // SiLU of positive values is positive
        let data = result.as_slice_f32().unwrap();
        for (i, &v) in data.iter().enumerate() {
            assert!(v > 0.0, "SiLU of positive input should be positive at [{i}], got {v}");
        }

        // Verify against separate: gemm then silu
        let gemm_result = kernels::gemm(&a, &b).unwrap();
        let expected = kernels::silu(&gemm_result).unwrap();
        let exp_data = expected.as_slice_f32().unwrap();
        for (i, (&e, &f)) in exp_data.iter().zip(data.iter()).enumerate() {
            assert!(
                (e - f).abs() < 1e-5,
                "mismatch at [{i}]: separate={e}, fused={f}"
            );
        }
    }

    #[test]
    fn test_fused_gemm_gelu_shape() {
        // [2,3] x [3,4] -> [2,4]
        let a = Tensor::from_vec_f32(vec![1.0; 6], vec![2, 3]).unwrap();
        let b = Tensor::from_vec_f32(vec![1.0; 12], vec![3, 4]).unwrap();

        let result = fused_gemm_gelu(&a, &b).unwrap();
        assert_eq!(result.shape(), &[2, 4]);

        // Verify against separate: gemm then gelu
        let gemm_result = kernels::gemm(&a, &b).unwrap();
        let expected = kernels::gelu(&gemm_result).unwrap();
        let exp_data = expected.as_slice_f32().unwrap();
        let fused_data = result.as_slice_f32().unwrap();
        for (i, (&e, &f)) in exp_data.iter().zip(fused_data.iter()).enumerate() {
            assert!(
                (e - f).abs() < 1e-5,
                "mismatch at [{i}]: separate={e}, fused={f}"
            );
        }
    }
}
