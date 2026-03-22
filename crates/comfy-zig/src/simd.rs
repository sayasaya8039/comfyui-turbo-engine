use comfy_core::{ComfyError, ComfyResult, DType, Tensor};

/// Generate deterministic noise as an F32 tensor.
///
/// Uses xoshiro256++ seeded from the given `seed` value, producing
/// uniformly distributed values in approximately [-1.0, 1.0].
/// This is a pure-Rust fallback; future zig-native builds will use SIMD.
pub fn generate_noise_f32(shape: &[usize], seed: u64) -> Tensor {
    let numel: usize = shape.iter().product();
    let mut state = Xoshiro256PlusPlus::new(seed);

    let values: Vec<f32> = (0..numel)
        .map(|_| {
            // Map u64 → f32 in [-1.0, 1.0)
            let bits = state.next_u64();
            (bits >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
        })
        .collect();

    // from_vec_f32 cannot fail here because numel matches shape
    Tensor::from_vec_f32(values, shape.to_vec()).expect("shape matches element count")
}

/// Fused multiply-add: `a * scale + b`, element-wise on F32 tensors.
///
/// Both `a` and `b` must be F32 with the same shape.
/// Returns a new tensor with the result.
pub fn fused_multiply_add(a: &Tensor, b: &Tensor, scale: f32) -> ComfyResult<Tensor> {
    if a.dtype() != DType::F32 {
        return Err(ComfyError::TensorError(format!(
            "fused_multiply_add requires F32 tensors, a is {:?}",
            a.dtype()
        )));
    }
    if b.dtype() != DType::F32 {
        return Err(ComfyError::TensorError(format!(
            "fused_multiply_add requires F32 tensors, b is {:?}",
            b.dtype()
        )));
    }
    if a.shape() != b.shape() {
        return Err(ComfyError::TensorError(format!(
            "shape mismatch: a={:?}, b={:?}",
            a.shape(),
            b.shape()
        )));
    }

    let a_slice = a.as_slice_f32()?;
    let b_slice = b.as_slice_f32()?;

    let result: Vec<f32> = a_slice
        .iter()
        .zip(b_slice.iter())
        .map(|(&av, &bv)| av.mul_add(scale, bv))
        .collect();

    Tensor::from_vec_f32(result, a.shape().to_vec())
}

/// Minimal xoshiro256++ PRNG for deterministic noise generation.
struct Xoshiro256PlusPlus {
    s: [u64; 4],
}

impl Xoshiro256PlusPlus {
    fn new(seed: u64) -> Self {
        // splitmix64 to expand single seed into 4-word state
        let mut sm = seed;
        let mut next_sm = || -> u64 {
            sm = sm.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = sm;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            z ^ (z >> 31)
        };
        Self {
            s: [next_sm(), next_sm(), next_sm(), next_sm()],
        }
    }

    fn next_u64(&mut self) -> u64 {
        let result = (self.s[0].wrapping_add(self.s[3]))
            .rotate_left(23)
            .wrapping_add(self.s[0]);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_noise_deterministic() {
        let t1 = generate_noise_f32(&[4, 4], 42);
        let t2 = generate_noise_f32(&[4, 4], 42);
        assert_eq!(t1.as_bytes(), t2.as_bytes(), "same seed must produce same noise");

        // Different seed should produce different output
        let t3 = generate_noise_f32(&[4, 4], 99);
        assert_ne!(t1.as_bytes(), t3.as_bytes(), "different seeds should differ");
    }

    #[test]
    fn test_fused_multiply_add() {
        let a = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let b = Tensor::from_vec_f32(vec![10.0, 20.0, 30.0], vec![3]).unwrap();

        let result = fused_multiply_add(&a, &b, 2.0).unwrap();
        let slice = result.as_slice_f32().unwrap();

        // a * scale + b = [1*2+10, 2*2+20, 3*2+30] = [12, 24, 36]
        assert!((slice[0] - 12.0).abs() < 1e-6, "got {}", slice[0]);
        assert!((slice[1] - 24.0).abs() < 1e-6, "got {}", slice[1]);
        assert!((slice[2] - 36.0).abs() < 1e-6, "got {}", slice[2]);
    }

    #[test]
    fn test_fused_multiply_add_shape_mismatch() {
        let a = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let b = Tensor::from_vec_f32(vec![10.0, 20.0], vec![2]).unwrap();

        let result = fused_multiply_add(&a, &b, 1.0);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("shape mismatch"), "Error: {msg}");
    }
}
