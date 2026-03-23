use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::error::{ComfyError, ComfyResult};

/// Supported data types for tensor elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DType {
    F32,
    F16,
    BF16,
    I8,
    I32,
    U8,
}

impl DType {
    /// Returns the size in bytes of a single element of this dtype.
    pub fn byte_size(&self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 => 2,
            DType::BF16 => 2,
            DType::I8 => 1,
            DType::I32 => 4,
            DType::U8 => 1,
        }
    }
}

/// An immutable, reference-counted tensor.
#[derive(Debug, Clone)]
pub struct Tensor {
    data: Arc<Vec<u8>>,
    shape: Vec<usize>,
    dtype: DType,
}

impl Tensor {
    /// Create a tensor from a Vec of f32 values with the given shape.
    pub fn from_vec_f32(values: Vec<f32>, shape: Vec<usize>) -> ComfyResult<Self> {
        let numel: usize = shape.iter().product();
        if values.len() != numel {
            return Err(ComfyError::TensorError(format!(
                "shape {:?} requires {} elements, got {}",
                shape, numel, values.len()
            )));
        }
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        Ok(Self {
            data: Arc::new(bytes),
            shape,
            dtype: DType::F32,
        })
    }

    /// Create a tensor from raw bytes with the given shape and dtype.
    pub fn from_raw(data: Vec<u8>, shape: Vec<usize>, dtype: DType) -> ComfyResult<Self> {
        let numel: usize = shape.iter().product();
        let expected_bytes = numel * dtype.byte_size();
        if data.len() != expected_bytes {
            return Err(ComfyError::TensorError(format!(
                "shape {:?} with dtype {:?} requires {} bytes, got {}",
                shape, dtype, expected_bytes, data.len()
            )));
        }
        Ok(Self {
            data: Arc::new(data),
            shape,
            dtype,
        })
    }

    /// Create a tensor of zeros with the given shape and dtype.
    pub fn zeros(shape: Vec<usize>, dtype: DType) -> Self {
        let numel: usize = shape.iter().product();
        let byte_count = numel * dtype.byte_size();
        Self {
            data: Arc::new(vec![0u8; byte_count]),
            shape,
            dtype,
        }
    }

    /// Create a tensor with random normal values (mean=0, std=1) using
    /// xoshiro256++ PRNG and Box-Muller transform.
    pub fn randn(shape: Vec<usize>, seed: u64) -> Self {
        let numel: usize = shape.iter().product();
        let mut state = Xoshiro256PlusPlus::new(seed);
        let mut values = Vec::with_capacity(numel);

        // Box-Muller produces pairs of normal values
        let pairs = (numel + 1) / 2;
        for _ in 0..pairs {
            let u1 = state.next_f64().max(f64::MIN_POSITIVE); // avoid log(0)
            let u2 = state.next_f64();
            let mag = (-2.0 * u1.ln()).sqrt();
            let z0 = mag * (2.0 * std::f64::consts::PI * u2).cos();
            let z1 = mag * (2.0 * std::f64::consts::PI * u2).sin();
            values.push(z0 as f32);
            if values.len() < numel {
                values.push(z1 as f32);
            }
        }

        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        Self {
            data: Arc::new(bytes),
            shape,
            dtype: DType::F32,
        }
    }

    /// Returns the shape of the tensor.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns the data type of the tensor.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Returns the total number of elements.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Returns the total size in bytes.
    pub fn byte_size(&self) -> usize {
        self.data.len()
    }

    /// Returns the raw bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Element-wise add: self + other (rayon-parallel for large tensors).
    pub fn add(&self, other: &Tensor) -> ComfyResult<Self> {
        if self.shape() != other.shape() {
            return Err(ComfyError::TensorError(format!(
                "add shape mismatch: {:?} vs {:?}",
                self.shape(),
                other.shape()
            )));
        }
        let a = self.as_slice_f32()?;
        let b = other.as_slice_f32()?;
        let result: Vec<f32> = if a.len() > 4096 {
            use rayon::prelude::*;
            a.par_iter()
                .zip(b.par_iter())
                .map(|(&x, &y)| x + y)
                .collect()
        } else {
            a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect()
        };
        Tensor::from_vec_f32(result, self.shape().to_vec())
    }

    /// Element-wise multiply: self * other (rayon-parallel for large tensors).
    pub fn mul(&self, other: &Tensor) -> ComfyResult<Self> {
        if self.shape() != other.shape() {
            return Err(ComfyError::TensorError(format!(
                "mul shape mismatch: {:?} vs {:?}",
                self.shape(),
                other.shape()
            )));
        }
        let a = self.as_slice_f32()?;
        let b = other.as_slice_f32()?;
        let result: Vec<f32> = if a.len() > 4096 {
            use rayon::prelude::*;
            a.par_iter()
                .zip(b.par_iter())
                .map(|(&x, &y)| x * y)
                .collect()
        } else {
            a.iter().zip(b.iter()).map(|(&x, &y)| x * y).collect()
        };
        Tensor::from_vec_f32(result, self.shape().to_vec())
    }

    /// Scalar multiply: self * scalar (rayon-parallel for large tensors).
    pub fn scale(&self, scalar: f32) -> ComfyResult<Self> {
        let data = self.as_slice_f32()?;
        let result: Vec<f32> = if data.len() > 4096 {
            use rayon::prelude::*;
            data.par_iter().map(|&x| x * scalar).collect()
        } else {
            data.iter().map(|&x| x * scalar).collect()
        };
        Tensor::from_vec_f32(result, self.shape().to_vec())
    }

    /// Interpret the tensor data as a slice of f32 values.
    /// Returns an error if the dtype is not F32.
    pub fn as_slice_f32(&self) -> ComfyResult<&[f32]> {
        if self.dtype != DType::F32 {
            return Err(ComfyError::TensorError(format!(
                "cannot view {:?} tensor as f32 slice",
                self.dtype
            )));
        }
        let ptr = self.data.as_ptr() as *const f32;
        let len = self.data.len() / 4;
        // SAFETY: data was created from f32 values with correct alignment via Vec<u8>,
        // and Arc ensures the data outlives this borrow.
        Ok(unsafe { std::slice::from_raw_parts(ptr, len) })
    }
}

/// Xoshiro256++ PRNG for deterministic random number generation.
struct Xoshiro256PlusPlus {
    s: [u64; 4],
}

impl Xoshiro256PlusPlus {
    fn new(seed: u64) -> Self {
        // Use splitmix64 to initialize state from a single seed
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

    fn next_f64(&mut self) -> f64 {
        // Convert to f64 in [0, 1)
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation_f32() {
        let values = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_vec_f32(values.clone(), vec![2, 3]).unwrap();
        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.dtype(), DType::F32);
        assert_eq!(tensor.numel(), 6);

        let slice = tensor.as_slice_f32().unwrap();
        assert_eq!(slice, &values[..]);
    }

    #[test]
    fn test_tensor_creation_shape_mismatch() {
        let values = vec![1.0f32, 2.0, 3.0];
        let result = Tensor::from_vec_f32(values, vec![2, 3]);
        assert!(result.is_err());
    }

    #[test]
    fn test_tensor_zeros() {
        let tensor = Tensor::zeros(vec![4, 4], DType::F32);
        assert_eq!(tensor.shape(), &[4, 4]);
        assert_eq!(tensor.numel(), 16);
        assert_eq!(tensor.byte_size(), 64); // 16 * 4 bytes

        let slice = tensor.as_slice_f32().unwrap();
        for &v in slice {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn test_tensor_randn() {
        let tensor = Tensor::randn(vec![100], 42);
        assert_eq!(tensor.shape(), &[100]);
        assert_eq!(tensor.dtype(), DType::F32);
        assert_eq!(tensor.numel(), 100);

        let slice = tensor.as_slice_f32().unwrap();
        // At least some values should be non-zero
        let non_zero_count = slice.iter().filter(|&&v| v != 0.0).count();
        assert!(non_zero_count > 90, "Expected most values non-zero, got {non_zero_count}");

        // Values should be roughly normal: most within [-4, 4]
        let in_range = slice.iter().filter(|&&v| v.abs() < 4.0).count();
        assert!(in_range > 95, "Expected most values in [-4, 4], got {in_range}");
    }

    #[test]
    fn test_tensor_randn_deterministic() {
        let t1 = Tensor::randn(vec![10], 123);
        let t2 = Tensor::randn(vec![10], 123);
        assert_eq!(t1.as_bytes(), t2.as_bytes());
    }

    #[test]
    fn test_tensor_byte_size() {
        // F16 tensor: 2 bytes per element
        let data = vec![0u8; 2 * 3 * 4]; // 3x4 F16 tensor = 24 bytes
        let tensor = Tensor::from_raw(data, vec![3, 4], DType::F16).unwrap();
        assert_eq!(tensor.byte_size(), 24);
        assert_eq!(tensor.numel(), 12);
    }

    #[test]
    fn test_tensor_from_raw_size_mismatch() {
        let data = vec![0u8; 10]; // wrong size
        let result = Tensor::from_raw(data, vec![3, 4], DType::F16);
        assert!(result.is_err());
    }

    #[test]
    fn test_dtype_size() {
        assert_eq!(DType::F32.byte_size(), 4);
        assert_eq!(DType::F16.byte_size(), 2);
        assert_eq!(DType::BF16.byte_size(), 2);
        assert_eq!(DType::I8.byte_size(), 1);
        assert_eq!(DType::I32.byte_size(), 4);
        assert_eq!(DType::U8.byte_size(), 1);
    }

    #[test]
    fn test_tensor_as_slice_f32_wrong_dtype() {
        let tensor = Tensor::zeros(vec![4], DType::U8);
        let result = tensor.as_slice_f32();
        assert!(result.is_err());
    }

    #[test]
    fn test_tensor_clone_shares_data() {
        let tensor = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let cloned = tensor.clone();
        // Arc means same backing data
        assert_eq!(tensor.as_bytes().as_ptr(), cloned.as_bytes().as_ptr());
    }

    // -- add / mul / scale tests --

    #[test]
    fn test_tensor_add_small() {
        let a = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let b = Tensor::from_vec_f32(vec![4.0, 5.0, 6.0], vec![3]).unwrap();
        let c = a.add(&b).unwrap();
        assert_eq!(c.shape(), &[3]);
        let data = c.as_slice_f32().unwrap();
        assert_eq!(data, &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_tensor_add_large_parallel() {
        let n = 8192;
        let a = Tensor::from_vec_f32(vec![1.0; n], vec![n]).unwrap();
        let b = Tensor::from_vec_f32(vec![2.0; n], vec![n]).unwrap();
        let c = a.add(&b).unwrap();
        let data = c.as_slice_f32().unwrap();
        for &v in data {
            assert!((v - 3.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_tensor_add_shape_mismatch() {
        let a = Tensor::from_vec_f32(vec![1.0, 2.0], vec![2]).unwrap();
        let b = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        assert!(a.add(&b).is_err());
    }

    #[test]
    fn test_tensor_mul_small() {
        let a = Tensor::from_vec_f32(vec![2.0, 3.0, 4.0], vec![3]).unwrap();
        let b = Tensor::from_vec_f32(vec![5.0, 6.0, 7.0], vec![3]).unwrap();
        let c = a.mul(&b).unwrap();
        let data = c.as_slice_f32().unwrap();
        assert_eq!(data, &[10.0, 18.0, 28.0]);
    }

    #[test]
    fn test_tensor_mul_large_parallel() {
        let n = 8192;
        let a = Tensor::from_vec_f32(vec![3.0; n], vec![n]).unwrap();
        let b = Tensor::from_vec_f32(vec![4.0; n], vec![n]).unwrap();
        let c = a.mul(&b).unwrap();
        let data = c.as_slice_f32().unwrap();
        for &v in data {
            assert!((v - 12.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_tensor_mul_shape_mismatch() {
        let a = Tensor::from_vec_f32(vec![1.0, 2.0], vec![2]).unwrap();
        let b = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        assert!(a.mul(&b).is_err());
    }

    #[test]
    fn test_tensor_scale_small() {
        let a = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let c = a.scale(2.5).unwrap();
        let data = c.as_slice_f32().unwrap();
        assert!((data[0] - 2.5).abs() < 1e-6);
        assert!((data[1] - 5.0).abs() < 1e-6);
        assert!((data[2] - 7.5).abs() < 1e-6);
    }

    #[test]
    fn test_tensor_scale_large_parallel() {
        let n = 8192;
        let a = Tensor::from_vec_f32(vec![2.0; n], vec![n]).unwrap();
        let c = a.scale(3.0).unwrap();
        let data = c.as_slice_f32().unwrap();
        for &v in data {
            assert!((v - 6.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_tensor_scale_zero() {
        let a = Tensor::from_vec_f32(vec![5.0, 10.0], vec![2]).unwrap();
        let c = a.scale(0.0).unwrap();
        let data = c.as_slice_f32().unwrap();
        assert_eq!(data, &[0.0, 0.0]);
    }

    #[test]
    fn test_tensor_add_2d() {
        let a = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::from_vec_f32(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2]).unwrap();
        let c = a.add(&b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        let data = c.as_slice_f32().unwrap();
        assert_eq!(data, &[11.0, 22.0, 33.0, 44.0]);
    }
}
