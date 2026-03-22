use std::path::PathBuf;

/// ONNX Runtime graph optimization level.
///
/// Maps to `ort::session::builder::GraphOptimizationLevel` values.
/// Higher levels enable more aggressive transformations that may increase
/// session creation time but produce faster inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationLevel {
    /// No optimizations applied.
    Disabled = 0,
    /// Basic optimizations (constant folding, redundant node elimination).
    Basic = 1,
    /// Extended optimizations (node fusion, layout optimization).
    Extended = 2,
    /// All available optimizations including hardware-specific transforms.
    All = 99,
}

impl OptimizationLevel {
    /// Convert to the integer value expected by ONNX Runtime's C API.
    pub fn to_ort_level(self) -> i32 {
        self as i32
    }
}

/// Configuration for ONNX graph optimization and quantization passes.
///
/// Controls which optimizations the ONNX Runtime session builder applies
/// when loading a model. Can optionally save the optimized graph to disk
/// so subsequent loads skip the optimization step.
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    /// Graph optimization level.
    pub level: OptimizationLevel,
    /// Enable FP16 (half-precision) weight conversion.
    pub enable_fp16: bool,
    /// Enable INT8 quantization.
    pub enable_int8: bool,
    /// Enable constant folding during optimization.
    pub constant_folding: bool,
    /// Save the optimized model to disk after optimization.
    pub save_optimized_model: bool,
    /// File path for the saved optimized model.
    pub optimized_model_path: Option<PathBuf>,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            level: OptimizationLevel::All,
            enable_fp16: false,
            enable_int8: false,
            constant_folding: true,
            save_optimized_model: false,
            optimized_model_path: None,
        }
    }
}

impl OptimizationConfig {
    /// Aggressive configuration: All optimizations + FP16 + save optimized model.
    ///
    /// Best for GPU inference where FP16 is natively supported (CUDA, DirectML).
    pub fn aggressive() -> Self {
        Self {
            level: OptimizationLevel::All,
            enable_fp16: true,
            enable_int8: false,
            constant_folding: true,
            save_optimized_model: true,
            optimized_model_path: None,
        }
    }

    /// INT8 quantized configuration: All optimizations + INT8 + save optimized model.
    ///
    /// Best for CPU inference where INT8 VNNI/AVX-512 instructions are available.
    pub fn int8_quantized() -> Self {
        Self {
            level: OptimizationLevel::All,
            enable_fp16: false,
            enable_int8: true,
            constant_folding: true,
            save_optimized_model: true,
            optimized_model_path: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimization_level_values() {
        assert_eq!(OptimizationLevel::Disabled.to_ort_level(), 0);
        assert_eq!(OptimizationLevel::Basic.to_ort_level(), 1);
        assert_eq!(OptimizationLevel::Extended.to_ort_level(), 2);
        assert_eq!(OptimizationLevel::All.to_ort_level(), 99);
    }

    #[test]
    fn test_default_config() {
        let config = OptimizationConfig::default();
        assert_eq!(config.level, OptimizationLevel::All);
        assert!(!config.enable_fp16);
        assert!(!config.enable_int8);
        assert!(config.constant_folding);
        assert!(!config.save_optimized_model);
        assert!(config.optimized_model_path.is_none());
    }

    #[test]
    fn test_aggressive_config() {
        let config = OptimizationConfig::aggressive();
        assert_eq!(config.level, OptimizationLevel::All);
        assert!(config.enable_fp16);
        assert!(!config.enable_int8);
        assert!(config.constant_folding);
        assert!(config.save_optimized_model);
    }

    #[test]
    fn test_int8_quantized_config() {
        let config = OptimizationConfig::int8_quantized();
        assert_eq!(config.level, OptimizationLevel::All);
        assert!(!config.enable_fp16);
        assert!(config.enable_int8);
        assert!(config.constant_folding);
        assert!(config.save_optimized_model);
    }
}
