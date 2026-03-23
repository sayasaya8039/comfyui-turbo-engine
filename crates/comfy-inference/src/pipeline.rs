//! Multi-device inference pipeline — routes model stages to optimal hardware.
//!
//! Enables GPU+NPU parallel execution by maintaining separate session caches
//! per execution provider, allowing CLIP to run on NPU while UNet runs on GPU.

use std::collections::HashMap;
use std::sync::Arc;

use comfy_core::{ComfyError, ComfyResult, Device, TaskType};
use ort::session::Session;

use crate::provider::ExecutionProvider;
use crate::session::SessionCache;

/// Model stage in the Stable Diffusion pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelStage {
    /// Text encoder (CLIP) — small model, ideal for NPU
    TextEncoder,
    /// UNet denoiser — large model, requires GPU
    UNet,
    /// VAE decoder — medium model, NPU or GPU
    VaeDecoder,
    /// VAE encoder — medium model, NPU or GPU
    VaeEncoder,
    /// Safety checker — small model, CPU or NPU
    SafetyChecker,
}

impl ModelStage {
    /// Returns the recommended TaskType for device routing.
    pub fn task_type(&self) -> TaskType {
        match self {
            ModelStage::TextEncoder => TaskType::TextEncoding,
            ModelStage::UNet => TaskType::ModelInference,
            ModelStage::VaeDecoder => TaskType::SmallModel,
            ModelStage::VaeEncoder => TaskType::SmallModel,
            ModelStage::SafetyChecker => TaskType::SmallModel,
        }
    }

    /// Returns the recommended execution provider based on available hardware.
    pub fn recommended_provider(&self, has_cuda: bool, has_npu: bool) -> ExecutionProvider {
        match self {
            ModelStage::TextEncoder => {
                if has_npu {
                    ExecutionProvider::OpenVino
                } else if has_cuda {
                    ExecutionProvider::Cuda
                } else {
                    ExecutionProvider::Cpu
                }
            }
            ModelStage::UNet => {
                if has_cuda {
                    ExecutionProvider::Cuda
                } else {
                    ExecutionProvider::DirectMl
                }
            }
            ModelStage::VaeDecoder | ModelStage::VaeEncoder => {
                if has_npu {
                    ExecutionProvider::OpenVino
                } else if has_cuda {
                    ExecutionProvider::Cuda
                } else {
                    ExecutionProvider::Cpu
                }
            }
            ModelStage::SafetyChecker => {
                if has_npu {
                    ExecutionProvider::OpenVino
                } else {
                    ExecutionProvider::Cpu
                }
            }
        }
    }
}

/// Multi-device inference pipeline with per-stage session caches.
///
/// Each model stage (CLIP, UNet, VAE) gets its own `SessionCache` configured
/// with the optimal execution provider for that workload, enabling true
/// GPU+NPU parallel execution.
pub struct InferencePipeline {
    /// Per-stage session caches with different execution providers
    stage_caches: HashMap<ModelStage, SessionCache>,
    /// Whether CUDA GPU is available
    has_cuda: bool,
    /// Whether NPU (OpenVINO) is available
    has_npu: bool,
}

impl InferencePipeline {
    /// Create a new pipeline with auto-detected hardware.
    pub fn new() -> Self {
        let has_cuda = ExecutionProvider::auto_detect() == ExecutionProvider::Cuda;
        let has_npu = std::env::var("INTEL_OPENVINO_DIR").is_ok();
        Self::with_hardware(has_cuda, has_npu)
    }

    /// Create a pipeline with explicit hardware flags.
    pub fn with_hardware(has_cuda: bool, has_npu: bool) -> Self {
        let mut stage_caches = HashMap::new();

        for stage in [
            ModelStage::TextEncoder,
            ModelStage::UNet,
            ModelStage::VaeDecoder,
            ModelStage::VaeEncoder,
            ModelStage::SafetyChecker,
        ] {
            let provider = stage.recommended_provider(has_cuda, has_npu);
            stage_caches.insert(stage, SessionCache::new(provider));
        }

        tracing::info!(
            has_cuda,
            has_npu,
            "InferencePipeline: initialized with per-stage providers"
        );

        Self {
            stage_caches,
            has_cuda,
            has_npu,
        }
    }

    /// Get or load an ONNX session for a specific pipeline stage.
    pub fn get_session(
        &self,
        stage: ModelStage,
        model_path: &str,
    ) -> ComfyResult<Arc<Session>> {
        let cache = self.stage_caches.get(&stage).ok_or_else(|| {
            ComfyError::InferenceError(format!("no cache configured for stage {stage:?}"))
        })?;
        cache.get_or_load(model_path)
    }

    /// Returns the execution provider assigned to a stage.
    pub fn provider_for(&self, stage: ModelStage) -> ExecutionProvider {
        self.stage_caches
            .get(&stage)
            .map(|c| c.provider())
            .unwrap_or(ExecutionProvider::Cpu)
    }

    /// Returns the target device for a stage.
    pub fn device_for(&self, stage: ModelStage) -> Device {
        match self.provider_for(stage) {
            ExecutionProvider::Cuda | ExecutionProvider::TensorRt | ExecutionProvider::DirectMl => {
                Device::Gpu(0)
            }
            ExecutionProvider::OpenVino => {
                if self.has_npu {
                    Device::Npu(0)
                } else {
                    Device::Cpu
                }
            }
            ExecutionProvider::Cpu => Device::Cpu,
        }
    }

    /// Check if GPU and NPU can run in parallel (both available).
    pub fn supports_parallel_devices(&self) -> bool {
        self.has_cuda && self.has_npu
    }

    /// Clear all cached sessions.
    pub fn clear_all(&self) {
        for cache in self.stage_caches.values() {
            cache.clear();
        }
    }

    /// Get pipeline status summary.
    pub fn status(&self) -> PipelineStatus {
        PipelineStatus {
            has_cuda: self.has_cuda,
            has_npu: self.has_npu,
            parallel_capable: self.supports_parallel_devices(),
            stages: self
                .stage_caches
                .iter()
                .map(|(stage, cache)| {
                    (*stage, StageStatus {
                        provider: cache.provider(),
                        cached_models: cache.len(),
                    })
                })
                .collect(),
        }
    }
}

/// Status of the inference pipeline.
#[derive(Debug)]
pub struct PipelineStatus {
    pub has_cuda: bool,
    pub has_npu: bool,
    pub parallel_capable: bool,
    pub stages: HashMap<ModelStage, StageStatus>,
}

/// Status of a single pipeline stage.
#[derive(Debug)]
pub struct StageStatus {
    pub provider: ExecutionProvider,
    pub cached_models: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_creation_cpu_only() {
        let pipeline = InferencePipeline::with_hardware(false, false);
        assert!(!pipeline.supports_parallel_devices());
        assert_eq!(
            pipeline.provider_for(ModelStage::TextEncoder),
            ExecutionProvider::Cpu
        );
        assert_eq!(
            pipeline.provider_for(ModelStage::UNet),
            ExecutionProvider::DirectMl // Windows fallback
        );
    }

    #[test]
    fn test_pipeline_creation_cuda_only() {
        let pipeline = InferencePipeline::with_hardware(true, false);
        assert!(!pipeline.supports_parallel_devices());
        assert_eq!(
            pipeline.provider_for(ModelStage::TextEncoder),
            ExecutionProvider::Cuda
        );
        assert_eq!(
            pipeline.provider_for(ModelStage::UNet),
            ExecutionProvider::Cuda
        );
    }

    #[test]
    fn test_pipeline_creation_cuda_and_npu() {
        let pipeline = InferencePipeline::with_hardware(true, true);
        assert!(pipeline.supports_parallel_devices());
        // CLIP goes to NPU, UNet to GPU
        assert_eq!(
            pipeline.provider_for(ModelStage::TextEncoder),
            ExecutionProvider::OpenVino
        );
        assert_eq!(
            pipeline.provider_for(ModelStage::UNet),
            ExecutionProvider::Cuda
        );
        assert_eq!(
            pipeline.provider_for(ModelStage::VaeDecoder),
            ExecutionProvider::OpenVino
        );
    }

    #[test]
    fn test_model_stage_task_types() {
        assert_eq!(ModelStage::TextEncoder.task_type(), TaskType::TextEncoding);
        assert_eq!(ModelStage::UNet.task_type(), TaskType::ModelInference);
        assert_eq!(ModelStage::VaeDecoder.task_type(), TaskType::SmallModel);
    }

    #[test]
    fn test_device_routing() {
        let pipeline = InferencePipeline::with_hardware(true, true);
        assert_eq!(pipeline.device_for(ModelStage::UNet), Device::Gpu(0));
        assert_eq!(pipeline.device_for(ModelStage::TextEncoder), Device::Npu(0));
    }

    #[test]
    fn test_pipeline_status() {
        let pipeline = InferencePipeline::with_hardware(true, false);
        let status = pipeline.status();
        assert!(status.has_cuda);
        assert!(!status.has_npu);
        assert!(!status.parallel_capable);
        assert!(status.stages.contains_key(&ModelStage::UNet));
    }

    #[test]
    fn test_pipeline_clear() {
        let pipeline = InferencePipeline::with_hardware(false, false);
        pipeline.clear_all(); // Should not panic
    }

    #[test]
    fn test_nonexistent_model_returns_error() {
        let pipeline = InferencePipeline::with_hardware(false, false);
        let result = pipeline.get_session(ModelStage::UNet, "nonexistent.onnx");
        assert!(result.is_err());
    }
}
