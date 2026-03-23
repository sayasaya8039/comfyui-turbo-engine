use comfy_core::{ComfyResult, Node, NodeInputs, NodeMetadata, NodeOutputs, NodeValue, Tensor};

/// KSampler — runs the sampling/denoising loop using comfy-julia samplers.
///
/// Parses sampler and scheduler names from inputs, generates a sigma schedule
/// via comfy-julia, then runs iterative sampling with a stub denoise function.
/// Real model inference will be wired in Phase 4.
pub struct KSampler;

impl Node for KSampler {
    fn execute(&self, inputs: &NodeInputs) -> ComfyResult<NodeOutputs> {
        let seed = inputs.get_int("seed")?;
        let steps = inputs.get_int("steps")?;
        let cfg = inputs.get_float("cfg")?;
        let sampler_name = inputs.get_string("sampler_name")?;
        let scheduler_name = inputs.get_string("scheduler").unwrap_or("normal");

        // Read linked inputs (model, positive, negative, latent_image)
        // In a real implementation these would be proper model/conditioning/latent types.
        let _model = inputs.get_string("model")?;
        let _positive = inputs.get_string("positive")?;
        let _negative = inputs.get_string("negative")?;
        let latent_image = inputs.get_tensor("latent_image")?;

        // Parse sampler and scheduler from comfy-julia
        let sampler = comfy_julia::SamplerType::from_name(sampler_name)
            .unwrap_or(comfy_julia::SamplerType::Euler);
        let scheduler = comfy_julia::SchedulerType::from_name(scheduler_name)
            .unwrap_or(comfy_julia::SchedulerType::Normal);

        let steps_usize = steps.max(1) as usize;

        tracing::info!(
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler = scheduler.name(),
            latent_shape = ?latent_image.shape(),
            "KSampler: sampling with {}/{}", sampler.name(), scheduler.name()
        );

        // Generate sigma schedule
        let sigmas = comfy_julia::get_sigmas(scheduler, steps_usize);

        // Use input latent shape, or generate random latent if needed
        let latent = Tensor::randn(latent_image.shape().to_vec(), seed as u64);

        // Stub denoise: real model inference comes in Phase 4
        let result = comfy_julia::sample(sampler, &latent, &sigmas, |x, _sigma| {
            let data = x.as_slice_f32()?;
            let scaled: Vec<f32> = data.iter().map(|v| v * 0.95).collect();
            Tensor::from_vec_f32_fast(scaled, x.shape().to_vec())
        })?;

        let mut outputs = NodeOutputs::new();
        outputs.set("output_0", NodeValue::Tensor(result));
        Ok(outputs)
    }

    fn metadata(&self) -> NodeMetadata {
        NodeMetadata {
            name: "KSampler".to_string(),
            display_name: "KSampler".to_string(),
            category: "sampling".to_string(),
            description: "Samples latent images using various samplers and schedulers".to_string(),
            output_node: false,
        }
    }

    fn device_hint(&self) -> comfy_core::Device {
        comfy_core::Device::Gpu(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use comfy_core::DType;

    /// Helper to build standard KSampler inputs with optional scheduler.
    fn make_inputs(seed: i64, steps: i64, sampler: &str, scheduler: Option<&str>) -> NodeInputs {
        let mut inputs = NodeInputs::new();
        inputs.set("seed", NodeValue::Int(seed));
        inputs.set("steps", NodeValue::Int(steps));
        inputs.set("cfg", NodeValue::Float(7.5));
        inputs.set("sampler_name", NodeValue::String(sampler.into()));
        if let Some(sched) = scheduler {
            inputs.set("scheduler", NodeValue::String(sched.into()));
        }
        inputs.set("model", NodeValue::String("MODEL:v1-5".into()));
        inputs.set("positive", NodeValue::String("CONDITIONING:cat".into()));
        inputs.set("negative", NodeValue::String("CONDITIONING:bad".into()));

        let latent = Tensor::zeros(vec![1, 4, 64, 64], DType::F32);
        inputs.set("latent_image", NodeValue::Tensor(latent));
        inputs
    }

    #[test]
    fn test_ksampler_execute() {
        let node = KSampler;
        let inputs = make_inputs(42, 20, "euler", None);

        let outputs = node.execute(&inputs).unwrap();
        match outputs.get("output_0") {
            Some(NodeValue::Tensor(t)) => {
                assert_eq!(t.shape(), &[1, 4, 64, 64]);
                assert_eq!(t.dtype(), DType::F32);
            }
            other => panic!("Expected Tensor output, got {other:?}"),
        }
    }

    #[test]
    fn test_ksampler_deterministic() {
        let node = KSampler;
        let out1 = node.execute(&make_inputs(123, 10, "euler_a", None)).unwrap();
        let out2 = node.execute(&make_inputs(123, 10, "euler_a", None)).unwrap();

        let t1 = match out1.get("output_0") {
            Some(NodeValue::Tensor(t)) => t,
            _ => panic!("Expected tensor"),
        };
        let t2 = match out2.get("output_0") {
            Some(NodeValue::Tensor(t)) => t,
            _ => panic!("Expected tensor"),
        };
        assert_eq!(t1.as_bytes(), t2.as_bytes());
    }

    #[test]
    fn test_ksampler_metadata() {
        let meta = KSampler.metadata();
        assert_eq!(meta.name, "KSampler");
        assert_eq!(meta.category, "sampling");
        assert!(!meta.output_node);
    }

    #[test]
    fn test_ksampler_device_hint_is_gpu() {
        use comfy_core::Node;
        assert_eq!(KSampler.device_hint(), comfy_core::Device::Gpu(0));
    }

    #[test]
    fn test_ksampler_with_karras_scheduler() {
        let node = KSampler;
        let inputs = make_inputs(99, 15, "euler", Some("karras"));

        let outputs = node.execute(&inputs).unwrap();
        match outputs.get("output_0") {
            Some(NodeValue::Tensor(t)) => {
                assert_eq!(t.shape(), &[1, 4, 64, 64]);
                assert_eq!(t.dtype(), DType::F32);
                // Verify tensor has been modified (not all zeros)
                let slice = t.as_slice_f32().unwrap();
                let non_zero = slice.iter().filter(|&&v| v != 0.0).count();
                assert!(non_zero > 0, "Expected non-zero values after sampling");
            }
            other => panic!("Expected Tensor output, got {other:?}"),
        }
    }

    #[test]
    fn test_ksampler_with_cosine_scheduler() {
        let node = KSampler;
        let inputs = make_inputs(77, 10, "ddim", Some("cosine"));

        let outputs = node.execute(&inputs).unwrap();
        match outputs.get("output_0") {
            Some(NodeValue::Tensor(t)) => {
                assert_eq!(t.shape(), &[1, 4, 64, 64]);
            }
            other => panic!("Expected Tensor output, got {other:?}"),
        }
    }

    #[test]
    fn test_ksampler_ddim_sampler() {
        let node = KSampler;
        let inputs = make_inputs(55, 20, "ddim", Some("normal"));

        let outputs = node.execute(&inputs).unwrap();
        match outputs.get("output_0") {
            Some(NodeValue::Tensor(t)) => {
                assert_eq!(t.shape(), &[1, 4, 64, 64]);
            }
            other => panic!("Expected Tensor output, got {other:?}"),
        }
    }

    #[test]
    fn test_ksampler_dpmpp_sampler() {
        let node = KSampler;
        let inputs = make_inputs(42, 10, "dpmpp_2m", Some("karras"));

        let outputs = node.execute(&inputs).unwrap();
        match outputs.get("output_0") {
            Some(NodeValue::Tensor(t)) => {
                assert_eq!(t.shape(), &[1, 4, 64, 64]);
            }
            other => panic!("Expected Tensor output, got {other:?}"),
        }
    }

    #[test]
    fn test_ksampler_unknown_sampler_falls_back_to_euler() {
        let node = KSampler;
        let inputs = make_inputs(42, 5, "unknown_sampler_xyz", None);

        // Should not error — falls back to Euler
        let outputs = node.execute(&inputs).unwrap();
        assert!(outputs.get("output_0").is_some());
    }

    #[test]
    fn test_ksampler_unknown_scheduler_falls_back_to_normal() {
        let node = KSampler;
        let inputs = make_inputs(42, 5, "euler", Some("unknown_scheduler_xyz"));

        // Should not error — falls back to Normal
        let outputs = node.execute(&inputs).unwrap();
        assert!(outputs.get("output_0").is_some());
    }
}
