use comfy_core::{ComfyResult, Node, NodeInputs, NodeMetadata, NodeOutputs, NodeValue, Tensor};

/// KSampler — runs the sampling/denoising loop (stub).
///
/// In the real implementation this would perform iterative denoising
/// using a scheduler and UNet model. For now it returns a random
/// latent tensor seeded by the input seed.
pub struct KSampler;

impl Node for KSampler {
    fn execute(&self, inputs: &NodeInputs) -> ComfyResult<NodeOutputs> {
        let seed = inputs.get_int("seed")?;
        let steps = inputs.get_int("steps")?;
        let cfg = inputs.get_float("cfg")?;
        let sampler_name = inputs.get_string("sampler_name")?;

        // Read linked inputs (model, positive, negative, latent_image) as strings
        // In a real implementation these would be proper model/conditioning/latent types.
        let _model = inputs.get_string("model")?;
        let _positive = inputs.get_string("positive")?;
        let _negative = inputs.get_string("negative")?;
        let latent_image = inputs.get_tensor("latent_image")?;

        tracing::info!(
            seed,
            steps,
            cfg,
            sampler_name,
            latent_shape = ?latent_image.shape(),
            "KSampler: sampling (stub)"
        );

        // Produce a random latent with the same shape as the input
        let output_latent = Tensor::randn(latent_image.shape().to_vec(), seed as u64);

        let mut outputs = NodeOutputs::new();
        outputs.set("output_0", NodeValue::Tensor(output_latent));
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

    #[test]
    fn test_ksampler_execute() {
        let node = KSampler;
        let mut inputs = NodeInputs::new();
        inputs.set("seed", NodeValue::Int(42));
        inputs.set("steps", NodeValue::Int(20));
        inputs.set("cfg", NodeValue::Float(7.5));
        inputs.set("sampler_name", NodeValue::String("euler".into()));
        inputs.set("model", NodeValue::String("MODEL:v1-5".into()));
        inputs.set("positive", NodeValue::String("CONDITIONING:cat".into()));
        inputs.set("negative", NodeValue::String("CONDITIONING:bad".into()));

        let latent = Tensor::zeros(vec![1, 4, 64, 64], DType::F32);
        inputs.set("latent_image", NodeValue::Tensor(latent));

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
        let make_inputs = || {
            let mut inputs = NodeInputs::new();
            inputs.set("seed", NodeValue::Int(123));
            inputs.set("steps", NodeValue::Int(10));
            inputs.set("cfg", NodeValue::Float(7.0));
            inputs.set("sampler_name", NodeValue::String("euler_a".into()));
            inputs.set("model", NodeValue::String("MODEL:test".into()));
            inputs.set("positive", NodeValue::String("CONDITIONING:pos".into()));
            inputs.set("negative", NodeValue::String("CONDITIONING:neg".into()));
            let latent = Tensor::zeros(vec![1, 4, 8, 8], DType::F32);
            inputs.set("latent_image", NodeValue::Tensor(latent));
            inputs
        };

        let node = KSampler;
        let out1 = node.execute(&make_inputs()).unwrap();
        let out2 = node.execute(&make_inputs()).unwrap();

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
}
