use comfy_core::{
    ComfyResult, DType, Node, NodeInputs, NodeMetadata, NodeOutputs, NodeValue, Tensor,
};

/// EmptyLatentImage — creates a blank latent tensor.
///
/// Produces a zeros tensor of shape [batch_size, 4, height/8, width/8]
/// which is the standard latent space representation for Stable Diffusion.
pub struct EmptyLatentImage;

impl Node for EmptyLatentImage {
    fn execute(&self, inputs: &NodeInputs) -> ComfyResult<NodeOutputs> {
        let width = inputs.get_int("width")? as usize;
        let height = inputs.get_int("height")? as usize;
        let batch_size = inputs.get_int("batch_size")? as usize;

        tracing::info!(
            width,
            height,
            batch_size,
            "EmptyLatentImage: creating empty latent"
        );

        let latent = Tensor::zeros(
            vec![batch_size, 4, height / 8, width / 8],
            DType::F32,
        );

        let mut outputs = NodeOutputs::new();
        outputs.set("output_0", NodeValue::Tensor(latent));
        Ok(outputs)
    }

    fn metadata(&self) -> NodeMetadata {
        NodeMetadata {
            name: "EmptyLatentImage".to_string(),
            display_name: "Empty Latent Image".to_string(),
            category: "latent".to_string(),
            description: "Creates an empty latent image of the specified size".to_string(),
            output_node: false,
        }
    }
}

/// VAEDecode — decodes a latent tensor to an image (stub).
///
/// In the real implementation this would run the VAE decoder network.
/// For now it returns a zeros tensor scaled up from latent space.
pub struct VAEDecode;

impl Node for VAEDecode {
    fn execute(&self, inputs: &NodeInputs) -> ComfyResult<NodeOutputs> {
        let samples = inputs.get_tensor("samples")?;
        let _vae = inputs.get_string("vae")?;

        let shape = samples.shape();
        tracing::info!(?shape, "VAEDecode: decoding latent to image (stub)");

        // shape is [batch, 4, h, w] in latent space
        // output is [batch, 3, h*8, w*8] in pixel space
        let batch = shape[0];
        let h = shape[2];
        let w = shape[3];

        let image = Tensor::zeros(vec![batch, 3, h * 8, w * 8], DType::F32);

        let mut outputs = NodeOutputs::new();
        outputs.set("output_0", NodeValue::Tensor(image));
        Ok(outputs)
    }

    fn metadata(&self) -> NodeMetadata {
        NodeMetadata {
            name: "VAEDecode".to_string(),
            display_name: "VAE Decode".to_string(),
            category: "latent".to_string(),
            description: "Decodes latent samples into images using a VAE".to_string(),
            output_node: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_latent_image() {
        let node = EmptyLatentImage;
        let mut inputs = NodeInputs::new();
        inputs.set("width", NodeValue::Int(512));
        inputs.set("height", NodeValue::Int(512));
        inputs.set("batch_size", NodeValue::Int(1));

        let outputs = node.execute(&inputs).unwrap();
        match outputs.get("output_0") {
            Some(NodeValue::Tensor(t)) => {
                assert_eq!(t.shape(), &[1, 4, 64, 64]); // 512/8 = 64
                assert_eq!(t.dtype(), DType::F32);
                // All zeros
                let slice = t.as_slice_f32().unwrap();
                assert!(slice.iter().all(|&v| v == 0.0));
            }
            other => panic!("Expected Tensor, got {other:?}"),
        }
    }

    #[test]
    fn test_empty_latent_image_768() {
        let node = EmptyLatentImage;
        let mut inputs = NodeInputs::new();
        inputs.set("width", NodeValue::Int(768));
        inputs.set("height", NodeValue::Int(512));
        inputs.set("batch_size", NodeValue::Int(2));

        let outputs = node.execute(&inputs).unwrap();
        match outputs.get("output_0") {
            Some(NodeValue::Tensor(t)) => {
                assert_eq!(t.shape(), &[2, 4, 64, 96]); // h=512/8=64, w=768/8=96
            }
            other => panic!("Expected Tensor, got {other:?}"),
        }
    }

    #[test]
    fn test_empty_latent_image_metadata() {
        let meta = EmptyLatentImage.metadata();
        assert_eq!(meta.name, "EmptyLatentImage");
        assert_eq!(meta.category, "latent");
        assert!(!meta.output_node);
    }

    #[test]
    fn test_vae_decode() {
        let node = VAEDecode;
        let mut inputs = NodeInputs::new();
        let latent = Tensor::zeros(vec![1, 4, 64, 64], DType::F32);
        inputs.set("samples", NodeValue::Tensor(latent));
        inputs.set("vae", NodeValue::String("VAE:v1-5".into()));

        let outputs = node.execute(&inputs).unwrap();
        match outputs.get("output_0") {
            Some(NodeValue::Tensor(t)) => {
                assert_eq!(t.shape(), &[1, 3, 512, 512]); // 64*8 = 512
                assert_eq!(t.dtype(), DType::F32);
            }
            other => panic!("Expected Tensor, got {other:?}"),
        }
    }

    #[test]
    fn test_vae_decode_metadata() {
        let meta = VAEDecode.metadata();
        assert_eq!(meta.name, "VAEDecode");
        assert_eq!(meta.category, "latent");
        assert!(!meta.output_node);
    }

    #[test]
    fn test_vae_decode_missing_vae() {
        let node = VAEDecode;
        let mut inputs = NodeInputs::new();
        let latent = Tensor::zeros(vec![1, 4, 64, 64], DType::F32);
        inputs.set("samples", NodeValue::Tensor(latent));
        // Missing "vae" input
        assert!(node.execute(&inputs).is_err());
    }
}
