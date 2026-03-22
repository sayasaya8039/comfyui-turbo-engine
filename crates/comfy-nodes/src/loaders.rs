use comfy_core::{ComfyResult, Node, NodeInputs, NodeMetadata, NodeOutputs, NodeValue};

/// CheckpointLoaderSimple — loads a Stable Diffusion checkpoint (stub).
///
/// In the real implementation this would load model weights from disk.
/// For now it just logs the checkpoint name and returns placeholder strings.
pub struct CheckpointLoaderSimple;

impl Node for CheckpointLoaderSimple {
    fn execute(&self, inputs: &NodeInputs) -> ComfyResult<NodeOutputs> {
        let ckpt_name = inputs.get_string("ckpt_name")?;
        tracing::info!(ckpt_name, "CheckpointLoaderSimple: loading checkpoint (stub)");

        let mut outputs = NodeOutputs::new();
        outputs.set(
            "output_0",
            NodeValue::String(format!("MODEL:{ckpt_name}")),
        );
        outputs.set(
            "output_1",
            NodeValue::String(format!("CLIP:{ckpt_name}")),
        );
        outputs.set(
            "output_2",
            NodeValue::String(format!("VAE:{ckpt_name}")),
        );
        Ok(outputs)
    }

    fn metadata(&self) -> NodeMetadata {
        NodeMetadata {
            name: "CheckpointLoaderSimple".to_string(),
            display_name: "Load Checkpoint".to_string(),
            category: "loaders".to_string(),
            description: "Loads a Stable Diffusion checkpoint (model, CLIP, VAE)".to_string(),
            output_node: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkpoint_loader_simple() {
        let node = CheckpointLoaderSimple;
        let mut inputs = NodeInputs::new();
        inputs.set("ckpt_name", NodeValue::String("v1-5.safetensors".into()));

        let outputs = node.execute(&inputs).unwrap();

        match outputs.get("output_0") {
            Some(NodeValue::String(s)) => assert!(s.contains("MODEL:")),
            other => panic!("Expected MODEL string, got {other:?}"),
        }
        match outputs.get("output_1") {
            Some(NodeValue::String(s)) => assert!(s.contains("CLIP:")),
            other => panic!("Expected CLIP string, got {other:?}"),
        }
        match outputs.get("output_2") {
            Some(NodeValue::String(s)) => assert!(s.contains("VAE:")),
            other => panic!("Expected VAE string, got {other:?}"),
        }
    }

    #[test]
    fn test_checkpoint_loader_metadata() {
        let meta = CheckpointLoaderSimple.metadata();
        assert_eq!(meta.name, "CheckpointLoaderSimple");
        assert_eq!(meta.category, "loaders");
        assert!(!meta.output_node);
    }

    #[test]
    fn test_checkpoint_loader_missing_input() {
        let node = CheckpointLoaderSimple;
        let inputs = NodeInputs::new();
        assert!(node.execute(&inputs).is_err());
    }
}
