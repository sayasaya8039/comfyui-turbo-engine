use comfy_core::{ComfyResult, Node, NodeInputs, NodeMetadata, NodeOutputs, NodeValue};

/// CLIPTextEncode — encodes text using a CLIP model (stub).
///
/// In the real implementation this would tokenize the text and run
/// the CLIP text encoder. For now it returns a placeholder string.
pub struct CLIPTextEncode;

impl Node for CLIPTextEncode {
    fn execute(&self, inputs: &NodeInputs) -> ComfyResult<NodeOutputs> {
        let text = inputs.get_string("text")?;
        let clip = inputs.get_string("clip")?;
        tracing::info!(text, clip, "CLIPTextEncode: encoding text (stub)");

        let mut outputs = NodeOutputs::new();
        outputs.set(
            "output_0",
            NodeValue::String(format!("CONDITIONING:{text}")),
        );
        Ok(outputs)
    }

    fn metadata(&self) -> NodeMetadata {
        NodeMetadata {
            name: "CLIPTextEncode".to_string(),
            display_name: "CLIP Text Encode".to_string(),
            category: "conditioning".to_string(),
            description: "Encodes text using a CLIP model to produce conditioning".to_string(),
            output_node: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clip_text_encode() {
        let node = CLIPTextEncode;
        let mut inputs = NodeInputs::new();
        inputs.set("text", NodeValue::String("a photo of a cat".into()));
        inputs.set("clip", NodeValue::String("CLIP:v1-5".into()));

        let outputs = node.execute(&inputs).unwrap();
        match outputs.get("output_0") {
            Some(NodeValue::String(s)) => {
                assert!(s.contains("CONDITIONING:"));
                assert!(s.contains("a photo of a cat"));
            }
            other => panic!("Expected CONDITIONING string, got {other:?}"),
        }
    }

    #[test]
    fn test_clip_text_encode_metadata() {
        let meta = CLIPTextEncode.metadata();
        assert_eq!(meta.name, "CLIPTextEncode");
        assert_eq!(meta.category, "conditioning");
        assert!(!meta.output_node);
    }

    #[test]
    fn test_clip_text_encode_missing_text() {
        let node = CLIPTextEncode;
        let mut inputs = NodeInputs::new();
        inputs.set("clip", NodeValue::String("CLIP:v1-5".into()));
        assert!(node.execute(&inputs).is_err());
    }

    #[test]
    fn test_clip_text_encode_missing_clip() {
        let node = CLIPTextEncode;
        let mut inputs = NodeInputs::new();
        inputs.set("text", NodeValue::String("hello".into()));
        assert!(node.execute(&inputs).is_err());
    }
}
