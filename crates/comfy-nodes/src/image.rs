use comfy_core::{ComfyResult, DType, Node, NodeInputs, NodeMetadata, NodeOutputs, NodeValue};

/// SaveImage — saves generated images to disk.
///
/// When the input tensor is a U8 [H, W, 3] tensor, this uses `comfy_zig::encode_png()`
/// to actually write the PNG file. For batch tensors (shape [B, H, W, 3] etc.)
/// it falls back to producing a stub UI output.
pub struct SaveImage;

impl Node for SaveImage {
    fn execute(&self, inputs: &NodeInputs) -> ComfyResult<NodeOutputs> {
        let filename_prefix = inputs.get_string("filename_prefix")?;
        let images = inputs.get_tensor("images")?;

        let shape = images.shape();
        let batch = shape[0];
        tracing::info!(
            filename_prefix,
            batch,
            shape = ?shape,
            "SaveImage: saving images"
        );

        let mut filenames = Vec::new();

        // If the tensor is U8 [H, W, 3] (single image, no batch dim), encode directly
        if images.dtype() == DType::U8 && shape.len() == 3 && shape[2] == 3 {
            let filename = format!("{filename_prefix}_00000.png");
            comfy_zig::encode_png(images, &filename)?;
            tracing::info!(filename = %filename, "SaveImage: wrote PNG via comfy-zig");
            filenames.push(filename);
        } else {
            // Batch or non-U8 tensor: generate stub filenames
            for i in 0..batch {
                filenames.push(format!("{filename_prefix}_{i:05}.png"));
            }
        }

        let ui_json = serde_json::json!({
            "images": filenames.iter().map(|f| {
                serde_json::json!({
                    "filename": f,
                    "subfolder": "",
                    "type": "output"
                })
            }).collect::<Vec<_>>()
        });

        let mut outputs = NodeOutputs::new();
        outputs.set("ui", NodeValue::String(ui_json.to_string()));
        Ok(outputs)
    }

    fn metadata(&self) -> NodeMetadata {
        NodeMetadata {
            name: "SaveImage".to_string(),
            display_name: "Save Image".to_string(),
            category: "image".to_string(),
            description: "Saves images to the output directory".to_string(),
            output_node: true,
        }
    }
}

/// PreviewImage — previews generated images in the UI (stub).
///
/// Similar to SaveImage but saves to a temporary directory for
/// preview purposes. This is an output node.
pub struct PreviewImage;

impl Node for PreviewImage {
    fn execute(&self, inputs: &NodeInputs) -> ComfyResult<NodeOutputs> {
        let images = inputs.get_tensor("images")?;

        let batch = images.shape()[0];
        tracing::info!(
            batch,
            shape = ?images.shape(),
            "PreviewImage: previewing images (stub)"
        );

        let mut filenames = Vec::new();
        for i in 0..batch {
            filenames.push(format!("preview_{i:05}.png"));
        }

        let ui_json = serde_json::json!({
            "images": filenames.iter().map(|f| {
                serde_json::json!({
                    "filename": f,
                    "subfolder": "",
                    "type": "temp"
                })
            }).collect::<Vec<_>>()
        });

        let mut outputs = NodeOutputs::new();
        outputs.set("ui", NodeValue::String(ui_json.to_string()));
        Ok(outputs)
    }

    fn metadata(&self) -> NodeMetadata {
        NodeMetadata {
            name: "PreviewImage".to_string(),
            display_name: "Preview Image".to_string(),
            category: "image".to_string(),
            description: "Previews images in the UI".to_string(),
            output_node: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use comfy_core::{DType, Tensor};

    #[test]
    fn test_save_image() {
        let node = SaveImage;
        let mut inputs = NodeInputs::new();
        inputs.set(
            "filename_prefix",
            NodeValue::String("ComfyUI".into()),
        );
        let image = Tensor::zeros(vec![2, 3, 512, 512], DType::F32);
        inputs.set("images", NodeValue::Tensor(image));

        let outputs = node.execute(&inputs).unwrap();
        match outputs.get("ui") {
            Some(NodeValue::String(s)) => {
                let json: serde_json::Value = serde_json::from_str(s).unwrap();
                let images = json["images"].as_array().unwrap();
                assert_eq!(images.len(), 2);
                assert_eq!(images[0]["filename"], "ComfyUI_00000.png");
                assert_eq!(images[1]["filename"], "ComfyUI_00001.png");
                assert_eq!(images[0]["type"], "output");
            }
            other => panic!("Expected ui string, got {other:?}"),
        }
    }

    #[test]
    fn test_save_image_metadata() {
        let meta = SaveImage.metadata();
        assert_eq!(meta.name, "SaveImage");
        assert_eq!(meta.category, "image");
        assert!(meta.output_node);
    }

    #[test]
    fn test_preview_image() {
        let node = PreviewImage;
        let mut inputs = NodeInputs::new();
        let image = Tensor::zeros(vec![1, 3, 512, 512], DType::F32);
        inputs.set("images", NodeValue::Tensor(image));

        let outputs = node.execute(&inputs).unwrap();
        match outputs.get("ui") {
            Some(NodeValue::String(s)) => {
                let json: serde_json::Value = serde_json::from_str(s).unwrap();
                let images = json["images"].as_array().unwrap();
                assert_eq!(images.len(), 1);
                assert_eq!(images[0]["filename"], "preview_00000.png");
                assert_eq!(images[0]["type"], "temp");
            }
            other => panic!("Expected ui string, got {other:?}"),
        }
    }

    #[test]
    fn test_preview_image_metadata() {
        let meta = PreviewImage.metadata();
        assert_eq!(meta.name, "PreviewImage");
        assert_eq!(meta.category, "image");
        assert!(meta.output_node);
    }

    #[test]
    fn test_save_image_missing_prefix() {
        let node = SaveImage;
        let mut inputs = NodeInputs::new();
        let image = Tensor::zeros(vec![1, 3, 512, 512], DType::F32);
        inputs.set("images", NodeValue::Tensor(image));
        // Missing filename_prefix
        assert!(node.execute(&inputs).is_err());
    }

    #[test]
    fn test_preview_image_missing_images() {
        let node = PreviewImage;
        let inputs = NodeInputs::new();
        assert!(node.execute(&inputs).is_err());
    }

    #[test]
    fn test_save_image_u8_writes_png() {
        // Create a 4x4 RGB U8 tensor
        let data = vec![128u8; 4 * 4 * 3]; // 4x4 pixels, 3 channels
        let tensor = Tensor::from_raw(data, vec![4, 4, 3], DType::U8).unwrap();

        let tmp_dir = std::env::temp_dir();
        let prefix = tmp_dir.join("comfy_save_test");
        let prefix_str = prefix.to_str().unwrap();

        let node = SaveImage;
        let mut inputs = NodeInputs::new();
        inputs.set("filename_prefix", NodeValue::String(prefix_str.to_string()));
        inputs.set("images", NodeValue::Tensor(tensor));

        let outputs = node.execute(&inputs).unwrap();
        match outputs.get("ui") {
            Some(NodeValue::String(s)) => {
                let json: serde_json::Value = serde_json::from_str(s).unwrap();
                let images = json["images"].as_array().unwrap();
                assert_eq!(images.len(), 1);

                let filename = images[0]["filename"].as_str().unwrap();
                // Verify the PNG file was actually written
                assert!(
                    std::path::Path::new(filename).exists(),
                    "PNG file should exist at {filename}"
                );

                // Cleanup
                let _ = std::fs::remove_file(filename);
            }
            other => panic!("Expected ui string, got {other:?}"),
        }
    }
}
