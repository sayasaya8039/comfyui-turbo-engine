//! comfy-nodes: Standard ComfyUI node implementations (stubs).
//!
//! This crate provides the 7 standard nodes that form the core of any
//! Stable Diffusion workflow in ComfyUI:
//!
//! - **CheckpointLoaderSimple** — loads model/CLIP/VAE from a checkpoint
//! - **CLIPTextEncode** — encodes text into conditioning
//! - **KSampler** — iterative denoising sampler
//! - **EmptyLatentImage** — creates a blank latent tensor
//! - **VAEDecode** — decodes latent to pixel image
//! - **SaveImage** — saves images to disk
//! - **PreviewImage** — previews images in the UI

pub mod conditioning;
pub mod image;
pub mod latent;
pub mod loaders;
pub mod pipeline;
pub mod sampling;

pub use conditioning::CLIPTextEncode;
pub use image::{PreviewImage, SaveImage};
pub use pipeline::{PipelineConfig, PipelineExecutor, PipelineResult, PipelineStats};
pub use latent::{EmptyLatentImage, VAEDecode};
pub use loaders::CheckpointLoaderSimple;
pub use sampling::KSampler;

use comfy_core::NodeRegistry;

/// Register all 7 standard nodes into the given registry.
pub fn register_all_nodes(registry: &mut NodeRegistry) {
    registry.register("CheckpointLoaderSimple", || {
        Box::new(CheckpointLoaderSimple)
    });
    registry.register("CLIPTextEncode", || Box::new(CLIPTextEncode));
    registry.register("KSampler", || Box::new(KSampler));
    registry.register("EmptyLatentImage", || Box::new(EmptyLatentImage));
    registry.register("VAEDecode", || Box::new(VAEDecode));
    registry.register("SaveImage", || Box::new(SaveImage));
    registry.register("PreviewImage", || Box::new(PreviewImage));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_all_nodes() {
        let mut registry = NodeRegistry::new();
        register_all_nodes(&mut registry);

        let expected = [
            "CheckpointLoaderSimple",
            "CLIPTextEncode",
            "KSampler",
            "EmptyLatentImage",
            "VAEDecode",
            "SaveImage",
            "PreviewImage",
        ];

        for name in &expected {
            assert!(
                registry.contains(name),
                "Node '{name}' was not registered"
            );
        }

        // Verify exact count
        assert_eq!(registry.list().len(), 7);
    }

    #[test]
    fn test_all_nodes_can_be_created() {
        let mut registry = NodeRegistry::new();
        register_all_nodes(&mut registry);

        for name in registry.list() {
            let node = registry.create(&name);
            assert!(node.is_some(), "Failed to create node '{name}'");
            let meta = node.unwrap().metadata();
            assert_eq!(meta.name, name);
        }
    }

    #[test]
    fn test_output_nodes_correctly_marked() {
        let mut registry = NodeRegistry::new();
        register_all_nodes(&mut registry);

        let save = registry.create("SaveImage").unwrap();
        assert!(save.metadata().output_node);

        let preview = registry.create("PreviewImage").unwrap();
        assert!(preview.metadata().output_node);

        // Non-output nodes
        for name in ["CheckpointLoaderSimple", "CLIPTextEncode", "KSampler", "EmptyLatentImage", "VAEDecode"] {
            let node = registry.create(name).unwrap();
            assert!(
                !node.metadata().output_node,
                "{name} should not be an output node"
            );
        }
    }
}
