use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// ComfyUI-compatible folder paths manager.
///
/// Provides the same directory layout that ComfyUI's `folder_paths` Python module
/// exposes, allowing Rust code to locate model checkpoints, LoRAs, and other assets.
pub struct FolderPaths {
    base_path: PathBuf,
    folders: HashMap<String, Vec<PathBuf>>,
}

/// Default folder names that ComfyUI expects.
const DEFAULT_FOLDERS: &[&str] = &[
    "checkpoints",
    "loras",
    "vae",
    "embeddings",
    "controlnet",
    "upscale_models",
    "input",
    "output",
    "temp",
];

impl FolderPaths {
    /// Create a new FolderPaths rooted at `base_path`.
    ///
    /// Initializes all default ComfyUI folder entries, each pointing to
    /// `<base_path>/<folder_name>`.
    pub fn new(base_path: impl AsRef<Path>) -> Self {
        let base = base_path.as_ref().to_path_buf();
        let mut folders = HashMap::new();

        for &name in DEFAULT_FOLDERS {
            folders.insert(name.to_string(), vec![base.join(name)]);
        }

        Self {
            base_path: base,
            folders,
        }
    }

    /// Get the list of paths registered for a given folder name.
    /// Returns an empty slice if the folder name is unknown.
    pub fn get_folder_paths(&self, name: &str) -> &[PathBuf] {
        self.folders.get(name).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Add an additional path for a folder name.
    /// Creates the entry if it does not exist.
    pub fn add_folder_path(&mut self, name: &str, path: impl AsRef<Path>) {
        self.folders
            .entry(name.to_string())
            .or_default()
            .push(path.as_ref().to_path_buf());
    }

    /// Return the primary input directory (`<base_path>/input`).
    pub fn get_input_directory(&self) -> PathBuf {
        self.base_path.join("input")
    }

    /// Return the primary output directory (`<base_path>/output`).
    pub fn get_output_directory(&self) -> PathBuf {
        self.base_path.join("output")
    }

    /// Return the sorted list of all registered folder names.
    pub fn folder_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.folders.keys().cloned().collect();
        names.sort();
        names
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_paths() {
        let fp = FolderPaths::new("/comfyui");
        let checkpoints = fp.get_folder_paths("checkpoints");
        assert_eq!(checkpoints.len(), 1);
        assert_eq!(checkpoints[0], PathBuf::from("/comfyui/checkpoints"));

        let loras = fp.get_folder_paths("loras");
        assert_eq!(loras.len(), 1);
        assert_eq!(loras[0], PathBuf::from("/comfyui/loras"));
    }

    #[test]
    fn test_add_custom_path() {
        let mut fp = FolderPaths::new("/comfyui");
        fp.add_folder_path("checkpoints", "/extra/models");

        let checkpoints = fp.get_folder_paths("checkpoints");
        assert_eq!(checkpoints.len(), 2);
        assert_eq!(checkpoints[1], PathBuf::from("/extra/models"));

        // Adding to a brand-new folder name
        fp.add_folder_path("custom_nodes", "/my/nodes");
        let custom = fp.get_folder_paths("custom_nodes");
        assert_eq!(custom.len(), 1);
        assert_eq!(custom[0], PathBuf::from("/my/nodes"));
    }

    #[test]
    fn test_unknown_folder_returns_empty() {
        let fp = FolderPaths::new("/comfyui");
        let unknown = fp.get_folder_paths("nonexistent_folder");
        assert!(unknown.is_empty());
    }

    #[test]
    fn test_input_output_directories() {
        let fp = FolderPaths::new("/comfyui");
        assert_eq!(fp.get_input_directory(), PathBuf::from("/comfyui/input"));
        assert_eq!(fp.get_output_directory(), PathBuf::from("/comfyui/output"));
    }

    #[test]
    fn test_folder_names() {
        let fp = FolderPaths::new("/comfyui");
        let names = fp.folder_names();

        assert_eq!(names.len(), 9);
        // Sorted alphabetically
        assert_eq!(names[0], "checkpoints");
        assert!(names.contains(&"loras".to_string()));
        assert!(names.contains(&"temp".to_string()));
        assert!(names.contains(&"upscale_models".to_string()));
    }
}
