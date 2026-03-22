use std::path::Path;

use comfy_core::ComfyResult;

/// Information about a discovered Python custom node package.
#[derive(Debug, Clone, PartialEq)]
pub struct PythonNodeInfo {
    /// Full path to the module directory.
    pub module_path: String,
    /// Name of the package (directory name).
    pub package_name: String,
}

/// Scan a directory for Python custom node packages.
///
/// A valid custom node package is a subdirectory that contains an `__init__.py` file.
/// Returns an empty vec if the directory does not exist or contains no valid packages.
pub fn scan_custom_nodes(dir: &Path) -> ComfyResult<Vec<PythonNodeInfo>> {
    if !dir.exists() || !dir.is_dir() {
        tracing::debug!("custom nodes directory does not exist: {}", dir.display());
        return Ok(Vec::new());
    }

    let mut nodes = Vec::new();

    let entries = std::fs::read_dir(dir)?;
    for entry in entries {
        let entry = entry?;
        let path = entry.path();

        if !path.is_dir() {
            continue;
        }

        let init_py = path.join("__init__.py");
        if init_py.exists() {
            let package_name = path
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default();

            nodes.push(PythonNodeInfo {
                module_path: path.to_string_lossy().to_string(),
                package_name,
            });
        }
    }

    tracing::info!("found {} custom node packages in {}", nodes.len(), dir.display());
    Ok(nodes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_nonexistent_dir_returns_empty() {
        let dir = PathBuf::from("/nonexistent/custom_nodes_xyz_12345");
        let result = scan_custom_nodes(&dir).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_empty_dir_returns_empty() {
        let dir = std::env::temp_dir().join("comfy_python_test_empty_dir");
        let _ = std::fs::create_dir_all(&dir);

        let result = scan_custom_nodes(&dir).unwrap();
        assert!(result.is_empty());

        let _ = std::fs::remove_dir_all(&dir);
    }
}
