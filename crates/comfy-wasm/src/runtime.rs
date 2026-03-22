//! WASM plugin runtime — loads, manages, and executes WASM modules.
//!
//! Each module is registered under a unique name together with its
//! [`WasmNodeMetadata`].  Execution honours the [`SandboxConfig`] limits.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use tracing::info;

use crate::node_api::{WasmNodeMetadata, WasmNodeRequest, WasmNodeResponse};
use crate::sandbox::SandboxConfig;

/// Describes a loaded WASM module.
#[derive(Debug)]
struct LoadedModule {
    path: PathBuf,
    metadata: WasmNodeMetadata,
}

/// Runtime that loads and executes WASM plugin modules within a sandbox.
#[derive(Debug)]
pub struct WasmRuntime {
    config: SandboxConfig,
    modules: HashMap<String, LoadedModule>,
}

impl WasmRuntime {
    /// Create a new runtime with the given sandbox configuration.
    pub fn new(config: SandboxConfig) -> Self {
        info!("WasmRuntime created (max_mem={}B)", config.max_memory_bytes);
        Self {
            config,
            modules: HashMap::new(),
        }
    }

    /// Register a WASM module by name.
    ///
    /// Returns an error if the file does not exist.
    pub fn register(
        &mut self,
        name: &str,
        path: &Path,
        metadata: WasmNodeMetadata,
    ) -> Result<(), String> {
        if !path.exists() {
            return Err(format!("WASM module not found: {}", path.display()));
        }

        self.modules.insert(
            name.to_string(),
            LoadedModule {
                path: path.to_path_buf(),
                metadata,
            },
        );

        info!("Registered WASM module: {name}");
        Ok(())
    }

    /// Execute a named module with the given request.
    ///
    /// - Returns `Err` if the module is not loaded.
    /// - Returns `Err` if the request payload exceeds the memory limit.
    pub fn execute(&self, name: &str, request: &WasmNodeRequest) -> Result<WasmNodeResponse, String> {
        let module = self
            .modules
            .get(name)
            .ok_or_else(|| format!("Module not loaded: {name}"))?;

        // Estimate payload size (serialised JSON) as a rough memory check.
        let payload_json =
            serde_json::to_string(request).map_err(|e| format!("Serialisation error: {e}"))?;
        let payload_size = payload_json.len() as u64;

        if payload_size > self.config.max_memory_bytes {
            return Err(format!(
                "Payload size ({payload_size}B) exceeds memory limit ({}B)",
                self.config.max_memory_bytes
            ));
        }

        info!(
            "Executing module '{}' from {}",
            name,
            module.path.display()
        );

        // Placeholder: real implementation would instantiate the WASM module
        // via wasmtime::Engine / Store / Instance and call the exported entry point.
        Ok(WasmNodeResponse {
            outputs: HashMap::new(),
            error: None,
        })
    }

    /// Returns a list of all registered module names.
    pub fn list_modules(&self) -> Vec<String> {
        self.modules.keys().cloned().collect()
    }

    /// Returns the metadata for a registered module, if present.
    pub fn get_metadata(&self, name: &str) -> Option<&WasmNodeMetadata> {
        self.modules.get(name).map(|m| &m.metadata)
    }

    /// Unload (remove) a module by name.  Returns `true` if it was present.
    pub fn unload(&mut self, name: &str) -> bool {
        let removed = self.modules.remove(name).is_some();
        if removed {
            info!("Unloaded WASM module: {name}");
        }
        removed
    }

    /// Returns a reference to the sandbox configuration.
    pub fn config(&self) -> &SandboxConfig {
        &self.config
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_metadata(name: &str) -> WasmNodeMetadata {
        WasmNodeMetadata {
            name: name.to_string(),
            category: "test".to_string(),
            description: "test module".to_string(),
            inputs: vec![],
            outputs: vec![],
        }
    }

    #[test]
    fn test_runtime_creation() {
        let rt = WasmRuntime::new(SandboxConfig::default());
        assert!(rt.list_modules().is_empty());
        assert_eq!(rt.config().max_memory_bytes, 64 * 1024 * 1024);
    }

    #[test]
    fn test_register_nonexistent_file() {
        let mut rt = WasmRuntime::new(SandboxConfig::default());
        let result = rt.register(
            "bad",
            Path::new("/nonexistent/plugin.wasm"),
            dummy_metadata("bad"),
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not found"));
    }

    #[test]
    fn test_execute_unloaded_module() {
        let rt = WasmRuntime::new(SandboxConfig::default());
        let req = WasmNodeRequest {
            node_name: "ghost".to_string(),
            inputs: HashMap::new(),
        };
        let result = rt.execute("ghost", &req);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not loaded"));
    }

    #[test]
    fn test_unload_module() {
        let mut rt = WasmRuntime::new(SandboxConfig::default());
        // Not registered → unload returns false
        assert!(!rt.unload("missing"));

        // After (hypothetical) registration the list would shrink;
        // we can still verify the empty-case logic.
        assert!(rt.list_modules().is_empty());
    }
}
