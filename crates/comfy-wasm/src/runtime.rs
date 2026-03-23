//! WASM plugin runtime — loads, manages, and executes WASM modules.
//!
//! Each module is registered under a unique name together with its
//! [`WasmNodeMetadata`].  Execution honours the [`SandboxConfig`] limits.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use tracing::info;
use wasmtime::*;

use crate::node_api::{WasmNodeMetadata, WasmNodeRequest, WasmNodeResponse};
use crate::sandbox::SandboxConfig;

/// Describes a loaded WASM module.
#[derive(Debug)]
struct LoadedModule {
    path: PathBuf,
    metadata: WasmNodeMetadata,
    module_bytes: Vec<u8>,
}

/// Runtime that loads and executes WASM plugin modules within a sandbox.
pub struct WasmRuntime {
    config: SandboxConfig,
    engine: Engine,
    modules: HashMap<String, LoadedModule>,
}

// Engine does not implement Debug, so provide a manual impl.
impl std::fmt::Debug for WasmRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WasmRuntime")
            .field("config", &self.config)
            .field("modules", &self.modules.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl WasmRuntime {
    /// Create a new runtime with the given sandbox configuration.
    pub fn new(config: SandboxConfig) -> Self {
        let mut engine_config = Config::new();
        engine_config.consume_fuel(true); // Enable fuel-based execution limits

        let engine = Engine::new(&engine_config).expect("Failed to create wasmtime engine");

        info!(
            "WasmRuntime created with wasmtime engine (max_mem={}B)",
            config.max_memory_bytes
        );
        Self {
            config,
            engine,
            modules: HashMap::new(),
        }
    }

    /// Register a WASM module by name.
    ///
    /// Returns an error if the file does not exist or the module is invalid.
    pub fn register(
        &mut self,
        name: &str,
        path: &Path,
        metadata: WasmNodeMetadata,
    ) -> Result<(), String> {
        if !path.exists() {
            return Err(format!("WASM module not found: {}", path.display()));
        }

        let module_bytes =
            std::fs::read(path).map_err(|e| format!("Failed to read WASM module: {e}"))?;

        // Validate the module compiles
        Module::new(&self.engine, &module_bytes)
            .map_err(|e| format!("Invalid WASM module: {e}"))?;

        self.modules.insert(
            name.to_string(),
            LoadedModule {
                path: path.to_path_buf(),
                metadata,
                module_bytes,
            },
        );

        info!("Registered WASM module: {name}");
        Ok(())
    }

    /// Execute a named module with the given request.
    ///
    /// - Returns `Err` if the module is not loaded.
    /// - Returns `Err` if the request payload exceeds the memory limit.
    pub fn execute(
        &self,
        name: &str,
        request: &WasmNodeRequest,
    ) -> Result<WasmNodeResponse, String> {
        let loaded = self
            .modules
            .get(name)
            .ok_or_else(|| format!("Module not loaded: {name}"))?;

        // Serialize request
        let payload_json =
            serde_json::to_string(request).map_err(|e| format!("Serialisation error: {e}"))?;
        let payload_size = payload_json.len() as u64;

        if payload_size > self.config.max_memory_bytes {
            return Err(format!(
                "Payload size ({payload_size}B) exceeds memory limit ({}B)",
                self.config.max_memory_bytes
            ));
        }

        // Compile module
        let module = Module::new(&self.engine, &loaded.module_bytes)
            .map_err(|e| format!("Module compilation failed: {e}"))?;

        // Create store with fuel limit
        let mut store = Store::new(&self.engine, ());
        let timeout_secs = self
            .config
            .timeout_seconds
            .unwrap_or(self.config.max_time_ms / 1000);
        let fuel_limit = timeout_secs.max(1) * 1_000_000;
        store
            .set_fuel(fuel_limit)
            .map_err(|e| format!("Failed to set fuel: {e}"))?;

        // Create linker and define host functions
        let mut linker = Linker::new(&self.engine);

        // Add host function for logging
        linker
            .func_wrap("env", "log", |_caller: Caller<'_, ()>, ptr: i32, len: i32| {
                tracing::debug!("WASM log: ptr={ptr}, len={len}");
            })
            .map_err(|e| format!("Failed to define log function: {e}"))?;

        // Try to instantiate
        let instance = linker
            .instantiate(&mut store, &module)
            .map_err(|e| format!("Module instantiation failed: {e}"))?;

        // Try to call the exported "execute" function
        // Convention: execute(input_ptr, input_len) -> output_ptr
        if let Ok(execute_fn) =
            instance.get_typed_func::<(i32, i32), i32>(&mut store, "execute")
        {
            // Write input to WASM memory
            let memory = instance
                .get_memory(&mut store, "memory")
                .ok_or("Module has no 'memory' export")?;

            let input_bytes = payload_json.as_bytes();
            let input_len = input_bytes.len();

            // Ensure WASM memory is large enough for the input
            if memory.data_size(&store) < input_len {
                let pages_needed = ((input_len / 65536) + 1) as u64;
                memory
                    .grow(&mut store, pages_needed)
                    .map_err(|e| format!("Memory grow failed: {e}"))?;
            }

            memory.data_mut(&mut store)[..input_len].copy_from_slice(input_bytes);

            // Call execute
            let result_ptr = execute_fn
                .call(&mut store, (0i32, input_len as i32))
                .map_err(|e| format!("WASM execution failed: {e}"))?;

            // Read result from memory (result_ptr points to JSON output)
            let mem_data = memory.data(&store);
            if (result_ptr as usize) < mem_data.len() {
                // Try to find null-terminated string at result_ptr
                let start = result_ptr as usize;
                let end = mem_data[start..]
                    .iter()
                    .position(|&b| b == 0)
                    .map(|pos| start + pos)
                    .unwrap_or(mem_data.len().min(start + 65536));
                let result_str = std::str::from_utf8(&mem_data[start..end]).unwrap_or("{}");
                if let Ok(response) = serde_json::from_str::<WasmNodeResponse>(result_str) {
                    return Ok(response);
                }
            }
        }

        info!(
            "Executed module '{}' (no 'execute' export found, returning empty)",
            name
        );
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
        // Not registered -> unload returns false
        assert!(!rt.unload("missing"));

        // After (hypothetical) registration the list would shrink;
        // we can still verify the empty-case logic.
        assert!(rt.list_modules().is_empty());
    }
}
