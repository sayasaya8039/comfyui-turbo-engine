//! WASM sandbox configuration for plugin isolation.
//!
//! Defines resource limits and permission boundaries for WASM plugins.
//! By default all settings are restrictive; use [`SandboxConfig::permissive`]
//! for development/testing.

use std::path::{Path, PathBuf};

/// Configuration that governs the sandbox in which a WASM plugin runs.
#[derive(Debug, Clone)]
pub struct SandboxConfig {
    /// Maximum linear memory the WASM module may allocate (bytes).
    pub max_memory_bytes: u64,
    /// Maximum wall-clock execution time per invocation (milliseconds).
    pub max_time_ms: u64,
    /// Optional timeout in seconds (takes precedence over `max_time_ms` when set).
    pub timeout_seconds: Option<u64>,
    /// Directories the plugin is allowed to *read* from.
    pub allowed_read_paths: Vec<PathBuf>,
    /// Directories the plugin is allowed to *write* to.
    pub allowed_write_paths: Vec<PathBuf>,
    /// Whether outbound network access is permitted.
    pub allow_network: bool,
}

impl Default for SandboxConfig {
    /// Returns a restrictive default: 64 MiB memory, 5 s timeout,
    /// no filesystem access, no network.
    fn default() -> Self {
        Self {
            max_memory_bytes: 64 * 1024 * 1024, // 64 MiB
            max_time_ms: 5_000,                  // 5 seconds
            timeout_seconds: None,
            allowed_read_paths: Vec::new(),
            allowed_write_paths: Vec::new(),
            allow_network: false,
        }
    }
}

impl SandboxConfig {
    /// Returns a permissive configuration suitable for trusted plugins
    /// or development: 1 GiB memory, 60 s timeout, network allowed,
    /// and read access to the current directory.
    pub fn permissive() -> Self {
        Self {
            max_memory_bytes: 1024 * 1024 * 1024, // 1 GiB
            max_time_ms: 60_000,                   // 60 seconds
            timeout_seconds: Some(60),
            allowed_read_paths: vec![PathBuf::from(".")],
            allowed_write_paths: vec![PathBuf::from(".")],
            allow_network: true,
        }
    }

    /// Returns `true` if the plugin is allowed to read `path`.
    ///
    /// A path is readable when it starts with any entry in
    /// [`allowed_read_paths`](Self::allowed_read_paths).
    pub fn can_read(&self, path: &Path) -> bool {
        self.allowed_read_paths
            .iter()
            .any(|allowed| path.starts_with(allowed))
    }

    /// Returns `true` if the plugin is allowed to write to `path`.
    ///
    /// A path is writable when it starts with any entry in
    /// [`allowed_write_paths`](Self::allowed_write_paths).
    pub fn can_write(&self, path: &Path) -> bool {
        self.allowed_write_paths
            .iter()
            .any(|allowed| path.starts_with(allowed))
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_is_restrictive() {
        let cfg = SandboxConfig::default();
        assert_eq!(cfg.max_memory_bytes, 64 * 1024 * 1024);
        assert_eq!(cfg.max_time_ms, 5_000);
        assert!(cfg.allowed_read_paths.is_empty());
        assert!(cfg.allowed_write_paths.is_empty());
        assert!(!cfg.allow_network);
    }

    #[test]
    fn test_permissive_config() {
        let cfg = SandboxConfig::permissive();
        assert_eq!(cfg.max_memory_bytes, 1024 * 1024 * 1024);
        assert_eq!(cfg.max_time_ms, 60_000);
        assert!(cfg.allow_network);
        assert!(!cfg.allowed_read_paths.is_empty());
        assert!(!cfg.allowed_write_paths.is_empty());
    }

    #[test]
    fn test_path_permissions() {
        let cfg = SandboxConfig {
            allowed_read_paths: vec![PathBuf::from("/data/models")],
            allowed_write_paths: vec![PathBuf::from("/tmp/output")],
            ..SandboxConfig::default()
        };

        // Read permissions
        assert!(cfg.can_read(Path::new("/data/models/sd15.safetensors")));
        assert!(!cfg.can_read(Path::new("/etc/passwd")));

        // Write permissions
        assert!(cfg.can_write(Path::new("/tmp/output/result.png")));
        assert!(!cfg.can_write(Path::new("/data/models/overwrite.bin")));
    }
}
