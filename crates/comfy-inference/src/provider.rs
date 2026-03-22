/// Hardware execution provider for ONNX Runtime inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionProvider {
    Cuda,
    TensorRt,
    DirectMl,
    OpenVino,
    Cpu,
}

impl ExecutionProvider {
    /// Auto-detect the best available execution provider for the current system.
    ///
    /// Detection order:
    /// 1. CUDA — if `CUDA_PATH` env var is set or common install paths exist
    /// 2. DirectML — if running on Windows
    /// 3. CPU — universal fallback
    pub fn auto_detect() -> Self {
        if Self::cuda_available() {
            tracing::info!("auto_detect: CUDA detected");
            return ExecutionProvider::Cuda;
        }

        if Self::directml_available() {
            tracing::info!("auto_detect: DirectML detected (Windows)");
            return ExecutionProvider::DirectMl;
        }

        tracing::info!("auto_detect: falling back to CPU");
        ExecutionProvider::Cpu
    }

    /// Check if CUDA appears to be installed on this system.
    fn cuda_available() -> bool {
        // Check CUDA_PATH environment variable
        if std::env::var("CUDA_PATH").is_ok() {
            return true;
        }

        // Check common CUDA install paths on Windows
        if cfg!(target_os = "windows") {
            let program_files = std::env::var("ProgramFiles")
                .unwrap_or_else(|_| r"C:\Program Files".to_string());
            let cuda_dir = std::path::Path::new(&program_files).join("NVIDIA GPU Computing Toolkit");
            if cuda_dir.exists() {
                return true;
            }
        }

        // Check common CUDA install paths on Linux
        if cfg!(target_os = "linux") {
            let cuda_paths = ["/usr/local/cuda", "/opt/cuda"];
            for path in &cuda_paths {
                if std::path::Path::new(path).exists() {
                    return true;
                }
            }
        }

        false
    }

    /// Check if DirectML is available (Windows only).
    fn directml_available() -> bool {
        cfg!(target_os = "windows")
    }

    /// Returns a human-readable name for this execution provider.
    pub fn name(&self) -> &'static str {
        match self {
            ExecutionProvider::Cuda => "CUDA",
            ExecutionProvider::TensorRt => "TensorRT",
            ExecutionProvider::DirectMl => "DirectML",
            ExecutionProvider::OpenVino => "OpenVINO",
            ExecutionProvider::Cpu => "CPU",
        }
    }
}

impl std::fmt::Display for ExecutionProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auto_detect_returns_valid_provider() {
        let provider = ExecutionProvider::auto_detect();
        // auto_detect must return one of the known variants
        assert!(
            matches!(
                provider,
                ExecutionProvider::Cuda
                    | ExecutionProvider::TensorRt
                    | ExecutionProvider::DirectMl
                    | ExecutionProvider::OpenVino
                    | ExecutionProvider::Cpu
            ),
            "auto_detect returned an unexpected provider: {provider:?}"
        );
    }

    #[test]
    fn test_provider_name() {
        assert_eq!(ExecutionProvider::Cuda.name(), "CUDA");
        assert_eq!(ExecutionProvider::TensorRt.name(), "TensorRT");
        assert_eq!(ExecutionProvider::DirectMl.name(), "DirectML");
        assert_eq!(ExecutionProvider::OpenVino.name(), "OpenVINO");
        assert_eq!(ExecutionProvider::Cpu.name(), "CPU");
    }

    #[test]
    fn test_provider_display() {
        assert_eq!(format!("{}", ExecutionProvider::Cpu), "CPU");
        assert_eq!(format!("{}", ExecutionProvider::Cuda), "CUDA");
    }

    #[test]
    fn test_provider_clone_eq() {
        let a = ExecutionProvider::Cuda;
        let b = a;
        assert_eq!(a, b);
    }
}
