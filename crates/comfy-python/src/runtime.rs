use comfy_core::{ComfyError, ComfyResult};

/// Python runtime manager wrapping pyo3 initialization and evaluation.
pub struct PythonRuntime {
    initialized: bool,
}

impl PythonRuntime {
    /// Create a new PythonRuntime (not yet initialized).
    pub fn new() -> Self {
        Self { initialized: false }
    }

    /// Initialize the Python interpreter.
    ///
    /// When the `python` feature is enabled, this calls pyo3 to prepare the GIL.
    /// Without the feature, this returns an error.
    pub fn initialize(&mut self) -> ComfyResult<()> {
        #[cfg(feature = "python")]
        {
            pyo3::prepare_freethreaded_python();
            self.initialized = true;
            tracing::info!("Python runtime initialized via pyo3");
            Ok(())
        }
        #[cfg(not(feature = "python"))]
        {
            Err(ComfyError::InferenceError(
                "python feature is not enabled — cannot initialize Python runtime".to_string(),
            ))
        }
    }

    /// Check whether the runtime has been successfully initialized.
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Evaluate a Python expression and return its string representation.
    ///
    /// Requires the `python` feature and a prior call to `initialize()`.
    pub fn eval(&self, code: &str) -> ComfyResult<String> {
        if !self.initialized {
            return Err(ComfyError::InferenceError(
                "Python runtime not initialized".to_string(),
            ));
        }

        #[cfg(feature = "python")]
        {
            pyo3::Python::with_gil(|py| {
                let result = py
                    .eval(
                        &std::ffi::CString::new(code).map_err(|e| {
                            ComfyError::InferenceError(format!("invalid code string: {e}"))
                        })?,
                        None,
                        None,
                    )
                    .map_err(|e| {
                        ComfyError::InferenceError(format!("Python eval error: {e}"))
                    })?;
                let s = result.str().map_err(|e| {
                    ComfyError::InferenceError(format!("Python str conversion error: {e}"))
                })?;
                Ok(s.to_string())
            })
        }
        #[cfg(not(feature = "python"))]
        {
            let _ = code;
            Err(ComfyError::InferenceError(
                "python feature is not enabled — cannot eval".to_string(),
            ))
        }
    }
}

impl Default for PythonRuntime {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runtime_creation() {
        let rt = PythonRuntime::new();
        assert!(!rt.is_initialized());
    }

    #[test]
    fn test_runtime_without_python_feature() {
        // Without the `python` feature, initialize should fail.
        #[cfg(not(feature = "python"))]
        {
            let mut rt = PythonRuntime::new();
            let result = rt.initialize();
            assert!(result.is_err());
            assert!(!rt.is_initialized());
        }
    }
}
