use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use comfy_core::ComfyError;
use ort::session::Session;

use crate::provider::ExecutionProvider;

/// Thread-safe cache for ONNX Runtime sessions, keyed by model file path.
///
/// Sessions are expensive to create (model parsing, graph optimization, memory allocation).
/// This cache ensures each model is loaded only once and subsequent requests receive
/// a reference-counted handle to the existing session.
pub struct SessionCache {
    sessions: Mutex<HashMap<String, Arc<Session>>>,
    provider: ExecutionProvider,
}

impl SessionCache {
    /// Create a new empty session cache using the given execution provider.
    pub fn new(provider: ExecutionProvider) -> Self {
        Self {
            sessions: Mutex::new(HashMap::new()),
            provider,
        }
    }

    /// Returns the execution provider this cache was configured with.
    pub fn provider(&self) -> ExecutionProvider {
        self.provider
    }

    /// Get a cached session for the given model path, or load it if not yet cached.
    ///
    /// On first access for a given path, the ONNX model is loaded and an optimized
    /// session is created with the configured execution provider. Subsequent calls
    /// return a clone of the `Arc<Session>`.
    pub fn get_or_load(&self, model_path: &str) -> Result<Arc<Session>, ComfyError> {
        let mut sessions = self.sessions.lock().map_err(|e| {
            ComfyError::InferenceError(format!("session cache lock poisoned: {e}"))
        })?;

        if let Some(session) = sessions.get(model_path) {
            tracing::debug!(path = model_path, "returning cached session");
            return Ok(Arc::clone(session));
        }

        tracing::info!(path = model_path, provider = %self.provider, "loading new ONNX session");
        let session = self.create_session(model_path)?;
        let session = Arc::new(session);
        sessions.insert(model_path.to_string(), Arc::clone(&session));
        Ok(session)
    }

    /// Create a new ONNX Runtime session for the given model path.
    fn create_session(&self, model_path: &str) -> Result<Session, ComfyError> {
        if !Path::new(model_path).exists() {
            return Err(ComfyError::InferenceError(format!(
                "model file not found: {model_path}"
            )));
        }

        let builder = Session::builder().map_err(|e| {
            ComfyError::InferenceError(format!("failed to create session builder: {e}"))
        })?;

        let mut builder = self.apply_execution_provider(builder)?;

        builder.commit_from_file(model_path).map_err(|e| {
            ComfyError::InferenceError(format!(
                "failed to load model '{model_path}': {e}"
            ))
        })
    }

    /// Apply the configured execution provider to the session builder.
    fn apply_execution_provider(
        &self,
        builder: ort::session::builder::SessionBuilder,
    ) -> Result<ort::session::builder::SessionBuilder, ComfyError> {
        use ort::ep;

        let providers: Vec<ort::ep::ExecutionProviderDispatch> = match self.provider {
            ExecutionProvider::Cuda => vec![ep::CUDA::default().build()],
            ExecutionProvider::TensorRt => vec![
                ep::TensorRT::default().build(),
                ep::CUDA::default().build(),
            ],
            ExecutionProvider::DirectMl => vec![ep::DirectML::default().build()],
            ExecutionProvider::OpenVino => vec![ep::OpenVINO::default().build()],
            ExecutionProvider::Cpu => vec![ep::CPU::default().build()],
        };

        builder
            .with_execution_providers(providers)
            .map_err(|e| {
                ComfyError::InferenceError(format!(
                    "failed to set execution provider '{}': {e}",
                    self.provider
                ))
            })
    }

    /// Returns the number of cached sessions.
    pub fn len(&self) -> usize {
        self.sessions
            .lock()
            .map(|s| s.len())
            .unwrap_or(0)
    }

    /// Returns true if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Remove a specific model from the cache.
    ///
    /// Returns `true` if the model was in the cache.
    pub fn evict(&self, model_path: &str) -> bool {
        self.sessions
            .lock()
            .map(|mut s| s.remove(model_path).is_some())
            .unwrap_or(false)
    }

    /// Remove all cached sessions.
    pub fn clear(&self) {
        if let Ok(mut sessions) = self.sessions.lock() {
            sessions.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_cache_creation() {
        let cache = SessionCache::new(ExecutionProvider::Cpu);
        assert_eq!(cache.provider(), ExecutionProvider::Cpu);
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);

        // Loading a nonexistent model should return an error
        let result = cache.get_or_load("nonexistent_model.onnx");
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("model file not found"),
            "expected 'model file not found' in error: {err_msg}"
        );
    }

    #[test]
    fn test_session_cache_evict_nonexistent() {
        let cache = SessionCache::new(ExecutionProvider::Cpu);
        // Evicting a key that was never inserted returns false
        assert!(!cache.evict("no_such_model.onnx"));
    }

    #[test]
    fn test_session_cache_clear_empty() {
        let cache = SessionCache::new(ExecutionProvider::DirectMl);
        cache.clear();
        assert!(cache.is_empty());
    }
}
