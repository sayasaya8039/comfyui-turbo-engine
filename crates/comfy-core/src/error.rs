use thiserror::Error;

#[derive(Error, Debug)]
pub enum ComfyError {
    #[error("node not found: {0}")]
    NodeNotFound(String),

    #[error("invalid input for node '{node}': {message}")]
    InvalidInput { node: String, message: String },

    #[error("type mismatch: expected {expected}, got {got}")]
    TypeMismatch { expected: String, got: String },

    #[error("execution error in node '{node}': {message}")]
    ExecutionError { node: String, message: String },

    #[error("workflow has no output nodes")]
    NoOutputNodes,

    #[error("cycle detected in workflow graph")]
    CycleDetected,

    #[error("tensor error: {0}")]
    TensorError(String),

    #[error("inference error: {0}")]
    InferenceError(String),

    #[error(transparent)]
    SerdeError(#[from] serde_json::Error),

    #[error(transparent)]
    IoError(#[from] std::io::Error),
}

pub type ComfyResult<T> = Result<T, ComfyError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = ComfyError::NodeNotFound("KSampler".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("KSampler"), "Expected 'KSampler' in: {msg}");
        assert!(msg.contains("node not found"), "Expected 'node not found' in: {msg}");
    }

    #[test]
    fn test_error_invalid_input() {
        let err = ComfyError::InvalidInput {
            node: "CLIPLoader".to_string(),
            message: "missing model path".to_string(),
        };
        let msg = format!("{err}");
        assert!(msg.contains("CLIPLoader"));
        assert!(msg.contains("missing model path"));
    }

    #[test]
    fn test_error_type_mismatch() {
        let err = ComfyError::TypeMismatch {
            expected: "Tensor".to_string(),
            got: "Float".to_string(),
        };
        let msg = format!("{err}");
        assert!(msg.contains("Tensor"));
        assert!(msg.contains("Float"));
    }

    #[test]
    fn test_error_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ComfyError>();
    }

    #[test]
    fn test_comfy_result_ok() {
        let result: ComfyResult<i32> = Ok(42);
        assert_eq!(result.unwrap(), 42);
    }

    #[test]
    fn test_comfy_result_err() {
        let result: ComfyResult<i32> = Err(ComfyError::CycleDetected);
        assert!(result.is_err());
        assert!(format!("{}", result.unwrap_err()).contains("cycle detected"));
    }

    #[test]
    fn test_serde_error_conversion() {
        let bad_json = "not json";
        let result: Result<serde_json::Value, ComfyError> =
            serde_json::from_str(bad_json).map_err(ComfyError::from);
        assert!(result.is_err());
    }
}
