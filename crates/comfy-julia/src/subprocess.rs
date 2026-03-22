/// Julia subprocess integration via JSON-over-stdio IPC.
///
/// This module is only compiled when the `julia` feature is enabled.
/// It launches a Julia process running `julia/src/server.jl` and communicates
/// via newline-delimited JSON on stdin/stdout.
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};

use comfy_core::{ComfyError, ComfyResult};
use serde::{Deserialize, Serialize};

/// Request message sent to the Julia server via stdin.
#[derive(Debug, Serialize)]
pub struct JuliaRequest {
    /// Unique request ID for correlation.
    pub id: u64,
    /// Method name to invoke (e.g., "sample", "schedule").
    pub method: String,
    /// JSON-encoded parameters.
    pub params: serde_json::Value,
}

/// Response message received from the Julia server via stdout.
#[derive(Debug, Deserialize)]
pub struct JuliaResponse {
    /// Request ID this response corresponds to.
    pub id: u64,
    /// Result payload (null on error).
    pub result: Option<serde_json::Value>,
    /// Error message (null on success).
    pub error: Option<String>,
}

/// Manages a Julia subprocess for high-performance sampling/scheduling.
pub struct JuliaProcess {
    child: Option<Child>,
    next_id: u64,
    julia_binary: String,
    server_script: PathBuf,
}

impl JuliaProcess {
    /// Create a new JuliaProcess configuration.
    ///
    /// `julia_binary`: path or name of the Julia executable (e.g., "julia")
    /// `server_script`: path to the Julia server script (e.g., "julia/src/server.jl")
    pub fn new(julia_binary: impl Into<String>, server_script: impl Into<PathBuf>) -> Self {
        Self {
            child: None,
            next_id: 1,
            julia_binary: julia_binary.into(),
            server_script: server_script.into(),
        }
    }

    /// Start the Julia subprocess.
    ///
    /// Returns an error if Julia is not installed or the server script is not found.
    pub fn start(&mut self) -> ComfyResult<()> {
        if self.child.is_some() {
            return Err(ComfyError::InferenceError(
                "Julia process is already running".to_string(),
            ));
        }

        if !self.server_script.exists() {
            return Err(ComfyError::InferenceError(format!(
                "Julia server script not found: {}",
                self.server_script.display()
            )));
        }

        let child = Command::new(&self.julia_binary)
            .arg("--startup-file=no")
            .arg("--project=.")
            .arg(self.server_script.as_os_str())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| {
                ComfyError::InferenceError(format!(
                    "failed to start Julia process '{}': {e}",
                    self.julia_binary
                ))
            })?;

        tracing::info!(
            julia = %self.julia_binary,
            script = %self.server_script.display(),
            pid = child.id(),
            "Julia subprocess started"
        );

        self.child = Some(child);
        Ok(())
    }

    /// Send a request to the Julia process and wait for a response.
    ///
    /// The protocol is newline-delimited JSON:
    /// - Write one JSON line to stdin
    /// - Read one JSON line from stdout
    pub fn call(
        &mut self,
        method: &str,
        params: serde_json::Value,
    ) -> ComfyResult<serde_json::Value> {
        let child = self.child.as_mut().ok_or_else(|| {
            ComfyError::InferenceError("Julia process is not running".to_string())
        })?;

        let request = JuliaRequest {
            id: self.next_id,
            method: method.to_string(),
            params,
        };
        self.next_id += 1;

        // Write request as a single JSON line to stdin
        let stdin = child.stdin.as_mut().ok_or_else(|| {
            ComfyError::InferenceError("Julia process stdin is not available".to_string())
        })?;

        let mut request_json = serde_json::to_string(&request)?;
        request_json.push('\n');
        stdin.write_all(request_json.as_bytes()).map_err(|e| {
            ComfyError::InferenceError(format!("failed to write to Julia stdin: {e}"))
        })?;
        stdin.flush().map_err(|e| {
            ComfyError::InferenceError(format!("failed to flush Julia stdin: {e}"))
        })?;

        // Read response as a single JSON line from stdout
        let stdout = child.stdout.as_mut().ok_or_else(|| {
            ComfyError::InferenceError("Julia process stdout is not available".to_string())
        })?;

        let mut reader = BufReader::new(stdout);
        let mut response_line = String::new();
        reader.read_line(&mut response_line).map_err(|e| {
            ComfyError::InferenceError(format!("failed to read from Julia stdout: {e}"))
        })?;

        if response_line.is_empty() {
            return Err(ComfyError::InferenceError(
                "Julia process returned empty response (process may have exited)".to_string(),
            ));
        }

        let response: JuliaResponse = serde_json::from_str(&response_line).map_err(|e| {
            ComfyError::InferenceError(format!(
                "failed to parse Julia response: {e}\nraw: {response_line}"
            ))
        })?;

        if response.id != request.id {
            return Err(ComfyError::InferenceError(format!(
                "Julia response ID mismatch: expected {}, got {}",
                request.id, response.id
            )));
        }

        if let Some(error) = response.error {
            return Err(ComfyError::InferenceError(format!(
                "Julia error: {error}"
            )));
        }

        response.result.ok_or_else(|| {
            ComfyError::InferenceError("Julia response has neither result nor error".to_string())
        })
    }

    /// Stop the Julia subprocess gracefully.
    pub fn stop(&mut self) -> ComfyResult<()> {
        if let Some(mut child) = self.child.take() {
            // Try to send a quit command first
            if let Some(stdin) = child.stdin.as_mut() {
                let quit_msg = r#"{"id":0,"method":"quit","params":null}"#;
                let _ = writeln!(stdin, "{quit_msg}");
                let _ = stdin.flush();
            }

            // Drop stdin to signal EOF
            drop(child.stdin.take());

            // Wait briefly, then kill if necessary
            match child.try_wait() {
                Ok(Some(status)) => {
                    tracing::info!(?status, "Julia process exited");
                }
                Ok(None) => {
                    tracing::warn!("Julia process did not exit, killing");
                    let _ = child.kill();
                    let _ = child.wait();
                }
                Err(e) => {
                    tracing::error!(?e, "failed to check Julia process status");
                    let _ = child.kill();
                }
            }
        }
        Ok(())
    }

    /// Check if the Julia subprocess is currently running.
    pub fn is_running(&mut self) -> bool {
        match &mut self.child {
            Some(child) => match child.try_wait() {
                Ok(Some(_)) => {
                    // Process has exited
                    self.child = None;
                    false
                }
                Ok(None) => true,
                Err(_) => false,
            },
            None => false,
        }
    }
}

impl Drop for JuliaProcess {
    fn drop(&mut self) {
        let _ = self.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_julia_process_not_started() {
        let mut proc = JuliaProcess::new("julia", "julia/src/server.jl");

        // Should not be running before start
        assert!(!proc.is_running());

        // Calling without starting should error
        let result = proc.call("test", serde_json::json!({}));
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("not running"),
            "Expected 'not running' in error: {err_msg}"
        );
    }

    #[test]
    fn test_julia_process_double_start_errors() {
        // We can't actually start Julia in tests, but we can test the script-not-found path
        let mut proc = JuliaProcess::new("julia", "nonexistent_script.jl");
        let result = proc.start();
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("not found"),
            "Expected 'not found' in error: {err_msg}"
        );
    }

    #[test]
    fn test_julia_process_stop_when_not_running() {
        let mut proc = JuliaProcess::new("julia", "julia/src/server.jl");
        // Stopping when not running should be fine (no-op)
        assert!(proc.stop().is_ok());
    }

    #[test]
    fn test_julia_request_serialization() {
        let req = JuliaRequest {
            id: 1,
            method: "sample".to_string(),
            params: serde_json::json!({"steps": 20, "sampler": "euler"}),
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"id\":1"));
        assert!(json.contains("\"method\":\"sample\""));
        assert!(json.contains("\"steps\":20"));
    }

    #[test]
    fn test_julia_response_deserialization_ok() {
        let json = r#"{"id":1,"result":{"sigmas":[1.0,0.5,0.0]},"error":null}"#;
        let resp: JuliaResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.id, 1);
        assert!(resp.result.is_some());
        assert!(resp.error.is_none());
    }

    #[test]
    fn test_julia_response_deserialization_error() {
        let json = r#"{"id":2,"result":null,"error":"unknown method"}"#;
        let resp: JuliaResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.id, 2);
        assert!(resp.result.is_none());
        assert_eq!(resp.error.as_deref(), Some("unknown method"));
    }
}
