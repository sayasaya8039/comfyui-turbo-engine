use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::{IntoResponse, Json},
};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::history::HistoryEntry;
use crate::state::AppState;

// ---------------------------------------------------------------------------
// POST /prompt — Submit a workflow for execution
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct PromptRequest {
    pub prompt: Value,
    #[serde(default)]
    pub prompt_id: Option<String>,
    #[serde(default)]
    pub front: bool,
    #[serde(default)]
    pub client_id: Option<String>,
}

pub async fn post_prompt(
    State(state): State<AppState>,
    Json(body): Json<PromptRequest>,
) -> impl IntoResponse {
    let prompt_id = body
        .prompt_id
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

    let number = {
        let mut queue = state.queue.lock().await;
        queue.enqueue(
            prompt_id.clone(),
            body.prompt,
            body.client_id,
            body.front,
        )
    };

    (
        StatusCode::OK,
        Json(json!({
            "prompt_id": prompt_id,
            "number": number,
            "node_errors": {}
        })),
    )
}

// ---------------------------------------------------------------------------
// GET /prompt — Queue info (how many items)
// ---------------------------------------------------------------------------

pub async fn get_prompt(State(state): State<AppState>) -> impl IntoResponse {
    let queue = state.queue.lock().await;
    Json(json!({
        "exec_info": {
            "queue_remaining": queue.remaining()
        }
    }))
}

// ---------------------------------------------------------------------------
// GET /queue — Queue status (running + pending)
// ---------------------------------------------------------------------------

pub async fn get_queue(State(state): State<AppState>) -> impl IntoResponse {
    let queue = state.queue.lock().await;
    Json(json!({
        "queue_running": queue.running_items(),
        "queue_pending": queue.pending_items()
    }))
}

// ---------------------------------------------------------------------------
// POST /queue — Clear or delete queue items
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct QueueAction {
    #[serde(default)]
    pub clear: bool,
    #[serde(default)]
    pub delete: Vec<String>,
}

pub async fn post_queue(
    State(state): State<AppState>,
    Json(body): Json<QueueAction>,
) -> impl IntoResponse {
    let mut queue = state.queue.lock().await;
    if body.clear {
        queue.clear();
    }
    if !body.delete.is_empty() {
        queue.delete(&body.delete);
    }
    StatusCode::OK
}

// ---------------------------------------------------------------------------
// POST /interrupt — Interrupt current execution
// ---------------------------------------------------------------------------

pub async fn post_interrupt() -> impl IntoResponse {
    // Interrupt is a no-op until we have a cancellation mechanism
    tracing::info!("Interrupt requested (no-op for now)");
    StatusCode::OK
}

// ---------------------------------------------------------------------------
// GET /history — List all history entries
// ---------------------------------------------------------------------------

pub async fn get_history(State(state): State<AppState>) -> impl IntoResponse {
    let history = state.history.lock().await;
    let entries = history.list();

    // ComfyUI returns history as { prompt_id: { ...entry }, ... }
    let mut result = serde_json::Map::new();
    for entry in entries {
        result.insert(
            entry.prompt_id.clone(),
            format_history_entry(entry),
        );
    }
    Json(Value::Object(result))
}

// ---------------------------------------------------------------------------
// GET /history/{prompt_id} — Get a single history entry
// ---------------------------------------------------------------------------

pub async fn get_history_by_id(
    State(state): State<AppState>,
    Path(prompt_id): Path<String>,
) -> impl IntoResponse {
    let history = state.history.lock().await;
    match history.get(&prompt_id) {
        Some(entry) => {
            let mut result = serde_json::Map::new();
            result.insert(prompt_id, format_history_entry(entry));
            (StatusCode::OK, Json(Value::Object(result)))
        }
        None => (StatusCode::OK, Json(json!({}))),
    }
}

// ---------------------------------------------------------------------------
// POST /history — Clear or delete history entries
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct HistoryAction {
    #[serde(default)]
    pub clear: bool,
    #[serde(default)]
    pub delete: Vec<String>,
}

pub async fn post_history(
    State(state): State<AppState>,
    Json(body): Json<HistoryAction>,
) -> impl IntoResponse {
    let mut history = state.history.lock().await;
    if body.clear {
        history.clear();
    }
    if !body.delete.is_empty() {
        history.delete(&body.delete);
    }
    StatusCode::OK
}

// ---------------------------------------------------------------------------
// GET /system_stats — System information
// ---------------------------------------------------------------------------

pub async fn get_system_stats() -> impl IntoResponse {
    use comfy_core::{Device, HardwareDetector};

    let hw = HardwareDetector::detect();
    let devices: Vec<serde_json::Value> = hw
        .devices
        .iter()
        .map(|d| {
            json!({
                "name": d.name,
                "type": format!("{}", d.device),
                "index": match d.device {
                    Device::Gpu(i) | Device::Npu(i) => i,
                    Device::Cpu => 0,
                },
                "vram_total": d.memory_bytes,
                "vram_free": 0,
                "torch_vram_total": 0,
                "torch_vram_free": 0,
                "compute_units": d.compute_units,
            })
        })
        .collect();

    Json(json!({
        "system": {
            "os": std::env::consts::OS,
            "ram_total": 0,
            "ram_free": 0,
            "comfyui_version": "0.8.24-turbo",
            "required_frontend_version": "1.41.21",
            "installed_templates_version": "0.9.26",
            "required_templates_version": "0.9.26",
            "python_version": "N/A (native engine)",
            "pytorch_version": "N/A (ONNX Runtime)",
            "embedded_python": false,
            "argv": []
        },
        "devices": devices
    }))
}

// ---------------------------------------------------------------------------
// GET /object_info — Node definitions from registry
// ---------------------------------------------------------------------------

pub async fn get_object_info(State(state): State<AppState>) -> impl IntoResponse {
    let names = state.registry.list();
    let mut result = serde_json::Map::new();

    for name in &names {
        if let Some(node) = state.registry.create(name) {
            let meta = node.metadata();
            result.insert(
                name.clone(),
                json!({
                    "input": { "required": {}, "optional": {} },
                    "input_order": { "required": [], "optional": [] },
                    "output": [],
                    "output_is_list": [],
                    "output_name": [],
                    "name": meta.name,
                    "display_name": meta.display_name,
                    "description": meta.description,
                    "python_module": "comfy_turbo",
                    "category": meta.category,
                    "output_node": meta.output_node
                }),
            );
        }
    }

    Json(Value::Object(result))
}

// ---------------------------------------------------------------------------
// GET /embeddings — Empty stub
// ---------------------------------------------------------------------------

pub async fn get_embeddings() -> impl IntoResponse {
    Json(json!([]))
}

// ---------------------------------------------------------------------------
// GET /models — list model folder types
// ---------------------------------------------------------------------------

pub async fn get_models(State(state): State<AppState>) -> impl IntoResponse {
    Json(serde_json::to_value(state.folder_paths.folder_names()).unwrap())
}

// ---------------------------------------------------------------------------
// GET /models/{folder} — list files in a model folder
// ---------------------------------------------------------------------------

pub async fn get_models_by_folder(
    State(state): State<AppState>,
    Path(folder): Path<String>,
) -> impl IntoResponse {
    let paths = state.folder_paths.get_folder_paths(&folder);
    let mut files = Vec::new();
    for p in paths {
        if let Ok(entries) = std::fs::read_dir(p) {
            for entry in entries.flatten() {
                if entry.path().is_file() {
                    files.push(entry.file_name().to_string_lossy().to_string());
                }
            }
        }
    }
    Json(serde_json::to_value(files).unwrap())
}

// ---------------------------------------------------------------------------
// GET /features — Engine capabilities
// ---------------------------------------------------------------------------

pub async fn get_features() -> impl IntoResponse {
    Json(json!({
        "engine": "comfyui-turbo",
        "version": env!("CARGO_PKG_VERSION"),
        "capabilities": {
            "native_samplers": true,
            "zig_simd_kernels": true,
            "wasm_plugins": true,
            "julia_acceleration": true,
            "python_compat": true,
            "kernel_fusion": true,
            "int8_quantization": true,
            "fp16_quantization": true,
            "multi_device": true,
            "memory_monitor": true,
            "graph_optimization": true
        }
    }))
}

// ---------------------------------------------------------------------------
// GET /extensions — List frontend extensions (JS files)
// ---------------------------------------------------------------------------

pub async fn get_extensions() -> impl IntoResponse {
    Json(json!([]))
}

// ---------------------------------------------------------------------------
// GET /settings — User settings
// ---------------------------------------------------------------------------

pub async fn get_settings() -> impl IntoResponse {
    Json(json!({}))
}

// POST /settings — Save user settings
pub async fn post_settings() -> impl IntoResponse {
    StatusCode::OK
}

// GET /settings/{id} — Get single setting
pub async fn get_setting_by_id(Path(_id): Path<String>) -> impl IntoResponse {
    Json(json!(null))
}

// ---------------------------------------------------------------------------
// GET /userdata/* — User data files
// ---------------------------------------------------------------------------

pub async fn get_userdata(Path(_path): Path<String>) -> impl IntoResponse {
    // Return empty JSON for known config files, 404 for others
    if _path.ends_with(".json") {
        return (StatusCode::OK, Json(json!({}))).into_response();
    }
    StatusCode::NOT_FOUND.into_response()
}

/// GET /userdata?dir=workflows&recurse=true — list user files in directory
pub async fn get_userdata_query() -> impl IntoResponse {
    Json(json!([]))
}

pub async fn post_userdata(Path(_path): Path<String>) -> impl IntoResponse {
    StatusCode::OK
}

// ---------------------------------------------------------------------------
// GET /internal/logs — Internal logs
// ---------------------------------------------------------------------------

pub async fn get_internal_logs() -> impl IntoResponse {
    Json(json!([]))
}

// ---------------------------------------------------------------------------
// GET /internal/files — Internal file listing
// ---------------------------------------------------------------------------

pub async fn get_internal_files() -> impl IntoResponse {
    Json(json!([]))
}

// ---------------------------------------------------------------------------
// POST /free — Free memory
// ---------------------------------------------------------------------------

pub async fn post_free() -> impl IntoResponse {
    StatusCode::OK
}

// ---------------------------------------------------------------------------
// GET /users — User list (required by frontend bootstrap)
// ---------------------------------------------------------------------------

pub async fn get_users() -> impl IntoResponse {
    Json(json!([]))
}

// ---------------------------------------------------------------------------
// GET /workflow_templates — Workflow template list
// ---------------------------------------------------------------------------

pub async fn get_workflow_templates() -> impl IntoResponse {
    Json(json!([]))
}

// ---------------------------------------------------------------------------
// GET /i18n — Internationalization strings
// ---------------------------------------------------------------------------

pub async fn get_i18n() -> impl IntoResponse {
    Json(json!({}))
}

// ---------------------------------------------------------------------------
// GET /jobs — Job queue status
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct JobsQuery {
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default = "default_jobs_limit")]
    pub limit: usize,
    #[serde(default)]
    pub offset: usize,
}
fn default_jobs_limit() -> usize { 200 }

pub async fn get_jobs(Query(_q): Query<JobsQuery>) -> impl IntoResponse {
    Json(json!([]))
}

// ---------------------------------------------------------------------------
// GET /global_subgraphs — Global subgraph definitions
// ---------------------------------------------------------------------------

pub async fn get_global_subgraphs() -> impl IntoResponse {
    Json(json!([]))
}

// ---------------------------------------------------------------------------
// GET /experiment/models — Experimental model endpoint
// ---------------------------------------------------------------------------

pub async fn get_experiment_models() -> impl IntoResponse {
    Json(json!([]))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn format_history_entry(entry: &HistoryEntry) -> Value {
    json!({
        "prompt": [0, entry.prompt_id, entry.prompt, {}, []],
        "outputs": entry.outputs,
        "status": {
            "status_str": entry.status.status_str,
            "completed": entry.status.completed,
        }
    })
}
