use axum::{
    extract::{Path, State},
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
    Json(json!({
        "system": {
            "os": std::env::consts::OS,
            "comfyui_version": "0.8.24-turbo",
            "python_version": "N/A (native)",
            "pytorch_version": "N/A (native)",
            "embedded_python": false,
            "argv": []
        },
        "devices": [
            {
                "name": "cpu",
                "type": "cpu",
                "index": 0,
                "vram_total": 0,
                "vram_free": 0,
                "torch_vram_total": 0,
                "torch_vram_free": 0
            }
        ]
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
