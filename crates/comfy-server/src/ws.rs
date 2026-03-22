use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        Query, State,
    },
    response::IntoResponse,
};
use serde::Deserialize;

use crate::state::AppState;

#[derive(Debug, Deserialize)]
pub struct WsQuery {
    #[serde(rename = "clientId")]
    pub client_id: Option<String>,
}

/// WebSocket upgrade handler.
///
/// Accepts an optional `?clientId=<uuid>` query parameter.
/// Sends an initial status message with the current queue remaining count.
pub async fn ws_handler(
    ws: WebSocketUpgrade,
    Query(query): Query<WsQuery>,
    State(state): State<AppState>,
) -> impl IntoResponse {
    let client_id = query.client_id.unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
    ws.on_upgrade(move |socket| handle_socket(socket, client_id, state))
}

async fn handle_socket(mut socket: WebSocket, client_id: String, state: AppState) {
    let queue_remaining = {
        let queue = state.queue.lock().await;
        queue.remaining()
    };

    // Send initial status message matching ComfyUI format
    let status_msg = serde_json::json!({
        "type": "status",
        "data": {
            "status": {
                "exec_info": {
                    "queue_remaining": queue_remaining
                }
            },
            "sid": client_id
        }
    });

    if let Ok(msg_str) = serde_json::to_string(&status_msg) {
        let _ = socket.send(Message::Text(msg_str.into())).await;
    }

    // Keep connection alive and handle incoming messages
    while let Some(Ok(msg)) = socket.recv().await {
        match msg {
            Message::Text(text) => {
                // Handle feature_flags or other client messages
                if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&text) {
                    if parsed.get("type").and_then(|t| t.as_str()) == Some("feature_flags") {
                        // Acknowledge feature flags (no-op for now)
                        tracing::debug!("Received feature_flags from client {client_id}");
                    }
                }
            }
            Message::Close(_) => break,
            _ => {}
        }
    }

    tracing::debug!("WebSocket client {client_id} disconnected");
}
