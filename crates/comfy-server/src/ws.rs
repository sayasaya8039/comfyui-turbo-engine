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
/// Forwards progress broadcast messages to the client in real time.
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

    // Subscribe to the progress broadcast channel
    let mut progress_rx = state.progress_tx.subscribe();

    // Keep connection alive, forwarding progress messages and handling client messages
    loop {
        tokio::select! {
            // Forward progress broadcasts to the WebSocket client
            Ok(progress_msg) = progress_rx.recv() => {
                if socket.send(Message::Text(progress_msg.into())).await.is_err() {
                    break;
                }
            }
            // Handle incoming messages from the client
            msg = socket.recv() => {
                match msg {
                    Some(Ok(Message::Text(text))) => {
                        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&text) {
                            if parsed.get("type").and_then(|t| t.as_str()) == Some("feature_flags") {
                                tracing::debug!("Received feature_flags from client {client_id}");
                            }
                        }
                    }
                    Some(Ok(Message::Close(_))) | None => break,
                    _ => {}
                }
            }
        }
    }

    tracing::debug!("WebSocket client {client_id} disconnected");
}
