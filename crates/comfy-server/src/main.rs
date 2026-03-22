mod history;
mod queue;
mod routes;
mod state;
mod upload;
mod ws;

use std::path::PathBuf;

use axum::{
    routing::{get, post},
    Router,
};
use comfy_core::NodeRegistry;
use tower_http::cors::CorsLayer;
use tracing_subscriber::EnvFilter;

use crate::state::AppState;

#[tokio::main]
async fn main() {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    // Register all standard nodes
    let mut registry = NodeRegistry::new();
    comfy_nodes::register_all_nodes(&mut registry);

    let base_path = PathBuf::from(".");
    let state = AppState::new(registry, base_path);

    let app = build_router(state);

    let addr = "127.0.0.1:8188";
    tracing::info!("ComfyUI Turbo server starting on http://{addr}");

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("failed to bind to address");

    axum::serve(listener, app)
        .await
        .expect("server error");
}

/// Build the axum Router with all routes.
///
/// Each endpoint is registered under both `/endpoint` and `/api/endpoint`
/// to match ComfyUI's dual-prefix convention.
fn build_router(state: AppState) -> Router {
    let api_routes = Router::new()
        // Prompt
        .route("/prompt", get(routes::get_prompt).post(routes::post_prompt))
        // Queue
        .route("/queue", get(routes::get_queue).post(routes::post_queue))
        // Interrupt
        .route("/interrupt", post(routes::post_interrupt))
        // History
        .route(
            "/history",
            get(routes::get_history).post(routes::post_history),
        )
        .route("/history/{prompt_id}", get(routes::get_history_by_id))
        // System stats
        .route("/system_stats", get(routes::get_system_stats))
        // Object info (node definitions)
        .route("/object_info", get(routes::get_object_info))
        // Embeddings stub
        .route("/embeddings", get(routes::get_embeddings))
        // Upload/View stubs
        .route("/view", get(upload::view_image))
        .route("/upload/image", post(upload::upload_image));

    Router::new()
        // Direct endpoints (e.g. /prompt)
        .merge(api_routes.clone())
        // /api prefixed endpoints (e.g. /api/prompt)
        .nest("/api", api_routes)
        // WebSocket
        .route("/ws", get(ws::ws_handler))
        // CORS
        .layer(CorsLayer::permissive())
        .with_state(state)
}
