pub mod history;
pub mod queue;
pub mod routes;
pub mod state;
pub mod upload;
pub mod ws;

use std::path::{Path, PathBuf};

use axum::{
    routing::{get, post},
    Router,
};
use comfy_core::NodeRegistry;
use tower_http::cors::CorsLayer;
use tower_http::services::ServeDir;

use crate::state::AppState;

/// Build app for integration tests (no frontend).
pub fn build_app() -> Router {
    build_app_with_frontend(None)
}

/// Build app with optional frontend static file serving.
pub fn build_app_with_frontend(frontend_dir: Option<&Path>) -> Router {
    let mut registry = NodeRegistry::new();
    comfy_nodes::register_all_nodes(&mut registry);

    let state = AppState::new(registry, PathBuf::from("."));
    build_router_with_frontend(state, frontend_dir)
}

/// Build the axum Router (no frontend serving).
pub fn build_router(state: AppState) -> Router {
    build_router_with_frontend(state, None)
}

/// Build the axum Router with optional frontend static file serving.
pub fn build_router_with_frontend(state: AppState, frontend_dir: Option<&Path>) -> Router {
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
        // Models
        .route("/models", get(routes::get_models))
        .route("/models/{folder}", get(routes::get_models_by_folder))
        // Upload/View stubs
        .route("/view", get(upload::view_image))
        .route("/upload/image", post(upload::upload_image))
        // Engine features
        .route("/features", get(routes::get_features))
        // Extensions (frontend JS plugins)
        .route("/extensions", get(routes::get_extensions))
        // Settings
        .route("/settings", get(routes::get_settings).post(routes::post_settings))
        .route("/settings/{id}", get(routes::get_setting_by_id))
        // User data
        .route("/userdata/{*path}", get(routes::get_userdata).post(routes::post_userdata))
        // Internal
        .route("/internal/logs", get(routes::get_internal_logs))
        .route("/internal/files", get(routes::get_internal_files))
        // Free memory
        .route("/free", post(routes::post_free))
        // Users
        .route("/users", get(routes::get_users));

    let mut app_router = Router::new()
        // Direct endpoints (e.g. /prompt)
        .merge(api_routes.clone())
        // /api prefixed endpoints (e.g. /api/prompt)
        .nest("/api", api_routes)
        // WebSocket
        .route("/ws", get(ws::ws_handler));

    // Serve frontend static files as fallback (if directory provided)
    // Must be added BEFORE .with_state() to keep Router<AppState> type
    if let Some(dir) = frontend_dir {
        if dir.exists() {
            tracing::info!("Serving frontend from: {}", dir.display());
            app_router = app_router.fallback_service(ServeDir::new(dir));
        } else {
            tracing::warn!("Frontend directory not found: {}", dir.display());
        }
    }

    app_router
        .layer(CorsLayer::permissive())
        .with_state(state)
}
