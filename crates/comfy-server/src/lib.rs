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
    build_router(state, frontend_dir)
}

/// Build the axum Router with optional frontend serving.
/// Templates are NOT served — use Python server for templates.
fn build_router(state: AppState, frontend_dir: Option<&Path>) -> Router {
    let api_routes = Router::new()
        .route("/prompt", get(routes::get_prompt).post(routes::post_prompt))
        .route("/queue", get(routes::get_queue).post(routes::post_queue))
        .route("/interrupt", post(routes::post_interrupt))
        .route("/history", get(routes::get_history).post(routes::post_history))
        .route("/history/{prompt_id}", get(routes::get_history_by_id))
        .route("/system_stats", get(routes::get_system_stats))
        .route("/object_info", get(routes::get_object_info))
        .route("/embeddings", get(routes::get_embeddings))
        .route("/models", get(routes::get_models))
        .route("/models/{folder}", get(routes::get_models_by_folder))
        .route("/view", get(upload::view_image))
        .route("/upload/image", post(upload::upload_image))
        .route("/features", get(routes::get_features))
        .route("/extensions", get(routes::get_extensions))
        .route("/settings", get(routes::get_settings).post(routes::post_settings))
        .route("/settings/{id}", get(routes::get_setting_by_id))
        .route("/userdata", get(routes::get_userdata_query))
        .route("/userdata/{*path}", get(routes::get_userdata).post(routes::post_userdata))
        .route("/internal/logs", get(routes::get_internal_logs))
        .route("/internal/files", get(routes::get_internal_files))
        .route("/free", post(routes::post_free))
        .route("/users", get(routes::get_users))
        .route("/workflow_templates", get(routes::get_workflow_templates))
        .route("/i18n", get(routes::get_i18n))
        .route("/jobs", get(routes::get_jobs))
        .route("/global_subgraphs", get(routes::get_global_subgraphs))
        .route("/experiment/models", get(routes::get_experiment_models));

    let mut app_router = Router::new()
        .merge(api_routes.clone())
        .nest("/api", api_routes)
        .route("/ws", get(ws::ws_handler));

    // Serve frontend static files as fallback
    if let Some(dir) = frontend_dir {
        if dir.exists() {
            tracing::info!("Serving frontend from: {}", dir.display());
            app_router = app_router.fallback_service(ServeDir::new(dir));
        }
    }

    app_router
        .layer(CorsLayer::permissive())
        .with_state(state)
}
