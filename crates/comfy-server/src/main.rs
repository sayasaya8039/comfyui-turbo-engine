use std::path::PathBuf;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let port = std::env::var("COMFY_PORT").unwrap_or_else(|_| "8188".to_string());
    let frontend_dir = std::env::var("COMFY_FRONTEND").ok().map(PathBuf::from);

    // Build template asset map from Python venv (if available)
    let template_map = std::env::var("COMFY_VENV").ok().and_then(|venv| {
        comfy_server::templates::build_asset_map_from_venv(&venv)
    });

    let app = comfy_server::build_app_full(
        frontend_dir.as_deref(),
        template_map,
    );

    let addr = format!("127.0.0.1:{port}");
    tracing::info!("ComfyUI Turbo server starting on http://{addr}");

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .expect("failed to bind to address");

    // Graceful shutdown on Ctrl+C or parent process termination
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .expect("server error");

    tracing::info!("Server shut down gracefully");
}

async fn shutdown_signal() {
    let ctrl_c = tokio::signal::ctrl_c();
    ctrl_c.await.expect("failed to listen for ctrl+c");
    tracing::info!("Received shutdown signal");
}
