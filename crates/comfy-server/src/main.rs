use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let app = comfy_server::build_app();

    let port = std::env::var("COMFY_PORT").unwrap_or_else(|_| "8188".to_string());
    let addr = format!("127.0.0.1:{port}");
    tracing::info!("ComfyUI Turbo server starting on http://{addr}");

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("failed to bind to address");

    axum::serve(listener, app)
        .await
        .expect("server error");
}
