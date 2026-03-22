use axum::{
    http::StatusCode,
    response::{IntoResponse, Json},
};

/// Stub handler for GET /view — returns 404 (not yet implemented).
pub async fn view_image() -> impl IntoResponse {
    (StatusCode::NOT_FOUND, Json(serde_json::json!({"error": "not implemented"})))
}

/// Stub handler for POST /upload/image — returns empty success response.
pub async fn upload_image() -> impl IntoResponse {
    (StatusCode::OK, Json(serde_json::json!({"name": "", "subfolder": "", "type": "input"})))
}
