//! E2E API compatibility tests for ComfyUI Turbo server.
//!
//! These tests exercise the axum Router directly via `tower::ServiceExt::oneshot`,
//! so no actual HTTP server is started. Each test constructs a fresh `Router`
//! through `comfy_server::build_app()` and sends a single request.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt; // for collect()
use tower::ServiceExt; // for oneshot()

/// Helper: send a request to a fresh app and return (status, body_bytes).
async fn send(req: Request<Body>) -> (StatusCode, serde_json::Value) {
    let app = comfy_server::build_app();
    let resp = app.oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value =
        serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
    (status, json)
}

// -----------------------------------------------------------------------
// 1. GET /prompt — queue info
// -----------------------------------------------------------------------

#[tokio::test]
async fn test_get_prompt_returns_queue_info() {
    let (status, json) = send(
        Request::builder()
            .uri("/prompt")
            .body(Body::empty())
            .unwrap(),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert!(
        json["exec_info"]["queue_remaining"].is_number(),
        "expected exec_info.queue_remaining to be a number, got: {json}"
    );
}

// -----------------------------------------------------------------------
// 2. POST /prompt — submit a workflow
// -----------------------------------------------------------------------

#[tokio::test]
async fn test_post_prompt_returns_prompt_id() {
    let body = serde_json::json!({
        "prompt": {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": { "ckpt_name": "test.safetensors" }
            }
        }
    });

    let (status, json) = send(
        Request::builder()
            .method("POST")
            .uri("/prompt")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap(),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert!(json["prompt_id"].is_string(), "prompt_id should be string");
    assert!(json["number"].is_number(), "number should be number");
    assert!(
        json["node_errors"].is_object(),
        "node_errors should be object"
    );
}

// -----------------------------------------------------------------------
// 3. GET /queue — running + pending arrays
// -----------------------------------------------------------------------

#[tokio::test]
async fn test_get_queue_returns_arrays() {
    let (status, json) = send(
        Request::builder()
            .uri("/queue")
            .body(Body::empty())
            .unwrap(),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert!(
        json["queue_running"].is_array(),
        "queue_running should be array"
    );
    assert!(
        json["queue_pending"].is_array(),
        "queue_pending should be array"
    );
}

// -----------------------------------------------------------------------
// 4. POST /queue {"clear": true} — clear the queue
// -----------------------------------------------------------------------

#[tokio::test]
async fn test_post_queue_clear() {
    let body = serde_json::json!({"clear": true});

    let app = comfy_server::build_app();
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/queue")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
}

// -----------------------------------------------------------------------
// 5. GET /system_stats — system information
// -----------------------------------------------------------------------

#[tokio::test]
async fn test_get_system_stats() {
    let (status, json) = send(
        Request::builder()
            .uri("/system_stats")
            .body(Body::empty())
            .unwrap(),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert!(json["system"]["os"].is_string(), "system.os should exist");
    assert!(
        json["system"]["comfyui_version"].is_string(),
        "comfyui_version should exist"
    );
    assert!(json["devices"].is_array(), "devices should be array");

    // Check at least one device entry
    let devices = json["devices"].as_array().unwrap();
    assert!(!devices.is_empty(), "should have at least one device");
    assert!(devices[0]["name"].is_string());
    assert!(devices[0]["type"].is_string());
}

// -----------------------------------------------------------------------
// 6. GET /object_info — registered node definitions
// -----------------------------------------------------------------------

#[tokio::test]
async fn test_get_object_info_contains_standard_nodes() {
    let (status, json) = send(
        Request::builder()
            .uri("/object_info")
            .body(Body::empty())
            .unwrap(),
    )
    .await;

    assert_eq!(status, StatusCode::OK);

    // Verify key nodes are registered
    assert!(
        json["CheckpointLoaderSimple"].is_object(),
        "CheckpointLoaderSimple should be in object_info"
    );
    assert!(
        json["KSampler"].is_object(),
        "KSampler should be in object_info"
    );
    assert!(
        json["SaveImage"].is_object(),
        "SaveImage should be in object_info"
    );

    // SaveImage should be marked as output_node
    assert_eq!(
        json["SaveImage"]["output_node"],
        serde_json::json!(true),
        "SaveImage should have output_node=true"
    );

    // Check node structure
    let ckpt = &json["CheckpointLoaderSimple"];
    assert!(ckpt["name"].is_string());
    assert!(ckpt["display_name"].is_string());
    assert!(ckpt["category"].is_string());
}

// -----------------------------------------------------------------------
// 7. GET /embeddings — returns empty array
// -----------------------------------------------------------------------

#[tokio::test]
async fn test_get_embeddings_returns_empty_array() {
    let (status, json) = send(
        Request::builder()
            .uri("/embeddings")
            .body(Body::empty())
            .unwrap(),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert!(json.is_array(), "embeddings should be an array");
    assert_eq!(json.as_array().unwrap().len(), 0, "embeddings should be empty");
}

// -----------------------------------------------------------------------
// 8. POST /interrupt — returns 200
// -----------------------------------------------------------------------

#[tokio::test]
async fn test_post_interrupt() {
    let app = comfy_server::build_app();
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/interrupt")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
}

// -----------------------------------------------------------------------
// 9. GET /history — returns object (empty initially)
// -----------------------------------------------------------------------

#[tokio::test]
async fn test_get_history_empty() {
    let (status, json) = send(
        Request::builder()
            .uri("/history")
            .body(Body::empty())
            .unwrap(),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert!(json.is_object(), "history should be an object");
    assert_eq!(
        json.as_object().unwrap().len(),
        0,
        "history should be empty initially"
    );
}

// -----------------------------------------------------------------------
// 10. POST /history {"clear": true} — returns 200
// -----------------------------------------------------------------------

#[tokio::test]
async fn test_post_history_clear() {
    let body = serde_json::json!({"clear": true});

    let app = comfy_server::build_app();
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/history")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
}

// -----------------------------------------------------------------------
// 11. GET /api/prompt — dual prefix compatibility
// -----------------------------------------------------------------------

#[tokio::test]
async fn test_api_prefix_prompt() {
    let (status, json) = send(
        Request::builder()
            .uri("/api/prompt")
            .body(Body::empty())
            .unwrap(),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert!(
        json["exec_info"]["queue_remaining"].is_number(),
        "GET /api/prompt should return queue_remaining"
    );
}

// -----------------------------------------------------------------------
// 12. GET /api/system_stats — dual prefix for system_stats
// -----------------------------------------------------------------------

#[tokio::test]
async fn test_api_prefix_system_stats() {
    let (status, json) = send(
        Request::builder()
            .uri("/api/system_stats")
            .body(Body::empty())
            .unwrap(),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert!(json["system"]["os"].is_string());
}

// -----------------------------------------------------------------------
// 12b. GET /system_stats — device details from HardwareDetector
// -----------------------------------------------------------------------

#[tokio::test]
async fn test_system_stats_has_device_details() {
    let (status, json) = send(
        Request::builder()
            .uri("/system_stats")
            .body(Body::empty())
            .unwrap(),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    let devices = json["devices"].as_array().unwrap();
    assert!(!devices.is_empty(), "should have at least one device");

    // At least CPU should be present
    assert!(
        devices.iter().any(|d| d["name"].as_str().unwrap().contains("CPU")),
        "should contain a CPU device"
    );

    // Each device should have required fields
    for d in devices {
        assert!(d["name"].is_string(), "device name should be string");
        assert!(d["type"].is_string(), "device type should be string");
        assert!(d["vram_total"].is_number(), "device vram_total should be number");
        assert!(d["compute_units"].is_number(), "device compute_units should be number");
    }
}

// -----------------------------------------------------------------------
// 13. GET /api/object_info — dual prefix for object_info
// -----------------------------------------------------------------------

#[tokio::test]
async fn test_api_prefix_object_info() {
    let (status, json) = send(
        Request::builder()
            .uri("/api/object_info")
            .body(Body::empty())
            .unwrap(),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert!(
        json["KSampler"].is_object(),
        "GET /api/object_info should return node definitions"
    );
}

// -----------------------------------------------------------------------
// 14. POST /upload/image — stub returns success
// -----------------------------------------------------------------------

#[tokio::test]
async fn test_post_upload_image_stub() {
    let app = comfy_server::build_app();
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/upload/image")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert!(json["type"].is_string(), "upload response should have type");
}

// -----------------------------------------------------------------------
// 15. GET /view — stub returns 404
// -----------------------------------------------------------------------

#[tokio::test]
async fn test_get_view_stub() {
    let app = comfy_server::build_app();
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/view")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

// -----------------------------------------------------------------------
// 16. GET /models — returns folder types
// -----------------------------------------------------------------------

#[tokio::test]
async fn test_get_models_returns_folder_types() {
    let (status, json) = send(
        Request::builder()
            .uri("/models")
            .body(Body::empty())
            .unwrap(),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    let arr = json.as_array().expect("models should be an array");
    let names: Vec<&str> = arr.iter().filter_map(|v| v.as_str()).collect();
    assert!(names.contains(&"checkpoints"), "should contain checkpoints");
    assert!(names.contains(&"loras"), "should contain loras");
    assert!(names.contains(&"vae"), "should contain vae");
    assert!(names.contains(&"embeddings"), "should contain embeddings");
}

// -----------------------------------------------------------------------
// 17. GET /api/models — dual prefix for models
// -----------------------------------------------------------------------

#[tokio::test]
async fn test_api_prefix_models() {
    let (status, json) = send(
        Request::builder()
            .uri("/api/models")
            .body(Body::empty())
            .unwrap(),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    let arr = json.as_array().expect("models should be an array");
    let names: Vec<&str> = arr.iter().filter_map(|v| v.as_str()).collect();
    assert!(names.contains(&"checkpoints"), "/api/models should contain checkpoints");
}

// -----------------------------------------------------------------------
// 18. GET /models/{folder} — returns files list (empty for non-existent dir)
// -----------------------------------------------------------------------

#[tokio::test]
async fn test_get_models_by_folder_returns_array() {
    let (status, json) = send(
        Request::builder()
            .uri("/models/checkpoints")
            .body(Body::empty())
            .unwrap(),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert!(json.is_array(), "/models/checkpoints should return an array");
}

// -----------------------------------------------------------------------
// 19. GET /api/models/{folder} — dual prefix
// -----------------------------------------------------------------------

#[tokio::test]
async fn test_api_prefix_models_by_folder() {
    let (status, json) = send(
        Request::builder()
            .uri("/api/models/loras")
            .body(Body::empty())
            .unwrap(),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert!(json.is_array(), "/api/models/loras should return an array");
}
