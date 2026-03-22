use axum::{
    body::Body,
    extract::{Path, State},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

/// Map of template filename -> absolute file path on disk
pub type TemplateAssetMap = Arc<HashMap<String, PathBuf>>;

/// Build asset map by running Python one-shot from the venv
pub fn build_asset_map_from_venv(venv_path: &str) -> Option<TemplateAssetMap> {
    let python = if cfg!(target_os = "windows") {
        format!("{}/Scripts/python.exe", venv_path)
    } else {
        format!("{}/bin/python", venv_path)
    };

    if !std::path::Path::new(&python).exists() {
        tracing::warn!("Python not found at {python}, templates disabled");
        return None;
    }

    let script = r#"
import json, os, glob
try:
    from comfyui_workflow_templates import iter_templates, get_asset_path
    asset_map = {}
    for t in iter_templates():
        for a in t.assets:
            asset_map[a.filename] = get_asset_path(t.template_id, a.filename)
    # Add index files from media-other bundle
    import comfyui_workflow_templates_media_other
    tdir = os.path.join(os.path.dirname(comfyui_workflow_templates_media_other.__file__), 'templates')
    for f in glob.glob(os.path.join(tdir, 'index*.json')) + glob.glob(os.path.join(tdir, 'fuse_options.json')):
        asset_map[os.path.basename(f)] = f
    print(json.dumps(asset_map))
except Exception as e:
    print(json.dumps({"error": str(e)}))
"#;

    tracing::info!("Building template asset map from venv: {venv_path}");

    let output = std::process::Command::new(&python)
        .args(["-c", script])
        .output()
        .ok()?;

    if !output.status.success() {
        tracing::warn!("Template map generation failed: {}", String::from_utf8_lossy(&output.stderr));
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let map: HashMap<String, String> = serde_json::from_str(&stdout).ok()?;

    if map.contains_key("error") {
        tracing::warn!("Template error: {}", map["error"]);
        return None;
    }

    let asset_map: HashMap<String, PathBuf> = map.into_iter()
        .map(|(k, v)| (k, PathBuf::from(v)))
        .collect();

    tracing::info!("Loaded {} template assets from venv", asset_map.len());
    Some(Arc::new(asset_map))
}

/// Serve a template asset by filename
pub async fn serve_template(
    Path(filename): Path<String>,
    State(map): State<TemplateAssetMap>,
) -> Response {
    let Some(file_path) = map.get(&filename) else {
        return StatusCode::NOT_FOUND.into_response();
    };

    let Ok(data) = tokio::fs::read(file_path).await else {
        return StatusCode::NOT_FOUND.into_response();
    };

    let content_type = match file_path.extension().and_then(|e| e.to_str()) {
        Some("json") => "application/json",
        Some("webp") => "image/webp",
        Some("png") => "image/png",
        Some("jpg" | "jpeg") => "image/jpeg",
        Some("mp3") => "audio/mpeg",
        Some("mp4") => "video/mp4",
        _ => "application/octet-stream",
    };

    Response::builder()
        .header(header::CONTENT_TYPE, content_type)
        .body(Body::from(data))
        .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
}
