use crate::state::AppState;
use crate::search;
use serde_json::{json, Value};
use tauri::{AppHandle, State, Manager, Emitter};
use crate::commands::models::*;

#[tauri::command]
pub fn open_path(path: String) -> Result<(), String> {
    #[cfg(target_os = "windows")]
    std::process::Command::new("explorer")
        .arg(&path)
        .spawn()
        .map_err(|e: std::io::Error| e.to_string())?;
    Ok(())
}

#[tauri::command]
pub fn find_similar(path: String, state: State<AppState>) -> Result<Vec<(String, f32, f32)>, String> {
    let conn = state.conn.lock().map_err(|e| e.to_string())?;
    let results = search::find_similar(&path, &conn, 20);
    Ok(results)
}

#[tauri::command]
pub fn mark_feedback(path: String, query: String, signal: i32, state: State<AppState>) -> Result<(), String> {
    let conn = state.conn.lock().map_err(|e| e.to_string())?;
    conn.execute(
        "INSERT INTO feedback (path, query, signal) VALUES (?1, ?2, ?3)
         ON CONFLICT(path, query) DO UPDATE SET signal = ?3",
        rusqlite::params![path, query, signal],
    ).map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
pub fn remove_feedback(path: String, query: String, state: State<AppState>) -> Result<(), String> {
    let conn = state.conn.lock().map_err(|e| e.to_string())?;
    conn.execute(
        "DELETE FROM feedback WHERE path = ?1 AND query = ?2",
        rusqlite::params![path, query],
    ).map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
pub async fn reload_model(app: AppHandle) -> Result<(), String> {
    tokio::task::spawn_blocking(move || -> Result<(), String> {
        let state = app.state::<AppState>();

        // Load new model paths from settings
        let (image_path, text_path, model_family) = {
            let conn = state.conn.lock().map_err(|e: std::sync::PoisonError<_>| e.to_string())?;
            let image = conn.query_row(
                "SELECT value FROM settings WHERE key = 'image_model'",
                [], |r| r.get::<_, String>(0)
            ).map_err(|e| e.to_string())?;
            let text = conn.query_row(
                "SELECT value FROM settings WHERE key = 'text_model'",
                [], |r| r.get::<_, String>(0)
            ).map_err(|e| e.to_string())?;
            let family = conn.query_row(
                "SELECT value FROM settings WHERE key = 'model_family'",
                [], |r| r.get::<_, String>(0)
            ).map_err(|e| e.to_string())?;
            (image, text, family)
        };

        // Load new sessions
        let sep = std::path::MAIN_SEPARATOR_STR;

        let new_image = ort::session::Session::builder()
            .map_err(|e| e.to_string())?
            .commit_from_file(state.models_dir.join(image_path.replace("/", sep)))
            .map_err(|e| e.to_string())?;

        let new_text = ort::session::Session::builder()
            .map_err(|e| e.to_string())?
            .commit_from_file(state.models_dir.join(text_path.replace("/", sep)))
            .map_err(|e| e.to_string())?;

        // Swap sessions
        *state.image_session.lock().map_err(|e: std::sync::PoisonError<_>| e.to_string())? = new_image;
        *state.text_session.lock().map_err(|e: std::sync::PoisonError<_>| e.to_string())? = new_text;

        // Check if embeddings already exist for this model family
        let existing_count: i64 = {
            let conn = state.conn.lock().map_err(|e: std::sync::PoisonError<_>| e.to_string())?;
            conn.query_row(
                "SELECT COUNT(*) FROM embeddings WHERE model_family = ?1",
                rusqlite::params![model_family],
                |r| r.get(0)
            ).unwrap_or(0)
        };

        if existing_count > 0 {
            eprintln!("[reload_model] {} embeddings found for '{}', skipping re-index", existing_count, model_family);
            let _ = app.emit("index-complete", ());
            return Ok(());
        }

        // No embeddings for this model yet - re-embed all images without touching thumbnails
        let norm = crate::indexer::resize::NormConfig::from_model_family(&model_family);

        let paths: Vec<String> = {
            let conn = state.conn.lock().map_err(|e: std::sync::PoisonError<_>| e.to_string())?;
            let mut stmt = conn.prepare("SELECT path FROM images")
                .map_err(|e| e.to_string())?;
            let x: Vec<String> = stmt.query_map([], |row| row.get(0))
                .map_err(|e| e.to_string())?
                .filter_map(|r| r.ok())
                .collect();
            x
        };

        let total = paths.len();
        let reindex_start = std::time::Instant::now();

        let _ = app.emit("reindex-started", total);
        let _ = app.emit("index-progress", IndexProgress { done: 0, total, estimated_remaining_ms: None });

        for (done, path) in paths.iter().enumerate() {
            if let Some(pixels) = crate::indexer::decode::load_image_for_embedding(
                std::path::Path::new(path), norm
            ) {
                let embedding = crate::indexer::embed::run_batch(
                    &mut *state.image_session.lock().map_err(|e: std::sync::PoisonError<_>| e.to_string())?,
                    &pixels,
                    1
                );
                {
                    let conn = state.conn.lock().map_err(|e: std::sync::PoisonError<_>| e.to_string())?;
                    conn.execute(
                        "INSERT OR REPLACE INTO embeddings (image_id, model_family, embedding)
                         VALUES ((SELECT id FROM images WHERE path = ?1), ?2, ?3)",
                        rusqlite::params![path, model_family, embedding],
                    ).map_err(|e| e.to_string())?;
                }

                let done_count = done + 1;
                let estimate = if done_count >= 5 {
                    let elapsed_ms = reindex_start.elapsed().as_millis() as i64;
                    let ms_per_item = elapsed_ms / done_count as i64;
                    Some(ms_per_item * (total - done_count) as i64)
                } else {
                    None
                };

                let _ = app.emit("index-progress", IndexProgress {
                    done: done_count,
                    total,
                    estimated_remaining_ms: estimate,
                });
            }
        }

        let _ = app.emit("index-complete", ());
        Ok(())
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
pub fn get_available_models() -> Result<Vec<Value>, String> {
    let models = vec![
        json!({
            "id": "CLIP ViT-B-16",
            "name": "CLIP ViT-B-16",
            "description": "Compact model suitable for general use.",
            "image_model": "CLIP ViT-B-16/image_encoder.onnx",
            "text_model": "CLIP ViT-B-16/text_encoder.onnx"
        }),
        json!({
            "id": "MetaCLIP",
            "name": "MetaCLIP ViT-B-16",
            "description": "Better search quality than CLIP, trained on cleaner curated data.",
            "image_model": "MetaCLIP ViT-B-16/image_encoder.onnx",
            "text_model": "MetaCLIP ViT-B-16/text_encoder.onnx"
        }),
        json!({
            "id": "MobileCLIP",
            "name": "MobileCLIP S2",
            "description": "Fast, efficient model with strong accuracy. Best for large libraries.",
            "image_model": "mobileclip-s2/mobileclip_image_encoder.onnx",
            "text_model": "mobileclip-s2/mobileclip_text_encoder.onnx"
        }),
    ];
    Ok(models)
}