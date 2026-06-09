use tauri::{AppHandle, Manager, Emitter, State};
use crate::state::AppState;
use crate::indexer;
use crate::commands::models::IndexProgress;

#[tauri::command]
pub async fn index_directory(
    path: String,
    app: AppHandle,
) -> Result<String, String> {
    let path_clone = path.clone();
    let app_clone = app.clone();

    tokio::task::spawn_blocking(move || {
        let state_ref = app_clone.state::<AppState>();
        let mut session = state_ref.image_session.lock().unwrap();
        let conn = state_ref.conn.lock().unwrap();

        // Read thumbnail size from settings
        let thumbnail_size: u32 = conn.query_row(
            "SELECT value FROM settings WHERE key = 'thumbnail_size'",
            [],
            |r| r.get::<_, String>(0),
        ).ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(224);

        indexer::index_directory(&path_clone, &mut session, &conn, thumbnail_size, |done, total, estimated_remaining_ms| {
            let _ = app_clone.emit("index-progress", IndexProgress { done, total, estimated_remaining_ms });
            let hwnd = app_clone.state::<AppState>().hwnd_get();
            crate::taskbar::set_progress(hwnd, done, total);
        });

        crate::taskbar::clear_progress(app_clone.state::<AppState>().hwnd_get());

        let _ = app_clone.emit("index-complete", ());
    })
    .await
    .map_err(|e: tokio::task::JoinError| e.to_string())?;

    Ok(format!("Indexing complete for {}", path))
}

#[tauri::command]
pub fn get_indexed_count(state: State<'_, AppState>) -> Result<i64, String> {
    let conn = state.conn.lock().map_err(|e: std::sync::PoisonError<_>| e.to_string())?;
    let model_family = conn.query_row(
        "SELECT value FROM settings WHERE key = 'model_family'",
        [], |r| r.get::<_, String>(0)
    ).unwrap_or_else(|_| "CLIP ViT-B-16".to_string());
    let count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM embeddings WHERE model_family = ?1",
            rusqlite::params![model_family],
            |r: &rusqlite::Row| r.get(0)
        )
        .map_err(|e: rusqlite::Error| e.to_string())?;
    Ok(count)
}

#[tauri::command]
pub fn unindex_folder(path: String, state: State<AppState>) -> Result<i64, String> {
    let conn = state.conn.lock().map_err(|e| e.to_string())?;
 
    let like_pattern = format!("{}%", path.replace('\\', "/"));
    let like_pattern_win = format!("{}%", path);
 
    let deleted: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM images WHERE path LIKE ?1 OR path LIKE ?2",
            rusqlite::params![like_pattern, like_pattern_win],
            |r| r.get(0),
        )
        .unwrap_or(0);
 
    conn.execute(
        "DELETE FROM images WHERE path LIKE ?1 OR path LIKE ?2",
        rusqlite::params![like_pattern, like_pattern_win],
    )
    .map_err(|e| e.to_string())?;
 
    conn.execute(
        "DELETE FROM watched_folders WHERE path = ?1",
        rusqlite::params![path],
    )
    .map_err(|e| e.to_string())?;
 
    Ok(deleted)
}