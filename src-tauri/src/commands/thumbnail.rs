use tauri::{State, AppHandle, Emitter, Manager};
use crate::state::AppState;
use crate::commands::models::IndexProgress;
use crate::indexer;

#[tauri::command]
pub fn get_thumbnails(
    paths: Vec<String>,
    state: State<AppState>,
) -> Result<Vec<(String, Vec<u8>, Option<Vec<u8>>)>, String> {
    let conn = state.conn.lock().map_err(|e| e.to_string())?;
    let results = crate::search::get_thumbnails(&paths, &conn);
    Ok(results)
}

#[tauri::command]
pub async fn regenerate_thumbnails(size: u32, app: AppHandle) -> Result<(), String> {
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    tokio::task::spawn_blocking(move || {
        let state = app.state::<AppState>();
        let paths: Vec<String> = {
            let conn = state.conn.lock().unwrap();
            let mut stmt = conn.prepare("SELECT path FROM images WHERE embedding IS NOT NULL").unwrap();
            stmt.query_map([], |r| r.get(0)).unwrap().filter_map(|r| r.ok()).collect()
        };

        let total = paths.len();
        let done = AtomicUsize::new(0);

        let thumbs: Vec<(String, Vec<u8>)> = paths.par_iter()
            .filter_map(|path| {
                let thumb = indexer::generate_thumbnail(path, size)?;
                let n = done.fetch_add(1, Ordering::Relaxed) + 1;
                let _ = app.emit("index-progress", IndexProgress { done: n, total, estimated_remaining_ms: None });
                Some((path.clone(), thumb))
            })
            .collect();

        // Write all at once
        let conn = state.conn.lock().unwrap();
        for (path, bytes) in thumbs {
            conn.execute(
                "UPDATE images SET thumbnail = ?1 WHERE path = ?2",
                rusqlite::params![bytes, path],
            ).ok();
        }
        let _ = app.emit("index-complete", ());
    })
    .await
    .map_err(|e| e.to_string())?;
    Ok(())
}
