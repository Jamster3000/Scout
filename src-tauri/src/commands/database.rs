use tauri::State;
use crate::state::AppState;
use crate::time_block;

#[tauri::command]
pub fn delete_from_db(path: String, state: State<'_, AppState>) -> Result<(), String> {
    let conn = state.conn.lock().map_err(|e: std::sync::PoisonError<_>| e.to_string())?;
    conn.execute("DELETE FROM images WHERE path = ?1", rusqlite::params![path])
        .map_err(|e: rusqlite::Error| e.to_string())?;
    Ok(())
}

#[tauri::command]
pub fn delete_from_system(path: String, state: State<'_, AppState>) -> Result<(), String> {
    let conn = state.conn.lock().map_err(|e: std::sync::PoisonError<_>| e.to_string())?;
    conn.execute("DELETE FROM images WHERE path = ?1", rusqlite::params![path])
        .map_err(|e: rusqlite::Error| e.to_string())?;
    std::fs::remove_file(&path).map_err(|e: std::io::Error| e.to_string())?;
    Ok(())
}

#[tauri::command]
pub fn delete_folder_from_system(path: String, state: State<AppState>) -> Result<i64, String> {
    let conn = state.conn.lock().map_err(|e| e.to_string())?;
 
    let like_pattern = format!("{}%", path.replace('\\', "/"));
    let like_pattern_win = format!("{}%", path);
 
    // Collect paths to delete from disk
    let mut stmt = conn
        .prepare(
            "SELECT path FROM images WHERE path LIKE ?1 OR path LIKE ?2",
        )
        .map_err(|e| e.to_string())?;
 
    let file_paths: Vec<String> = stmt
        .query_map(rusqlite::params![like_pattern, like_pattern_win], |r| {
            r.get(0)
        })
        .map_err(|e| e.to_string())?
        .filter_map(|r| r.ok())
        .collect();
 
    let count = file_paths.len() as i64;
 
    // Delete files from disk
    for file_path in &file_paths {
        if let Err(e) = std::fs::remove_file(file_path) {
            eprintln!("[delete_folder_from_system] failed to delete {:?}: {}", file_path, e);
        }
    }
 
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
 
    Ok(count)
}

#[tauri::command]
pub fn clear_database(state: State<AppState>) -> Result<(), String> {
    let conn = state.conn.lock().map_err(|e| e.to_string())?;

    time_block!("clear_database", {
        conn.execute_batch("
            PRAGMA foreign_keys = OFF;
            BEGIN TRANSACTION;
            DELETE FROM collection_items;
            DELETE FROM embeddings;
            DELETE FROM images;
            DELETE FROM feedback;
            DELETE FROM watched_folders;
            COMMIT;
            PRAGMA foreign_keys = ON;
        ").map_err(|e| e.to_string())?;
    });

    let _ = conn.execute("VACUUM;", []);

    Ok(())
}
