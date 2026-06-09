use crate::state::AppState;
use tauri::State;
use crate::commands::models::FolderInfo;

#[tauri::command]  
pub fn show_in_folder(path: String) -> Result<(), String> {
    #[cfg(target_os = "windows")]
    std::process::Command::new("explorer")
        .args(["/select,", &path])
        .spawn()
        .map_err(|e: std::io::Error| e.to_string())?;
    Ok(())
}

#[tauri::command]
pub fn save_folder(path: String, auto_index: bool, state: State<AppState>) -> Result<(), String> {
    let conn = state.conn.lock().map_err(|e| e.to_string())?;
    conn.execute(
        "INSERT INTO watched_folders (path, auto_index) VALUES (?1, ?2)
         ON CONFLICT(path) DO UPDATE SET auto_index = ?2",
        rusqlite::params![path, auto_index as i32],
    ).map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
pub fn get_watched_folders(state: State<AppState>) -> Result<Vec<(String, bool)>, String> {
    let conn = state.conn.lock().map_err(|e| e.to_string())?;
    let mut stmt = conn.prepare(
        "SELECT path, auto_index FROM watched_folders"
    ).map_err(|e| e.to_string())?;
    let folders = stmt.query_map([], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, bool>(1)?))
    }).map_err(|e| e.to_string())?
    .filter_map(|r| r.ok())
    .collect();
    Ok(folders)
}

#[tauri::command]
pub fn get_folder_stats(state: State<AppState>) -> Result<Vec<FolderInfo>, String> {
    let conn = state.conn.lock().map_err(|e| e.to_string())?;

    let mut stmt = conn.prepare(
        "SELECT path, auto_index, last_indexed FROM watched_folders ORDER BY path"
    ).map_err(|e| e.to_string())?;

    let folders: Vec<(String, bool, Option<i64>)> = stmt
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, bool>(1)?,
                row.get::<_, Option<i64>>(2)?,
            ))
        })
        .map_err(|e| e.to_string())?
        .filter_map(|r| r.ok())
        .collect();

    let mut result = Vec::with_capacity(folders.len());

    for (path, auto_index, last_indexed) in folders {
        //count all images that are inside this FolderInfo
        let like_pattern = format!("{}%", path.replace('\\', "/"));
        let like_pattern_win = format!("{}%", path);

        let image_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM images WHERE path LIKE ?1 OR path LIKE ?2",
                rusqlite::params![like_pattern, like_pattern_win],
                |r| r.get(0),
            )
            .unwrap_or(0);

        let last_image_path: Option<String> = conn
            .query_row(
                "SELECT path FROM images WHERE (path LIKE ?1 OR path LIKE ?2)
                 ORDER BY indexed_at DESC LIMIT 1",
                 rusqlite::params![like_pattern, like_pattern_win],
                 |r| r.get(0),
            )
            .ok();

        result.push(FolderInfo {
            path,
            auto_index,
            last_indexed,
            image_count,
            last_image_path,
        });
    }

    Ok(result)
}

#[tauri::command]
pub fn set_folder_auto_index(
    path: String,
    auto_index: bool,
    state: State<AppState>,
) -> Result<(), String> {
    let conn = state.conn.lock().map_err(|e| e.to_string())?;
    conn.execute(
        "UPDATE watched_folders SET auto_index = ?1 WHERE path = ?2",
        rusqlite::params![auto_index as i32, path],
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}
