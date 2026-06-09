use tauri::State;
use crate::state::AppState;
use crate::commands::models::Collection;
use crate::commands::models::CollectionItem;

#[tauri::command]
pub fn create_collection(name: String, description: Option<String>, state: State<AppState>) -> Result<i32, String> {
    let conn = state.conn.lock().map_err(|e| e.to_string())?;
    conn.execute(
        "INSERT INTO collections (name, description) VALUES (?1, ?2)",
        rusqlite::params![name, description],
    ).map_err(|e| e.to_string())?;
    
    let collection_id: i32 = conn.query_row(
        "SELECT last_insert_rowid()",
        [],
        |r| r.get(0),
    ).map_err(|e| e.to_string())?;
    
    Ok(collection_id)
}

#[tauri::command]
pub fn get_collections(state: State<AppState>) -> Result<Vec<Collection>, String> {
    let conn = state.conn.lock().map_err(|e| e.to_string())?;
    let mut stmt = conn.prepare(
        "SELECT id, name, description, created_at, updated_at FROM collections ORDER BY updated_at DESC"
    ).map_err(|e| e.to_string())?;
    
    let collections = stmt.query_map([], |row| {
        Ok(Collection {
            id: row.get(0)?,
            name: row.get(1)?,
            description: row.get(2)?,
            created_at: row.get(3)?,
            updated_at: row.get(4)?,
        })
    }).map_err(|e| e.to_string())?
        .filter_map(|r| r.ok())
        .collect();
    
    Ok(collections)
}

#[tauri::command]
pub fn rename_collection(collection_id: i32, new_name: String, state: State<AppState>) -> Result<(), String> {
    let conn = state.conn.lock().map_err(|e| e.to_string())?;
    conn.execute(
        "UPDATE collections SET name = ?1, updated_at = unixepoch() WHERE id = ?2",
        rusqlite::params![new_name, collection_id],
    ).map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
pub fn delete_collection(collection_id: i32, state: State<AppState>) -> Result<(), String> {
    let conn = state.conn.lock().map_err(|e| e.to_string())?;
    conn.execute(
        "DELETE FROM collections WHERE id = ?1",
        rusqlite::params![collection_id],
    ).map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
pub fn add_to_collection(collection_id: i32, image_path: String, state: State<AppState>) -> Result<(), String> {
    let conn = state.conn.lock().map_err(|e| e.to_string())?;
    
    // Get the image_id from the path
    let image_id: i32 = conn.query_row(
        "SELECT id FROM images WHERE path = ?1",
        rusqlite::params![image_path],
        |r| r.get(0),
    ).map_err(|e| e.to_string())?;
    
    // Insert into collection_items
    conn.execute(
        "INSERT INTO collection_items (collection_id, image_id) VALUES (?1, ?2)",
        rusqlite::params![collection_id, image_id],
    ).map_err(|e| e.to_string())?;
    
    Ok(())
}

#[tauri::command]
pub fn remove_from_collection(collection_id: i32, image_path: String, state: State<AppState>) -> Result<(), String> {
    let conn = state.conn.lock().map_err(|e| e.to_string())?;
    
    // Get the image_id from the path
    let image_id: i32 = conn.query_row(
        "SELECT id FROM images WHERE path = ?1",
        rusqlite::params![image_path],
        |r| r.get(0),
    ).map_err(|e| e.to_string())?;
    
    // Delete from collection_items
    conn.execute(
        "DELETE FROM collection_items WHERE collection_id = ?1 AND image_id = ?2",
        rusqlite::params![collection_id, image_id],
    ).map_err(|e| e.to_string())?;
    
    Ok(())
}

#[tauri::command]
pub fn get_collection_items(collection_id: i32, state: State<AppState>) -> Result<Vec<CollectionItem>, String> {
    let conn = state.conn.lock().map_err(|e| e.to_string())?;
    let mut stmt = conn.prepare(
        "SELECT ci.id, ci.collection_id, i.path, i.thumbnail, i.aspect_ratio, ci.added_at 
         FROM collection_items ci
         JOIN images i ON ci.image_id = i.id
         WHERE ci.collection_id = ?1
         ORDER BY ci.added_at DESC"
    ).map_err(|e| e.to_string())?;
    
    let items = stmt.query_map(rusqlite::params![collection_id], |row| {
        Ok(CollectionItem {
            id: row.get(0)?,
            collection_id: row.get(1)?,
            path: row.get(2)?,
            thumbnail: row.get(3)?,
            aspect_ratio: row.get(4)?,
            added_at: row.get(5)?,
        })
    }).map_err(|e| e.to_string())?
        .filter_map(|r| r.ok())
        .collect();
    
    Ok(items)
}

#[tauri::command]
pub fn is_image_in_collection(collection_id: i32, image_path: String, state: State<AppState>) -> Result<bool, String> {
    let conn = state.conn.lock().map_err(|e| e.to_string())?;
    
    let image_id: i32 = conn.query_row(
        "SELECT id FROM images WHERE path = ?1",
        rusqlite::params![image_path],
        |r| r.get(0),
    ).map_err(|e| e.to_string())?;
    
    let count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM collection_items WHERE collection_id = ?1 AND image_id = ?2",
        rusqlite::params![collection_id, image_id],
        |r| r.get(0),
    ).map_err(|e| e.to_string())?;
    
    Ok(count > 0)
}

#[tauri::command]
pub fn get_image_collections(image_path: String, state: State<AppState>) -> Result<Vec<Collection>, String> {
    let conn = state.conn.lock().map_err(|e| e.to_string())?;
    
    let image_id: i32 = conn.query_row(
        "SELECT id FROM images WHERE path = ?1",
        rusqlite::params![image_path],
        |r| r.get(0),
    ).map_err(|e| e.to_string())?;
    
    let mut stmt = conn.prepare(
        "SELECT c.id, c.name, c.description, c.created_at, c.updated_at
         FROM collections c
         JOIN collection_items ci ON c.id = ci.collection_id
         WHERE ci.image_id = ?1
         ORDER BY c.updated_at DESC"
    ).map_err(|e| e.to_string())?;
    
    let collections = stmt.query_map(rusqlite::params![image_id], |row| {
        Ok(Collection {
            id: row.get(0)?,
            name: row.get(1)?,
            description: row.get(2)?,
            created_at: row.get(3)?,
            updated_at: row.get(4)?,
        })
    }).map_err(|e| e.to_string())?
        .filter_map(|r| r.ok())
        .collect();
    
    Ok(collections)
}

#[tauri::command]
pub fn get_collection_count(collection_id: i32, state: State<AppState>) -> Result<i64, String> {
    let conn = state.conn.lock().map_err(|e| e.to_string())?;
    let count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM collection_items WHERE collection_id = ?1",
        rusqlite::params![collection_id],
        |r| r.get(0),
    ).map_err(|e| e.to_string())?;
    Ok(count)
}
