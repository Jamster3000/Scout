use crate::state::AppState;
use crate::search;
use tauri::State;

#[tauri::command]
pub fn search(query: String, state: State<AppState>) -> Result<Vec<(String, f32, f32)>, String> {
    let mut session = state.text_session.lock().unwrap();
    let conn = state.conn.lock().map_err(|e| e.to_string())?;
    let results = search::search(&query, &state.tokenizer, &mut session, &conn);
    Ok(results)
}