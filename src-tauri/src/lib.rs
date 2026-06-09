use ort::session::Session;
use std::sync::{Arc, Mutex};
use tauri::Manager;
use tauri::Emitter;
use crate::state::AppState;

mod db;
mod indexer;
mod search;
mod state;
mod commands;
mod watcher;
mod taskbar;
mod performance;
#[macro_use]
mod timing;

pub fn get_setting(conn: &rusqlite::Connection, key: &str, default: &str) -> String {
    conn.query_row(
        "SELECT value FROM settings WHERE key = ?1",
        [key],
        |row| row.get(0),
    )
    .unwrap_or_else(|_| default.to_string())
}

pub fn run() {
    // Enable ONNX Runtime verbose logging
    std::env::set_var("ORT_RUST_LOG", "verbose");
    
    let exe_dir = std::env::current_exe()
        .unwrap().parent().unwrap().to_path_buf();

    //Ensures the paths are correct for build relesae
    let (models_dir, db_path) = if cfg!(debug_assertions) {
        let root_dir = exe_dir.join("..").join("..").join("..");
        (root_dir.join("models"), root_dir.join("scout.db"))
    } else {
        (exe_dir.join("_up_").join("models"), exe_dir.join("scout.db"))
    };

    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_notification::init())
        .setup(move |app| {     
            let conn = time_block!("Initialize database connection", {
                Arc::new(Mutex::new(
                    db::open(db_path.to_str().unwrap()).unwrap()
                ))
            });

            let image_model_path = {
                let db = conn.lock().unwrap();
                get_setting(&db, "image_model", "")
            };

            let text_model_path = {
                let db = conn.lock().unwrap();
                get_setting(&db, "text_model", "")
            };

            let mut providers = Vec::new();

            #[cfg(any(target_os = "windows", target_os = "linux"))]
            {
                // Windows/Linux builds get NVIDIA CUDA acceleration
                providers.push(ort::ep::CUDA::default().build());
            }
            
            #[cfg(target_os = "macos")]
            {
                // Mac builds get Apple Silicon Neural Engine acceleration
                providers.push(ort::ep::CoreML::default().build());
            }

            let image_session = time_block!("Initialize image session", {
                Arc::new(Mutex::new(
                    Session::builder().unwrap()
                        .with_execution_providers(&providers).unwrap()
                        .commit_from_file(models_dir.join(image_model_path.replace("/", std::path::MAIN_SEPARATOR_STR))).unwrap()
                ))
            });

            let text_session = time_block!("Initialize text session", {
                Arc::new(Mutex::new(
                    Session::builder().unwrap()
                        .with_execution_providers(&providers).unwrap()
                        .commit_from_file(models_dir.join(text_model_path.replace("/", std::path::MAIN_SEPARATOR_STR))).unwrap()
                ))
            });

            let tokenizer = search::load_tokenizer();

            let image_session_watcher = Arc::clone(&image_session);
            let conn_watcher = Arc::clone(&conn);

            // Manage the state
            app.manage(state::AppState {
                image_session,
                text_session,
                conn,
                tokenizer,
                hwnd: 0.into(),
                models_dir
            });

            let hwnd = app.get_webview_window("main")
                .and_then(|w| w.hwnd().ok())
                .map(|h| h.0 as isize)
                .unwrap_or(0);
 
            app.state::<AppState>().hwnd_set(hwnd);
 
            let handle = app.handle().clone();
            let handle2 = app.handle().clone();
            watcher::startup_auto_index(Arc::clone(&image_session_watcher), Arc::clone(&conn_watcher), handle);
            watcher::start_watching(image_session_watcher, conn_watcher, handle2);
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::index::index_directory,
            commands::search::search,
            commands::thumbnail::get_thumbnails,
            commands::index::get_indexed_count,
            commands::database::delete_from_db,
            commands::database::delete_from_system,
            commands::misc::open_path,
            commands::folder::show_in_folder,
            commands::misc::find_similar,
            commands::misc::mark_feedback,
            commands::misc::remove_feedback,
            commands::folder::save_folder,
            commands::folder::get_watched_folders,
            commands::settings::set_setting,
            commands::settings::get_settings,
            commands::thumbnail::regenerate_thumbnails,
            commands::duplicate::remove_duplicates,
            commands::duplicate::find_duplicates,
            commands::folder::get_folder_stats,
            commands::folder::set_folder_auto_index,
            commands::index::unindex_folder,
            commands::database::delete_folder_from_system,
            commands::database::clear_database,
            commands::collection::create_collection,
            commands::collection::get_collections,
            commands::collection::rename_collection,
            commands::collection::delete_collection,
            commands::collection::add_to_collection,
            commands::collection::remove_from_collection,
            commands::collection::get_collection_items,
            commands::collection::is_image_in_collection,
            commands::collection::get_image_collections,
            commands::collection::get_collection_count,
            commands::misc::get_available_models,
            commands::misc::reload_model,
        ])
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                let state = window.state::<AppState>();
                if state.image_session.try_lock().is_err() {
                    api.prevent_close();
                    let _ = window.emit("close-requested-while-indexing", ());
                }
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
