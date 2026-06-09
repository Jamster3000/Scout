use tauri::{AppHandle, Emitter};
use crate::commands::models::IndexProgress;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use notify_debouncer_mini::{new_debouncer, notify::RecursiveMode, DebounceEventResult};
use ort::session::Session;
use rusqlite::Connection;

static EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "webp", "bmp", "JPG", "JPEG", "PNG", "WEBP", "BMP"];

/// Checks if the given path has an image file extension.
///
/// This function looks at the file extension of the provided path and checks if it matches any of the known 
/// image extensions defined in the `EXTENSIONS` array. It returns `true` if the extension is recognized 
/// as an image format, and `false` otherwise.
///
/// # Arguments
/// * `path` - A reference to a `Path` that represents the file path to check.
///
/// # Returns
/// * `bool` - `true` if the file has a recognized image extension, `false` otherwise.
fn is_image(path: &Path, conn: &Connection) -> bool {
    let excluded_types = get_excluded_file_types(conn);
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| {
            let ext = e.to_lowercase();
            EXTENSIONS.contains(&e.to_lowercase().as_str()) && !excluded_types.contains(&ext)
        })
        .unwrap_or(false)
}

/// Checks if the given path is already indexed in the database.
///
/// This function queries the database to determine if a record exists for the provided file path.
///
/// # Arguments
/// * `path` - A reference to a `Path` that represents the file path to check.
/// * `conn` - A reference to a `Connection` object that represents the database connection.
///
/// # Returns
/// * `bool` - `true` if the file path is already indexed in the database, `false` otherwise.
fn is_already_indexed(path: &Path, conn: &Connection) -> bool {
    conn.query_row(
        "SELECT COUNT(*) FROM images WHERE path = ?1",
        rusqlite::params![path.to_str().unwrap_or("")],
        |r: &rusqlite::Row| r.get::<_, i64>(0),
    ).unwrap_or(0) > 0
}

/// Retrieves a list of folders that are marked for automatic indexing from the database.
///
/// This function queries the `watched_folders` table in the database to find all folders that have the `auto_index` flag set to 1.
///
/// # Arguments
/// * `conn` - A reference to a `Connection` object that represents the database connection.
///
/// # Returns
/// * `Vec<String>` - A vector of strings, where each string is a path to a folder that should be automatically indexed.
fn get_auto_index_folders(conn: &Connection) -> Vec<String> {
    let mut stmt = match conn.prepare(
        "SELECT path FROM watched_folders WHERE auto_index = 1"
    ) {
        Ok(s) => s,
        Err(_) => return vec![],
    };
    stmt.query_map([], |r| r.get(0))
        .unwrap()
        .filter_map(|r| r.ok())
        .collect()
}

/// Retrieves the file path of the database from the given connection.
///
/// This function attempts to extract the file path of the database associated with the provided connection.
///
/// # Arguments
/// * `conn` - A reference to an `Arc<Mutex<Connection>>` that represents the database connection wrapped in a thread-safe manner.
///
/// # Returns
/// * `Option<String>` - An `Option` that contains the database file path as a `String` if it can be retrieved, or `None` if it cannot.
fn get_db_path(conn: &Arc<Mutex<Connection>>) -> Option<String> {
    let c = conn.lock().unwrap();
    c.path().map(|p| p.to_string().to_string())
}

/// Starts the automatic indexing process for folders marked with the auto_index flag in the database.
///
/// This function spawns a new thread that waits for a short duration before retrieving the list of folders to index.
/// It then opens a dedicated database connection and iterates through each folder, indexing its contents and emitting 
/// progress events to the application.
///
/// # Arguments
/// * `image_session` - An `Arc<Mutex<Session>>` that represents the ONNX runtime session wrapped in a thread-safe manner.
/// * `conn` - An `Arc<Mutex<Connection>>` that represents the database connection wrapped in a thread-safe manner.
/// * `app` - An `AppHandle` that allows emitting events to the Tauri application.
///
/// # Example
/// ```rust
/// let image_session = Arc::new(Mutex::new(Session::new(...)));
/// let conn = Arc::new(Mutex::new(Connection::open("mydb.sqlite").unwrap()));
/// let app_handle = app.handle();
/// startup_auto_index(image_session, conn, app_handle);
/// ```
pub fn startup_auto_index(
    image_session: Arc<Mutex<Session>>,
    conn: Arc<Mutex<Connection>>,
    app: AppHandle,
) {
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_secs(2));

        let folders = {
            let c = conn.lock().unwrap();
            get_auto_index_folders(&c)
        };

        if folders.is_empty() { return; }

        let db_path = get_db_path(&conn);
        let index_conn = match db_path.as_deref().and_then(|p| crate::db::open(p).ok()) {
            Some(c) => c,
            None => {
                eprintln!("[watcher] failed to open dedicated DB connection for startup");
                return;
            }
        };

        for folder in folders {
            let app_clone = app.clone();
            let mut session = image_session.lock().unwrap();
            
            // Read thumbnail size from settings
            let thumbnail_size: u32 = index_conn.query_row(
                "SELECT value FROM settings WHERE key = 'thumbnail_size'",
                [],
                |r| r.get::<_, String>(0),
            ).ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(224);

            crate::indexer::index_directory(&folder, &mut session, &index_conn, thumbnail_size, |done, total, estimated_remaining_ms| {
                let _ = app_clone.emit("index-progress", IndexProgress { done, total, estimated_remaining_ms });
            });
            drop(session);

            index_conn.execute(
                "UPDATE watched_folders SET last_indexed = unixepoch() WHERE path = ?1",
                rusqlite::params![folder],
            ).ok();

            let _ = app.emit("index-complete", ());
        }
    });
}

/// Starts watching for file changes in folders marked with the auto_index flag and updates the database accordingly.
///
/// This function spawns a new thread that continuously monitors the folders specified in the database for any file changes.
///
/// # Arguments
/// * `image_session` - An `Arc<Mutex<Session>>` that represents the ONNX runtime session wrapped in a thread-safe manner.
/// * `conn` - An `Arc<Mutex<Connection>>` that represents the database connection wrapped in a thread-safe manner.
/// * `app` - An `AppHandle` that allows emitting events to the Tauri application.
///
/// # Example
/// ```rust
/// let image_session = Arc::new(Mutex::new(Session::new(...)));
/// let conn = Arc::new(Mutex::new(Connection::open("mydb.sqlite").unwrap()));
/// let app_handle = app.handle();
/// start_watching(image_session, conn, app_handle);
/// ```
pub fn start_watching(
    image_session: Arc<Mutex<Session>>,
    conn: Arc<Mutex<Connection>>,
    app: AppHandle,
) {
    std::thread::spawn(move || {
        let db_path = get_db_path(&conn);
        let watch_conn = match db_path.as_deref().and_then(|p| crate::db::open(p).ok()) {
            Some(c) => c,
            None => {
                return;
            }
        };
        let watch_conn = Arc::new(Mutex::new(watch_conn));

        let session_for_handler = Arc::clone(&image_session);
        let conn_for_handler = Arc::clone(&watch_conn);
        let app_for_handler = app.clone();

        //debounce/wait seconds before actually triggering event handlers
        let mut debouncer = new_debouncer(
            Duration::from_secs(2),
            move |result: DebounceEventResult| {
                match result {
                    Ok(events) => {
                        for event in events {
                            let path: PathBuf = event.path;
                            
                            // Handle file deletion/removal - check if file no longer exists
                            if !path.exists() {
                                if !is_image(&path, &conn_for_handler.lock().unwrap()) { continue; }
                                eprintln!("[watcher] Deleting from DB: {:?}", path.to_str().unwrap_or(""));
                                let c = conn_for_handler.lock().unwrap();
                                match c.execute(
                                    "DELETE FROM images WHERE path = ?1",
                                    rusqlite::params![path.to_str().unwrap_or("")],
                                ) {
                                    Ok(count) => eprintln!("[watcher] Deleted {} row(s)", count),
                                    Err(e) => eprintln!("[watcher] Delete failed: {}", e),
                                }
                                let _ = app_for_handler.emit("index-complete", ());
                                continue;
                            }
                            
                            // Handle file creation/modification
                            if !is_image(&path, &conn_for_handler.lock().unwrap()) { continue; }
                            {
                                let c = conn_for_handler.lock().unwrap();
                                if is_already_indexed(&path, &c) { continue; }
                            }
                            let parent = path.parent()
                                .map(|p| p.to_path_buf())
                                .unwrap_or_else(|| path.clone());
                            let mut session = session_for_handler.lock().unwrap();
                            let conn = conn_for_handler.lock().unwrap();
                            
                            // Read thumbnail size from settings
                            let thumbnail_size: u32 = conn.query_row(
                                "SELECT value FROM settings WHERE key = 'thumbnail_size'",
                                [],
                                |r| r.get::<_, String>(0),
                            ).ok()
                                .and_then(|v| v.parse().ok())
                                .unwrap_or(224);
                            
                            let app_clone = app_for_handler.clone();
                            crate::indexer::index_directory(
                                parent.to_str().unwrap_or(""),
                                &mut session,
                                &conn,
                                thumbnail_size,
                                |done, total, estimated_remaining_ms| {
                                    let _ = app_clone.emit("index-progress", IndexProgress { done, total, estimated_remaining_ms });
                                },
                            );
                            drop(conn);
                            drop(session);
                            let _ = app_for_handler.emit("index-complete", ());
                        }
                    }
                    Err(e) => eprintln!("[watcher] error: {:?}", e),
                }
            }
        ).unwrap();

        let mut watched: std::collections::HashSet<String> = std::collections::HashSet::new();

        loop {
            let folders = {
                let c = watch_conn.lock().unwrap();
                get_auto_index_folders(&c)
            };

            for folder in &folders {
                if !watched.contains(folder) {
                    debouncer.watcher()
                        .watch(Path::new(folder), RecursiveMode::Recursive)
                        .ok();
                    watched.insert(folder.clone());
                }
            }

            std::thread::sleep(Duration::from_secs(2));
        }
    });
}

fn get_excluded_file_types(conn: &Connection) -> Vec<String> {
    conn.query_row(
        "SELECT value FROM settings WHERE key = 'excluded_file_types'",
        [],
        |r| r.get::<_, String>(0),
    )
    .unwrap_or_else(|_| String::new())
    .split(',')
    .filter(|s| !s.trim().is_empty())
    .map(|s| s.trim().to_lowercase().to_string())
    .collect()
}
