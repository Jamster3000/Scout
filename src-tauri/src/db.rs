use rusqlite::{Connection, Result};

pub fn open(path: &str) -> Result<Connection> {
	let conn = Connection::open(path)?;

	conn.execute_batch("
        PRAGMA journal_mode=WAL;
        PRAGMA foreign_keys=ON;

        CREATE TABLE IF NOT EXISTS images (
            id  INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT NOT NULL UNIQUE,
            embedding BLOB,
            thumbnail BLOB,
            raw_preview BLOB,
            aspect_ratio REAL DEFAULT 1.0,
            indexed_at INTEGER NOT NULL DEFAULT (unixepoch())
        );

        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT NOT NULL,
            query TEXT NOT NULL,
            signal INTEGER NOT NULL,
            created_at INTEGER NOT NULL DEFAULT (unixepoch()),
            UNIQUE(path, query)
        );

        CREATE TABLE IF NOT EXISTS embeddings (
            image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
            model_family TEXT NOT NULL,
            embedding BLOB NOT NULL,
            PRIMARY KEY (image_id, model_family)
        );

        CREATE TABLE IF NOT EXISTS watched_folders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT NOT NULL UNIQUE,
            auto_index INTEGER NOT NULL DEFAULT 0,
            last_indexed INTEGER
        );

        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
 
        CREATE TABLE IF NOT EXISTS performance_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            operation_type TEXT NOT NULL,
            file_size INTEGER,
            image_width INTEGER,
            image_height INTEGER,
            duration_ms INTEGER,
            created_at INTEGER NOT NULL DEFAULT (unixepoch())
        );

        CREATE TABLE IF NOT EXISTS collections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            description TEXT,
            created_at INTEGER NOT NULL DEFAULT (unixepoch()),
            updated_at INTEGER NOT NULL DEFAULT (unixepoch())
         );

        CREATE TABLE IF NOT EXISTS collection_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            collection_id INTEGER NOT NULL REFERENCES collections(id) ON DELETE CASCADE,    
            image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
            added_at INTEGER NOT NULL DEFAULT (unixepoch()),
            UNIQUE(collection_id, image_id)
        );

        INSERT OR IGNORE INTO settings (key, value) VALUES
            ('thumbnail_size', '224'),
            ('image_layout', 'grid'),
            ('notify_on_complete', '1'),
            ('prompt_delete_db', '1'),
            ('prompt_delete_system', '1'),
            ('deduplicate_mode', 'db'),
            ('excluded_file_types', ''),
            ('image_model', 'CLIP ViT-B-16/image_encoder.onnx'),
            ('text_model', 'CLIP ViT-B-16/text_encoder.onnx'),
            ('model_family', 'CLIP ViT-B-16');

        CREATE INDEX IF NOT EXISTS idx_metrics_operation ON performance_metrics(operation_type);
        CREATE INDEX IF NOT EXISTS idx_path ON images(path);
        CREATE INDEX IF NOT EXISTS idx_collection_items_collection ON collection_items(collection_id);
        CREATE INDEX IF NOT EXISTS idx_collection_items_image ON collection_items(image_id);
    ")?;
	Ok(conn)
}
