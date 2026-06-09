pub mod decode;
pub mod resize;
pub mod embed;

use std::path::{Path, PathBuf};
use std::collections::HashSet;
use std::sync::{mpsc, Arc};
use std::thread;
use rayon::prelude::*;
use ort::session::Session;
use rusqlite::Connection;
use crate::{time_start, time_end, time_block};
use resize::NormConfig;

pub use resize::generate_thumbnail;

const BATCH_SIZE: usize = 1; //Anything above 1 didn't really provide a smaller wall time
const CHANNEL_BUFFER: usize = 32;
const COMMIT_INTERVAL: usize = 500;

pub const ALL_IMAGE_EXTENSIONS: &[&str] = &[
    "jpg", "jpeg", "png", "webp", "tiff", "bmp", "avif", "heic", "heif", "dds", "exr", "ff", "hrd", "ico", "pnm", "qoi", "tga", "jxl", "svg",
];

#[inline]
pub fn is_image(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext_str| {
            ALL_IMAGE_EXTENSIONS
                .iter()
                .any(|&expected| expected.eq_ignore_ascii_case(ext_str))
        })
        .unwrap_or(false)
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

fn get_model_family(conn: &Connection) -> String {
    conn.query_row(
        "SELECT value FROM settings WHERE key = 'model_family'",
        [],
        |r| r.get::<_, String>(0),
    )
    .unwrap_or_else(|_| "CLIP ViT-B-16".to_string())
}

/// Determines whether to use jwalk or walkdir based on directory structure.
/// jwalk has overhead for small/flat directories so we probe first.
fn choose_walker(dir: &Path) -> bool {
    let mut dir_count = 0;
    let mut file_count = 0;
    for entry in walkdir::WalkDir::new(dir)
        .max_depth(2)
        .into_iter()
        .filter_map(Result::ok)
    {
        if entry.file_type().is_dir() { dir_count += 1; }
        else if entry.file_type().is_file() { file_count += 1; }
        if dir_count >= 4 || file_count > 40 { return true; }
    }
    false
}

/// Collects unindexed image paths from a directory, skipping already-embedded
/// images for the current model family and excluded types.
fn collect_paths(
    dir: &Path,
    known: &HashSet<String>,
    excluded_types: &[String],
    use_jwalker: bool,
) -> (Vec<PathBuf>, u64) {
    let mut total_bytes: u64 = 0;
    let mut paths = Vec::new();

    macro_rules! process_entry {
        ($path:expr, $metadata_fn:expr) => {
            if !is_image($path) { continue; }
            if let Some(ext) = $path.extension().and_then(|e| e.to_str()) {
                let ext_lower = ext.to_ascii_lowercase();
                if excluded_types.iter().any(|x| x == &ext_lower) { continue; }
            }
            let path_str = $path.to_string_lossy();
            if known.contains(path_str.as_ref()) { continue; }
            if let Some(size) = $metadata_fn {
                total_bytes += size;
            }
            paths.push($path.to_path_buf());
        };
    }

    if use_jwalker {
        for entry in jwalk::WalkDir::new(dir).into_iter().filter_map(Result::ok) {
            if !entry.file_type().is_file() { continue; }
            let path = entry.path();
            process_entry!(&path, entry.metadata().ok().map(|m| m.len()));
        }
    } else {
        for entry in walkdir::WalkDir::new(dir).into_iter().filter_map(Result::ok) {
            if !entry.file_type().is_file() { continue; }
            let path = entry.path();
            process_entry!(path, entry.metadata().ok().map(|m| m.len()));
        }
    }

    (paths, total_bytes)
}

fn flush_batch(
    session: &mut Session,
    batch_indices: &mut Vec<usize>,
    batch_pixels: &mut Vec<Vec<f32>>,
    batch_thumbs: &mut Vec<Vec<u8>>,
    paths: &[PathBuf],
    stmt_image: &mut rusqlite::CachedStatement,
    stmt_embedding: &mut rusqlite::CachedStatement,
    done: &mut usize,
    flat_buf: &mut Vec<f32>,
    batch_ratios: &mut Vec<f32>,
    batch_raw_previews: &mut Vec<Option<Vec<u8>>>,
    model_family: &str,
) {
    if batch_pixels.is_empty() { return; }

    let n = batch_pixels.len();
    flat_buf.clear();
    for pix in batch_pixels.iter() {
        flat_buf.extend_from_slice(pix);
    }

    let bytes = embed::run_batch(session, flat_buf, n);
    let stride = bytes.len() / n;

    for (i, emb_bytes) in bytes.chunks(stride).enumerate() {
        let path = paths[batch_indices[i]].to_str().unwrap_or("");
        let thumb = &batch_thumbs[i];
        let ratio = batch_ratios[i];
        let raw_preview = batch_raw_previews[i].as_deref();

        // Insert the image row (thumbnail, aspect ratio etc). If the path already
        // exists (image was indexed by a different model previously) this is a no-op
        // thanks to OR IGNORE, and the existing id is reused by the embedding insert below.
        stmt_image.execute(rusqlite::params![path, thumb, ratio, raw_preview]).ok();

        // Upsert the embedding for this model family
        stmt_embedding.execute(rusqlite::params![path, model_family, emb_bytes]).ok();

        *done += 1;
    }

    batch_indices.clear();
    batch_pixels.clear();
    batch_thumbs.clear();
    batch_ratios.clear();
    batch_raw_previews.clear();
}

pub fn index_directory<F>(
    dir: &str,
    session: &mut Session,
    conn: &Connection,
    thumbnail_size: u32,
    mut on_progress: F,
) where F: FnMut(usize, usize, Option<i64>) {
    time_start!(t_index);

    let model_family = get_model_family(conn);
    let norm = NormConfig::from_model_family(&model_family);

    // Load paths that already have an embedding for this specific model family
    // those can be skipped entirely.
    let known: HashSet<String> = time_block!("index: load known paths from DB", {
        let count: usize = conn
            .query_row("SELECT COUNT(*) FROM images", [], |r| r.get(0))
            .unwrap_or(0);
        let mut stmt = conn.prepare_cached(
            "SELECT i.path FROM images i
             INNER JOIN embeddings e ON e.image_id = i.id AND e.model_family = ?1"
        ).unwrap();
        let mut known_set = HashSet::with_capacity(count);
        let rows = stmt.query_map(rusqlite::params![model_family], |r| r.get::<_, String>(0)).unwrap();
        for row in rows {
            if let Ok(path_str) = row { known_set.insert(path_str); }
        }
        known_set
    });

    let dir_path = Path::new(dir);
    let excluded_types = get_excluded_file_types(conn);

    let (paths, avg_file_size) = time_block!("index: directory walk & size calculation", {
        let use_jwalker = choose_walker(dir_path);
        let (collected_paths, total_bytes) = collect_paths(dir_path, &known, &excluded_types, use_jwalker);
        let avg = if !collected_paths.is_empty() {
            (total_bytes / collected_paths.len() as u64).min(i32::MAX as u64) as i32
        } else { 0 };
        (collected_paths, avg)
    });

    let total = paths.len();
    if total == 0 {
        on_progress(0, 0, None);
        return;
    }
    eprintln!("[timing] index: {} new images to process", total);

    let _db_estimate = crate::performance::estimate_batch_time(conn, "index", total, avg_file_size);

    let (tx, rx) = mpsc::sync_channel::<(usize, Vec<f32>, Vec<u8>, f32, Option<Vec<u8>>, std::time::Instant)>(CHANNEL_BUFFER);

    let paths = Arc::new(paths);
    let paths_clone = Arc::clone(&paths);

    let producer = thread::spawn(move || {
        paths_clone.par_iter().enumerate().for_each(|(i, path)| {
            let start = std::time::Instant::now();
            if let Some((data, thumb, ratio, raw_preview)) = decode::load_image(path.as_path(), thumbnail_size, norm) {
                let _ = tx.send((i, data, thumb, ratio, raw_preview, start));
            }
        });
    });

    let mut batch_indices: Vec<usize> = Vec::with_capacity(BATCH_SIZE);
    let mut batch_pixels:  Vec<Vec<f32>> = Vec::with_capacity(BATCH_SIZE);
    let mut batch_thumbs:  Vec<Vec<u8>> = Vec::with_capacity(BATCH_SIZE);
    let mut batch_ratios:  Vec<f32> = Vec::with_capacity(BATCH_SIZE);
    let mut batch_raw_previews: Vec<Option<Vec<u8>>> = Vec::with_capacity(BATCH_SIZE);
    let mut flat_buf: Vec<f32> = Vec::with_capacity(BATCH_SIZE * 3 * 224 * 224);
    let mut done = 0;

    // Insert the image metadata row (path, thumbnail, aspect ratio) OR IGNORE.
    let mut stmt_image = conn.prepare_cached(
        "INSERT OR IGNORE INTO images (path, thumbnail, aspect_ratio, raw_preview)
         VALUES (?1, ?2, ?3, ?4)"
    ).unwrap();

    // Upsert the embedding for this model family specifically.
    let mut stmt_embedding = conn.prepare_cached(
        "INSERT OR REPLACE INTO embeddings (image_id, model_family, embedding)
         VALUES ((SELECT id FROM images WHERE path = ?1), ?2, ?3)"
    ).unwrap();

    conn.execute_batch("BEGIN").unwrap();
    time_start!(t_pipeline);

    let mut perf_records: Vec<i64> = Vec::new();
    let index_start = std::time::Instant::now();

    for (i, pixels, thumb, ratio, raw_preview, item_start) in rx.iter() {
        batch_indices.push(i);
        batch_pixels.push(pixels);
        batch_thumbs.push(thumb);
        batch_ratios.push(ratio);
        batch_raw_previews.push(raw_preview);

        if batch_pixels.len() >= BATCH_SIZE {
            flush_batch(session, &mut batch_indices, &mut batch_pixels, &mut batch_thumbs,
                &paths, &mut stmt_image, &mut stmt_embedding, &mut done, &mut flat_buf,
                &mut batch_ratios, &mut batch_raw_previews, &model_family);

            perf_records.push(item_start.elapsed().as_millis() as i64);

            if done % COMMIT_INTERVAL == 0 {
                conn.execute_batch("COMMIT").unwrap();
                conn.execute_batch("BEGIN").unwrap();
            }

            let elapsed_ms = index_start.elapsed().as_millis() as i64;
            let estimate = crate::performance::estimate_remaining_time(
                conn,
                "index",
                total,
                avg_file_size,
                done,
                elapsed_ms,
            );

            on_progress(done, total, estimate.map(|e| e.estimated_ms));
        }
    }

    flush_batch(session, &mut batch_indices, &mut batch_pixels, &mut batch_thumbs,
        &paths, &mut stmt_image, &mut stmt_embedding, &mut done, &mut flat_buf,
        &mut batch_ratios, &mut batch_raw_previews, &model_family);

    on_progress(done, total, Some(0));
    time_end!(t_pipeline, "index: load+infer+insert pipeline");

    time_block!("index: DB commit", { conn.execute_batch("COMMIT").unwrap() });

    for duration_ms in perf_records {
        let _ = crate::performance::record_performance(conn, "index", avg_file_size, duration_ms);
    }

    time_end!(t_index, "index: total wall time");
    producer.join().ok();
}
