use crate::AppState;
use crate::{time_start, time_end, time_block};
use tauri::{AppHandle, Manager};

#[tauri::command]
pub async fn find_duplicates(app: AppHandle) -> Result<Vec<Vec<String>>, String> {
    tokio::task::spawn_blocking(move || {
        let state = app.state::<AppState>();
        let conn = state.conn.lock().unwrap();

        let model_family = conn.query_row(
            "SELECT value FROM settings WHERE key = 'model_family'",
            [], |r| r.get::<_, String>(0)
        ).unwrap_or_else(|_| "CLIP ViT-B-16".to_string());

        time_start!(t_dup);

        let images: Vec<(String, Vec<f32>)> = time_block!("find_duplicates: load embeddings", {
            let mut stmt = conn.prepare(
                "SELECT i.path, e.embedding FROM images i
                 INNER JOIN embeddings e ON e.image_id = i.id AND e.model_family = ?1"
            ).unwrap();

            stmt.query_map(rusqlite::params![model_family], |row| {
                let path: String = row.get(0)?;
                let bytes: Vec<u8> = row.get(1)?;
                Ok((path, bytes))
            }).unwrap()
            .filter_map(|r| r.ok())
            .map(|(path, bytes)| {
                let embedding = bytes.chunks(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect();
                (path, embedding)
            })
            .collect()
        });

        eprintln!("[timing] find_duplicates: {} embeddings loaded", images.len());

        // Find groups of duplicates
        let duplicate_groups = time_block!("find_duplicates: pairwise comparison", {
            let mut duplicate_groups: Vec<Vec<String>> = Vec::new();
            let mut already_grouped: std::collections::HashSet<String> = std::collections::HashSet::new();

            for i in 0..images.len() {
                if already_grouped.contains(&images[i].0) { continue; }
                let mut group = vec![images[i].0.clone()];

                for j in (i + 1)..images.len() {
                    if already_grouped.contains(&images[j].0) { continue; }
                    let score: f32 = images[i].1.iter()
                        .zip(images[j].1.iter())
                        .map(|(a, b)| a * b)
                        .sum();
                    if score >= 0.99 {
                        group.push(images[j].0.clone());
                        already_grouped.insert(images[j].0.clone());
                    }
                }

                if group.len() > 1 {
                    already_grouped.insert(images[i].0.clone());
                    duplicate_groups.push(group);
                }
            }
            duplicate_groups
        });

        eprintln!("[timing] find_duplicates: {} duplicate groups found", duplicate_groups.len());
        time_end!(t_dup, "find_duplicates: total");

        Ok(duplicate_groups)
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
pub async fn remove_duplicates(mode: String, app: AppHandle) -> Result<usize, String> {
    tokio::task::spawn_blocking(move || {
        let state = app.state::<AppState>();
        let conn = state.conn.lock().unwrap();

        let model_family = conn.query_row(
            "SELECT value FROM settings WHERE key = 'model_family'",
            [], |r| r.get::<_, String>(0)
        ).unwrap_or_else(|_| "CLIP ViT-B-16".to_string());

        time_start!(t_dedup);

        let images: Vec<(String, Vec<f32>)> = time_block!("remove_duplicates: load embeddings", {
            let mut stmt = conn.prepare(
                "SELECT i.path, e.embedding FROM images i
                 INNER JOIN embeddings e ON e.image_id = i.id AND e.model_family = ?1
                 ORDER BY i.path"
            ).unwrap();

            stmt.query_map(rusqlite::params![model_family], |row| {
                let path: String = row.get(0)?;
                let bytes: Vec<u8> = row.get(1)?;
                Ok((path, bytes))
            }).unwrap()
            .filter_map(|r| r.ok())
            .map(|(path, bytes)| {
                let embedding = bytes.chunks(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect();
                (path, embedding)
            })
            .collect()
        });

        let to_remove: Vec<String> = time_block!("remove_duplicates: pairwise comparison", {
            let mut to_remove: Vec<String> = Vec::new();
            let mut already_grouped: std::collections::HashSet<String> = std::collections::HashSet::new();

            for i in 0..images.len() {
                if already_grouped.contains(&images[i].0) { continue; }
                for j in (i + 1)..images.len() {
                    if already_grouped.contains(&images[j].0) { continue; }
                    let score: f32 = images[i].1.iter()
                        .zip(images[j].1.iter())
                        .map(|(a, b)| a * b)
                        .sum();
                    if score >= 0.99 {
                        to_remove.push(images[j].0.clone());
                        already_grouped.insert(images[j].0.clone());
                    }
                }
            }
            to_remove
        });

        eprintln!("[timing] remove_duplicates: {} duplicates to remove", to_remove.len());

        let count = to_remove.len();
        time_block!("remove_duplicates: DB delete", {
            for path in &to_remove {
                conn.execute("DELETE FROM images WHERE path = ?1", rusqlite::params![path]).ok();
                if mode == "system" {
                    std::fs::remove_file(path).ok();
                }
            }
        });

        time_end!(t_dedup, "remove_duplicates: total");
        Ok(count)
    })
    .await
    .map_err(|e| e.to_string())?
}
