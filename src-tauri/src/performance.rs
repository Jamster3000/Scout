use rusqlite::{Connection, params};

#[derive(Clone, Copy, Debug)]
pub struct PerformanceEstimate {
    pub estimated_ms: i64,
}

pub fn record_performance(
    conn: &Connection,
    operation_type: &str,
    file_size: i32,
    duration_ms: i64,
) -> Result<(), rusqlite::Error> {
    conn.execute(
        "INSERT INTO performance_metrics (operation_type, file_size, duration_ms, created_at)
         VALUES (?1, ?2, ?3, unixepoch('now'))",
        params![operation_type, file_size, duration_ms],
    )?;
    Ok(())
}

pub fn estimate_operation_time(
    conn: &Connection,
    operation_type: &str,
    file_size: i32,
) -> Option<PerformanceEstimate> {
    // Get the average duration and averagee file size
    let mut stmt = conn
        .prepare(
            "SELECT AVG(duration_ms), AVG(file_size)
             FROM (
                 SELECT duration_ms, file_size
                 FROM performance_metrics
                 WHERE operation_type = ?1
                 AND created_at > unixepoch('now', '-30 days')
                 ORDER BY created_at DESC
                 LIMIT 10000
             )",
        )
        .ok()?;

    let (avg_duration, avg_size): (Option<i64>, Option<i64>) = stmt
        .query_row(params![operation_type], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })
        .ok()?;

    let avg_duration = avg_duration?;
    let avg_size = avg_size.unwrap_or(0);

    //This can happen when the user is indexing for the first time and has no images in the database
    if avg_size <= 0 {
        return Some(PerformanceEstimate { estimated_ms: avg_duration });
    }

    // Scale the estimate based on how the requested file_size compares to historical avg_size
    let size_ratio = (file_size as f64 / avg_size as f64).clamp(0.1, 10.0);
    let scaled_estimate = (avg_duration as f64 * size_ratio) as i64;

    Some(PerformanceEstimate { estimated_ms: scaled_estimate })
}

pub fn estimate_remaining_time(
    conn: &Connection,
    operation_type: &str,
    total_files: usize,
    avg_file_size: i32,
    done: usize,
    elapsed_ms: i64,
) -> Option<PerformanceEstimate> {
    // Historical per-item estimate
    let hist_per_item_opt = estimate_operation_time(conn, operation_type, avg_file_size)
        .map(|p| p.estimated_ms as f64);

    if total_files == 0 {
        return Some(PerformanceEstimate { estimated_ms: 0 });
    }

    // Return the histoical total estimate if no items have been done yet
    if done == 0 {
        if let Some(h) = hist_per_item_opt {
            let total_est = (h * total_files as f64).round() as i64;
            return Some(PerformanceEstimate { estimated_ms: total_est.max(0) });
        }
        return None;
    }

    let avg_current = (elapsed_ms as f64) / (done as f64);

    // Weighting factor: 2.0 for faster adaptation, 5.0 for slower
    let k = 2.0_f64;
    let alpha = (done as f64) / ((done as f64) + k); // in [0,1)
    let per_item = if let Some(hist) = hist_per_item_opt {
        alpha * avg_current + (1.0 - alpha) * hist
    } else {
        avg_current
    };

    let remaining_count = (total_files.saturating_sub(done)) as f64;
    let remaining_ms = (per_item * remaining_count).round() as i64;

    Some(PerformanceEstimate { estimated_ms: remaining_ms.max(0) })
}

pub fn estimate_batch_time(
    conn: &Connection,
    operation_type: &str,
    total_files: usize,
    avg_file_size: i32,
) -> Option<PerformanceEstimate> {
    estimate_operation_time(conn, operation_type, avg_file_size)
        .map(|p| PerformanceEstimate { estimated_ms: p.estimated_ms * total_files as i64 })
}