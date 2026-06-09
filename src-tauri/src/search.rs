use ort::session::Session;
use ort::value::Tensor;
use rusqlite::Connection;
use instant_clip_tokenizer::Tokenizer;
use crate::{time_start, time_end, time_block};

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn bytes_to_embedding(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

pub fn load_tokenizer() -> Tokenizer {
    Tokenizer::new()
}

fn encode_text(query: &str, tokenizer: &Tokenizer, session: &mut Session) -> Vec<f32> {
    let mut tokens = Vec::new();
    tokens.push(tokenizer.start_of_text());
    tokenizer.encode(query, &mut tokens);
    tokens.push(tokenizer.end_of_text());

    let mut input_ids = vec![0i64; 77];
    let mut attention_mask = vec![0i64; 77];
    let len = tokens.len().min(77);
    for i in 0..len {
        input_ids[i] = tokens[i].to_u16() as i64;
        attention_mask[i] = 1;
    }

    let ids_tensor = match Tensor::<i64>::from_array(([1usize, 77], input_ids.into_boxed_slice())) {
        Ok(tensor) => tensor,
        Err(e) => {
            eprintln!("Failed to create input tensor: {:?}", e);
            return vec![];
        }
    };

    let query_embedding = {
        //try passing only a single input (mobile clip only takes one input)
        let single_embedding = match session.run(ort::inputs![ids_tensor.clone()]) {
            Ok(outputs) => match outputs[0].try_extract_tensor::<f32>() {
                Ok((_, embedding)) => Some(embedding.to_vec()),
                Err(e) => {
                    eprintln!("Failed to extract tensor from single input: {:?}", e);
                    None
                }
            },
            Err(e) => {
                eprintln!("Failed to run session with single input: {:?}", e);
                None
            }
        };

        if let Some(embedding) = single_embedding {
            embedding
        } else {
            //fallback to passing two inputs (clip ViT 16 uses two inputs)
            let mask_tensor = match Tensor::<i64>::from_array(([1usize, 77], attention_mask.into_boxed_slice())) {
                Ok(tensor) => tensor,
                Err(e) => {
                    eprintln!("Failed to create attention mask tensor: {:?}", e);
                    return vec![];
                }
            };

            match session.run(ort::inputs![ids_tensor, mask_tensor]) {
                Ok(outputs) => match outputs[0].try_extract_tensor::<f32>() {
                    Ok((_, embedding)) => embedding.to_vec(),
                    Err(e) => {
                        eprintln!("Failed to extract tensor from dual input: {:?}", e);
                        vec![]
                    }
                },
                Err(e) => {
                    eprintln!("Failed to run session with dual input: {:?}", e);
                    vec![]
                }
            }
        }
    };

    query_embedding
}

fn normalize_embedding(embedding: &mut [f32]) {
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in embedding.iter_mut() {
            *x /= norm;
        }
    }
}

pub fn search(
    query: &str,
    tokenizer: &Tokenizer,
    session: &mut Session,
    conn: &Connection,
) -> Vec<(String, f32, f32)> {
    time_start!(t_search);

    //parse the query for positive and negative terms
    let (positive_query, negative_queries) = parse_query(query); 

    //encode only positive part of the query
    let mut combined_embedding = encode_text(&positive_query, tokenizer, session);

    //subtract negative queries from the embedding
    for negative_term in negative_queries {
        let negative_embedding = encode_text(&negative_term, tokenizer, session);
        for (i, neg_val) in negative_embedding.iter().enumerate() {
            combined_embedding[i] -= neg_val * 0.5; //weight factor for negative terms
        }
    }

    normalize_embedding(&mut combined_embedding);

    let model_family = {
        let db = conn;
        db.query_row(
            "SELECT value FROM settings WHERE key = 'model_family'",
            [], |r| r.get::<_, String>(0)
        ).unwrap_or_else(|_| "CLIP ViT-B-16".to_string())
    };

    let feedback: std::collections::HashMap<String, i32> = {
        let mut stmt = conn.prepare(
            "SELECT path, signal FROM feedback WHERE query = ?1"
        ).unwrap();
        stmt.query_map(rusqlite::params![query], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, i32>(1)?))
        }).unwrap()
        .filter_map(|r| r.ok())
        .collect()
    };

    // Only fetch path + embedding + ratio - skip thumbnails entirely at scoring time
    let mut stmt = conn.prepare(
        "SELECT i.path, e.embedding, COALESCE(i.aspect_ratio, 1.0) 
         FROM images i
         INNER JOIN embeddings e ON e.image_id = i.id AND e.model_family = ?1"
    ).unwrap();

    let mut results: Vec<(String, f32, f32)> = stmt.query_map(rusqlite::params![model_family], |row: &rusqlite::Row| {
        let path: String = row.get(0)?;
        let bytes: Vec<u8> = row.get(1)?;
        let ratio: f32 = row.get(2).unwrap_or(1.0);
        Ok((path, bytes, ratio))
    }).unwrap()
    .filter_map(|r: rusqlite::Result<(String, Vec<u8>, f32)>| r.ok())
    .filter_map(|(path, bytes, ratio)| {
        let signal = feedback.get(&path).copied().unwrap_or(0);
        if signal == -1 { return None; }
        let embedding = bytes_to_embedding(&bytes);
        let mut score = cosine_similarity(&combined_embedding, &embedding);
        if signal == 1 { score *= 1.2; }
        Some((path, score, ratio))
    })
    .collect();

    time_block!("search: sort", {
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    });

    time_end!(t_search, "search: total");
    results
}

pub fn get_thumbnails(
    paths: &[String],
    conn: &Connection,
) -> Vec<(String, Vec<u8>, Option<Vec<u8>>)> {
    if paths.is_empty() { return vec![]; }

    // Build a positional map so we can return results in the same order as `paths`
    let index: std::collections::HashMap<&str, usize> = paths
        .iter()
        .enumerate()
        .map(|(i, p)| (p.as_str(), i))
        .collect();

    let placeholders: String = paths.iter().enumerate()
        .map(|(i, _)| format!("?{}", i + 1))
        .collect::<Vec<_>>()
        .join(", ");

    let sql = format!(
        "SELECT path, thumbnail, raw_preview FROM images WHERE path IN ({})",
        placeholders
    );

    let mut stmt = match conn.prepare(&sql) {
        Ok(s) => s,
        Err(_) => return vec![],
    };

    let params: Vec<&dyn rusqlite::ToSql> = paths.iter()
        .map(|p| p as &dyn rusqlite::ToSql)
        .collect();

    let mapped = match stmt.query_map(params.as_slice(), |row| {
        let path: String = row.get(0)?;
        let thumb: Vec<u8> = row.get(1).unwrap_or_default();
        let raw_preview: Option<Vec<u8>> = row.get(2).ok().flatten();
        Ok((path, thumb, raw_preview))
    }) {
        Ok(rows) => rows,
        Err(_) => return vec![],
    };

    let mut rows: Vec<(usize, String, Vec<u8>, Option<Vec<u8>>)> = mapped
        .filter_map(|r| r.ok())
        .filter_map(|(path, thumb, raw)| {
            let pos = *index.get(path.as_str())?;
            Some((pos, path, thumb, raw))
        })
        .collect();

    rows.sort_by_key(|(pos, _, _, _)| *pos);
    rows.into_iter().map(|(_, path, thumb, raw)| (path, thumb, raw)).collect()
}

fn parse_query(query: &str) -> (String, Vec<String>) {
    let parts: Vec<&str> = query.split_whitespace().collect();
    let mut positive = Vec::new();
    let mut negative = Vec::new();

    for part in parts {
        if part.starts_with('-') && part.len() > 1 {
            negative.push(part[1..].to_string());
        } else {
            positive.push(part.to_string());
        }
    }

    let positive_query = if positive.is_empty() {
        "".to_string()
    } else {
        positive.join(" ")
    };

    (positive_query, negative)
}

pub fn find_similar(
    source_path: &str,
    conn: &Connection,
    top_n: usize,
) -> Vec<(String, f32, f32)> {
    time_start!(t_similar);

    let source_embedding: Vec<f32> = match conn.query_row(
        "SELECT embedding FROM images WHERE path = ?1",
        rusqlite::params![source_path],
        |r| r.get::<_, Vec<u8>>(0),
    ) {
        Ok(bytes) => bytes_to_embedding(&bytes),
        Err(_) => return vec![],
    };

    let mut stmt = conn.prepare(
        "SELECT path, embedding, COALESCE(aspect_ratio, 1.0) FROM images WHERE embedding IS NOT NULL AND path != ?1"
    ).unwrap();

    let mut results: Vec<(String, f32, f32)> = stmt.query_map(
        rusqlite::params![source_path],
        |row: &rusqlite::Row| {
            let path: String = row.get(0)?;
            let bytes: Vec<u8> = row.get(1)?;
            let ratio: f32 = row.get(2).unwrap_or(1.0);
            Ok((path, bytes, ratio))
        }
    ).unwrap()
    .filter_map(|r: rusqlite::Result<(String, Vec<u8>, f32)>| r.ok())
    .map(|(path, bytes, ratio)| {
        let embedding = bytes_to_embedding(&bytes);
        let score = cosine_similarity(&source_embedding, &embedding);
        (path, score, ratio)
    })
    .collect();

    time_block!("find_similar: sort + truncate", {
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.truncate(top_n);
    });

    time_end!(t_similar, "find_similar: total");
    results
}
