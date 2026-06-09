use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Clone)]
pub struct IndexProgress {
    pub done: usize,
    pub total: usize,
    pub estimated_remaining_ms: Option<i64>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct FolderInfo {
    pub path: String,
    pub auto_index: bool,
    pub last_indexed: Option<i64>,
    pub image_count: i64,
    pub last_image_path: Option<String>
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Collection {
    pub id: i32,
    pub name: String,
    pub description: Option<String>,
    pub created_at: i64,
    pub updated_at: i64,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct CollectionItem {
    pub id: i32,
    pub collection_id: i32,
    pub path: String,
    pub thumbnail: Vec<u8>,
    pub aspect_ratio: f32,
    pub added_at: i64,
}
