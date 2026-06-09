use std::sync::{Arc, Mutex};
use std::sync::atomic::AtomicIsize;
use ort::session::Session;
use rusqlite::Connection;
use instant_clip_tokenizer::Tokenizer;
use std::path::PathBuf;

pub struct AppState {
    pub image_session: Arc<Mutex<Session>>,
    pub text_session:  Arc<Mutex<Session>>,
    pub conn: Arc<Mutex<Connection>>,
    pub tokenizer: Tokenizer,
    pub hwnd: AtomicIsize,
    pub models_dir: PathBuf,
}

impl AppState {
    pub fn hwnd_set(&self, hwnd: isize) {
        self.hwnd.store(hwnd, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn hwnd_get(&self) -> isize {
        self.hwnd.load(std::sync::atomic::Ordering::Relaxed)
    }
}