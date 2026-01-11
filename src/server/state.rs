//! Server application state

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokio::sync::RwLock;
use eywa::{BM25Index, Embedder, SearchEngine, SharedJobQueue, VectorDB};
use serde::Serialize;

// ─────────────────────────────────────────────────────────────────────────────
// Download Job Tracking
// ─────────────────────────────────────────────────────────────────────────────

/// Status of a model download
#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum DownloadStatus {
    Pending,
    Downloading,
    Done,
    Failed,
}

/// Progress for a single file being downloaded
#[derive(Debug, Clone, Serialize)]
pub struct FileProgress {
    pub name: String,
    pub bytes_downloaded: u64,
    pub total_bytes: Option<u64>,
    pub done: bool,
}

/// A model download job
#[derive(Debug, Clone, Serialize)]
pub struct DownloadJob {
    pub id: String,
    pub model_type: String,  // "embedder" or "reranker"
    pub model_id: String,
    pub model_name: String,
    pub status: DownloadStatus,
    pub files: Vec<FileProgress>,
    pub started_at: String,
    pub completed_at: Option<String>,
    pub error: Option<String>,
}

impl DownloadJob {
    /// Calculate total progress as a fraction (0.0 to 1.0)
    pub fn total_progress(&self) -> f64 {
        let total: u64 = self.files.iter().filter_map(|f| f.total_bytes).sum();
        let downloaded: u64 = self.files.iter().map(|f| f.bytes_downloaded).sum();
        if total == 0 {
            if self.files.iter().all(|f| f.done) { 1.0 } else { 0.0 }
        } else {
            downloaded as f64 / total as f64
        }
    }
}

/// Thread-safe download job tracker
pub type DownloadTracker = Arc<Mutex<HashMap<String, DownloadJob>>>;

pub fn create_download_tracker() -> DownloadTracker {
    Arc::new(Mutex::new(HashMap::new()))
}

// ─────────────────────────────────────────────────────────────────────────────
// App State
// ─────────────────────────────────────────────────────────────────────────────

/// Shared application state for all route handlers
pub struct AppState {
    pub embedder: Arc<Embedder>,
    pub db: Arc<RwLock<VectorDB>>,
    pub bm25_index: Arc<BM25Index>,
    pub search_engine: SearchEngine,
    pub job_queue: SharedJobQueue,
    pub data_dir: String,
    pub downloads: DownloadTracker,
}
