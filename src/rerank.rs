//! Reranker using cross-encoder for better ranking
//!
//! Supports multiple reranker models configured via ~/.eywa/config.toml.

use crate::config::{Config, DevicePreference, RerankerModel};
use crate::embed::{device_name, resolve_device};
use anyhow::{Context, Result};
use candle_core::{Device, Tensor, DType, IndexOp};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use tokenizers::Tokenizer;

/// Get optimal batch size for reranking based on device
fn get_rerank_batch_size(device: &Device) -> usize {
    match device {
        Device::Cpu => 8,   // CPU: smaller batches
        _ => 16,            // GPU: larger batches for better utilization
    }
}

pub struct Reranker {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
}

impl Reranker {
    /// Create a new reranker using the model from config
    pub fn new() -> Result<Self> {
        let config = Config::load()?
            .ok_or_else(|| anyhow::anyhow!("Eywa not initialized. Run 'eywa' or 'eywa init' first."))?;
        Self::new_with_model(&config.reranker_model, &config.device, true)
    }

    /// Create a new reranker with a specific model and device preference
    pub fn new_with_model(
        reranker_model: &RerankerModel,
        device_pref: &DevicePreference,
        show_progress: bool,
    ) -> Result<Self> {
        let device = resolve_device(device_pref)?;
        let model_id = reranker_model.hf_id();

        if show_progress {
            eprintln!(
                "  {} ({} MB) on {}",
                reranker_model.name(),
                reranker_model.size_mb(),
                device_name(&device)
            );
        }

        // Download model files from HuggingFace with progress
        let api = ApiBuilder::new()
            .with_progress(show_progress)
            .build()
            .context("Failed to create HuggingFace API")?;
        let repo = api.repo(Repo::new(model_id.to_string(), RepoType::Model));

        let config_path = repo.get("config.json").context("Failed to get config.json")?;
        let tokenizer_path = repo.get("tokenizer.json").context("Failed to get tokenizer.json")?;
        let weights_path = repo.get("model.safetensors").context("Failed to get model.safetensors")?;

        // Load config
        let config_str = std::fs::read_to_string(&config_path)?;
        let bert_config: BertConfig = serde_json::from_str(&config_str)?;

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        // Load model weights
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)?
        };
        let model = BertModel::load(vb, &bert_config)?;

        if show_progress {
            eprintln!("done");
        }

        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }

    /// Score query-document pairs
    /// Returns relevance scores (higher = more relevant)
    pub fn rerank(&self, query: &str, documents: &[String]) -> Result<Vec<f32>> {
        if documents.is_empty() {
            return Ok(vec![]);
        }

        let batch_size = get_rerank_batch_size(&self.device);
        let mut all_scores = Vec::with_capacity(documents.len());

        // Process in batches for better GPU utilization
        for batch in documents.chunks(batch_size) {
            let batch_scores = self.score_batch(query, batch)?;
            all_scores.extend(batch_scores);
        }

        Ok(all_scores)
    }

    /// Score a batch of query-document pairs
    fn score_batch(&self, query: &str, documents: &[String]) -> Result<Vec<f32>> {
        const MAX_SEQ_LEN: usize = 512;

        // Tokenize all pairs
        let pairs: Vec<(&str, &str)> = documents.iter().map(|d| (query, d.as_str())).collect();
        let encodings = self.tokenizer
            .encode_batch(pairs, true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

        // Find max length (capped at 512)
        let max_len = encodings.iter()
            .map(|e| e.get_ids().len().min(MAX_SEQ_LEN))
            .max()
            .unwrap_or(0);

        let batch_size = encodings.len();

        // Build padded tensors
        let mut input_ids_vec = Vec::with_capacity(batch_size * max_len);
        let mut attention_mask_vec = Vec::with_capacity(batch_size * max_len);
        let mut token_type_ids_vec = Vec::with_capacity(batch_size * max_len);

        for encoding in &encodings {
            let ids: Vec<u32> = encoding.get_ids().iter().take(MAX_SEQ_LEN).copied().collect();
            let mask: Vec<u32> = encoding.get_attention_mask().iter().take(MAX_SEQ_LEN).copied().collect();
            let types: Vec<u32> = encoding.get_type_ids().iter().take(MAX_SEQ_LEN).copied().collect();

            let mut padded_ids = ids;
            let mut padded_mask = mask;
            let mut padded_types = types;

            padded_ids.resize(max_len, 0);
            padded_mask.resize(max_len, 0);
            padded_types.resize(max_len, 0);

            input_ids_vec.extend(padded_ids);
            attention_mask_vec.extend(padded_mask);
            token_type_ids_vec.extend(padded_types);
        }

        let input_ids = Tensor::from_vec(input_ids_vec, (batch_size, max_len), &self.device)?;
        let attention_mask = Tensor::from_vec(attention_mask_vec, (batch_size, max_len), &self.device)?;
        let token_type_ids = Tensor::from_vec(token_type_ids_vec, (batch_size, max_len), &self.device)?;

        // Run model forward pass
        let output = self.model.forward(&input_ids, &token_type_ids, Some(&attention_mask))?;

        // Get [CLS] token output for each item in batch (first token, first hidden dim)
        let cls_outputs = output.i((.., 0, 0))?;  // Shape: [batch_size]
        let raw_scores: Vec<f32> = cls_outputs.to_vec1()?;

        // Apply sigmoid to all scores
        let scores: Vec<f32> = raw_scores.iter()
            .map(|&s| 1.0 / (1.0 + (-s).exp()))
            .collect();

        Ok(scores)
    }

    /// Rerank search results and return sorted by reranker score
    pub fn rerank_results<T: Clone>(
        &self,
        query: &str,
        results: Vec<(T, String)>, // (item, content)
        top_k: usize,
    ) -> Result<Vec<(T, f32)>> {
        if results.is_empty() {
            return Ok(vec![]);
        }

        let documents: Vec<String> = results.iter().map(|(_, content)| content.clone()).collect();
        let scores = self.rerank(query, &documents)?;

        // Combine items with scores
        let mut scored: Vec<(T, f32)> = results
            .into_iter()
            .zip(scores)
            .map(|((item, _), score)| (item, score))
            .collect();

        // Sort by score descending
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Return top K
        Ok(scored.into_iter().take(top_k).collect())
    }
}
