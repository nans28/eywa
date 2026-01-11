//! Initialization flow for Eywa
//!
//! Handles first-run setup and model selection.

use crate::config::{Config, DevicePreference, EmbeddingModelConfig, RerankerModelConfig};
use anyhow::Result;
use std::io::{self, Write};

/// Result of running the init flow
#[derive(Debug)]
pub enum InitResult {
    /// User completed init with this config
    Configured(Config),
    /// User cancelled the init
    Cancelled,
}

/// Run the interactive init flow
pub fn run_init(existing_config: Option<&Config>) -> Result<InitResult> {
    let is_reinit = existing_config.is_some();

    if is_reinit {
        println!("\nCurrent configuration:");
        if let Some(config) = existing_config {
            println!("  Embedding: {}", config.embedding_model.name);
            println!("  Reranker:  {}", config.reranker_model.name);
        }
        println!();
    }

    // Show options
    let default_embed = EmbeddingModelConfig::default();
    let default_rerank = RerankerModelConfig::default();
    println!("[D] Default - {} ({}MB) + {} ({}MB)",
        default_embed.name,
        default_embed.size_mb,
        default_rerank.name,
        default_rerank.size_mb
    );
    println!("[C] Custom  - Choose your models");
    println!();

    print!("Choice [D/c]: ");
    io::stdout().flush()?;

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    let input = input.trim().to_lowercase();

    let config = if input == "c" || input == "custom" {
        run_custom_selection(existing_config)?
    } else {
        Config::default()
    };

    // Check if embedding model changed (requires reindex)
    let needs_reindex = if let Some(existing) = existing_config {
        existing.embedding_model != config.embedding_model
    } else {
        false
    };

    if needs_reindex {
        println!();
        println!("\x1b[33m⚠\x1b[0m  Embedding model changed. This requires reindexing.");
        println!("    All documents will be re-chunked and re-embedded.");
        println!();
        print!("Continue? [y/N]: ");
        io::stdout().flush()?;

        let mut confirm = String::new();
        io::stdin().read_line(&mut confirm)?;
        let confirm = confirm.trim().to_lowercase();

        if confirm != "y" && confirm != "yes" {
            return Ok(InitResult::Cancelled);
        }
    }

    // Save config
    config.save()?;

    Ok(InitResult::Configured(config))
}

/// Run custom model selection
fn run_custom_selection(existing_config: Option<&Config>) -> Result<Config> {
    let embedding_model = select_embedding_model(existing_config)?;
    let reranker_model = select_reranker_model(existing_config)?;

    Ok(Config {
        embedding_model,
        reranker_model,
        device: DevicePreference::default(),
        version: 2,
    })
}

/// Select embedding model interactively
fn select_embedding_model(existing_config: Option<&Config>) -> Result<EmbeddingModelConfig> {
    println!();
    println!("Embedding model:");

    let models = EmbeddingModelConfig::curated_models();
    let current_id = existing_config.map(|c| &c.embedding_model.id);

    for (i, model) in models.iter().enumerate() {
        let current_marker = if Some(&model.id) == current_id { " ← current" } else { "" };
        println!("  [{}] {} ({}MB, {} dims){}",
            i + 1,
            model.name,
            model.size_mb,
            model.dimensions,
            current_marker
        );
    }
    println!();

    let default_idx = current_id
        .and_then(|id| models.iter().position(|m| &m.id == id))
        .unwrap_or(0);

    print!("Choice [{}]: ", default_idx + 1);
    io::stdout().flush()?;

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    let input = input.trim();

    if input.is_empty() {
        return Ok(models[default_idx].clone());
    }

    match input.parse::<usize>() {
        Ok(n) if n >= 1 && n <= models.len() => Ok(models[n - 1].clone()),
        _ => {
            println!("Invalid selection, using default.");
            Ok(models[default_idx].clone())
        }
    }
}

/// Select reranker model interactively
fn select_reranker_model(existing_config: Option<&Config>) -> Result<RerankerModelConfig> {
    println!();
    println!("Reranker model:");

    let models = RerankerModelConfig::curated_models();
    let current_id = existing_config.map(|c| &c.reranker_model.id);

    for (i, model) in models.iter().enumerate() {
        let current_marker = if Some(&model.id) == current_id { " ← current" } else { "" };
        println!("  [{}] {} ({}MB){}",
            i + 1,
            model.name,
            model.size_mb,
            current_marker
        );
    }
    println!();

    let default_idx = current_id
        .and_then(|id| models.iter().position(|m| &m.id == id))
        .unwrap_or(0);

    print!("Choice [{}]: ", default_idx + 1);
    io::stdout().flush()?;

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    let input = input.trim();

    if input.is_empty() {
        return Ok(models[default_idx].clone());
    }

    match input.parse::<usize>() {
        Ok(n) if n >= 1 && n <= models.len() => Ok(models[n - 1].clone()),
        _ => {
            println!("Invalid selection, using default.");
            Ok(models[default_idx].clone())
        }
    }
}

/// Display status information
pub fn show_status(config: &Config, sources: usize, documents: usize, chunks: usize) {
    println!("Eywa v{} - The memory your team never loses\n",
        env!("CARGO_PKG_VERSION")
    );

    println!("Status:");
    println!("  Sources:   {}", sources);
    println!("  Documents: {}", documents);
    println!("  Chunks:    {}", chunks);
    println!();

    println!("Models:");
    println!("  Embedding: {}", config.embedding_model.name);
    println!("  Reranker:  {}", config.reranker_model.name);
    println!();

    println!("Run 'eywa --help' for commands.");
}

/// Display first-run welcome message
pub fn show_welcome() {
    println!("Eywa v{} - The memory your team never loses\n",
        env!("CARGO_PKG_VERSION")
    );
    println!("First run detected. Let's set you up.\n");
}
