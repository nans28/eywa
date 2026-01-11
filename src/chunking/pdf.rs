//! PDF Chunker
//!
//! PDF text extraction and chunking using pdf_oxide.
//! Converts PDF → Markdown, then delegates to MarkdownChunker.

use super::{Chunk, Chunker, DocMetadata, MarkdownChunker};
use anyhow::Result;
use pdf_oxide::converters::ConversionOptions;
use std::path::Path;

/// Extract text from base64-encoded PDF content
/// Used for web uploads where PDF is sent as base64
pub fn extract_text_from_base64_pdf(base64_content: &str) -> Result<String> {
    use base64::Engine;

    // Decode base64
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(base64_content)
        .map_err(|e| anyhow::anyhow!("Failed to decode base64: {}", e))?;

    // Write to temp file (pdf_oxide requires file path)
    let temp_path = std::env::temp_dir().join(format!("eywa_pdf_{}.pdf", uuid::Uuid::new_v4()));
    std::fs::write(&temp_path, &bytes)
        .map_err(|e| anyhow::anyhow!("Failed to write temp PDF: {}", e))?;

    // Extract text
    let result = extract_text_from_pdf(&temp_path);

    // Clean up temp file
    let _ = std::fs::remove_file(&temp_path);

    result
}

/// Extract text from PDF file, converting all pages to Markdown
pub fn extract_text_from_pdf(pdf_path: &Path) -> Result<String> {
    let mut doc = pdf_oxide::PdfDocument::open(pdf_path)
        .map_err(|e| anyhow::anyhow!("Failed to open PDF: {}", e))?;

    let page_count = doc.page_count()
        .map_err(|e| anyhow::anyhow!("Failed to get page count: {}", e))?;

    let mut all_text = String::new();
    let options = ConversionOptions::default();

    for page_idx in 0..page_count {
        match doc.to_markdown(page_idx, &options) {
            Ok(markdown) => {
                if !all_text.is_empty() {
                    all_text.push_str("\n\n---\n\n"); // Page separator
                }
                all_text.push_str(&markdown);
            }
            Err(e) => {
                eprintln!("Warning: Failed to extract page {}: {}", page_idx + 1, e);
            }
        }
    }

    Ok(all_text)
}

/// PDF chunker (stub - delegates to markdown chunker)
pub struct PdfChunker {
    md_chunker: MarkdownChunker,
}

impl PdfChunker {
    pub fn new() -> Self {
        Self {
            md_chunker: MarkdownChunker::new(),
        }
    }
}

impl Default for PdfChunker {
    fn default() -> Self {
        Self::new()
    }
}

impl Chunker for PdfChunker {
    fn chunk(&self, content: &str, metadata: &DocMetadata) -> Vec<Chunk> {
        // Extracted PDF text is converted to Markdown by pdf_oxide
        // MarkdownChunker handles the hierarchical structure
        self.md_chunker.chunk(content, metadata)
    }

    fn supported_extensions(&self) -> &[&str] {
        &["pdf"]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_doc() -> DocMetadata {
        DocMetadata {
            document_id: "doc1".to_string(),
            source_id: "src1".to_string(),
            file_path: Some("document.pdf".to_string()),
        }
    }

    #[test]
    fn test_pdf_chunker_delegates_to_markdown() {
        let chunker = PdfChunker::new();
        let content = "# Document Title\n\nSome extracted PDF content here.";

        let chunks = chunker.chunk(content, &test_doc());

        // Should produce chunks like markdown chunker would
        assert!(!chunks.is_empty() || content.len() < super::super::MIN_CHUNK);
    }

    #[test]
    fn test_supported_extensions() {
        let chunker = PdfChunker::new();
        assert_eq!(chunker.supported_extensions(), &["pdf"]);
    }
}
