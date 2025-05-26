# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

rust-tokenizers is a high-performance tokenization library for Rust that implements tokenizers for modern transformer-based language models. It provides both a Rust library and Python bindings.

## Architecture

### Core Components

1. **Tokenizers** (`main/src/tokenizer/`):
   - WordPiece: BERT, DistilBERT, FNet
   - BPE: GPT, GPT2, RoBERTa, CTRL, DeBERTa  
   - SentencePiece: ALBERT, XLMRoBERTa, XLNet, T5, Marian, Reformer, DeBERTa V2
   - Base trait: `BaseTokenizer` in `base_tokenizer.rs`

2. **Vocabularies** (`main/src/vocab/`):
   - Each tokenizer has a corresponding vocabulary implementation
   - BPE vocabularies include merge rules
   - SentencePiece vocabularies use protobuf models

3. **Python Bindings** (`python-bindings/`):
   - Uses PyO3 to expose Rust tokenizers to Python
   - Each tokenizer wrapped with `Py` prefix (e.g., `PyBertTokenizer`)

### Key Design Patterns

- Tokenizers implement the `Tokenizer` trait from `base_tokenizer.rs`
- Vocabularies implement the `Vocab` trait from `base_vocab.rs`
- Multi-threaded tokenization available for WordPiece tokenizers
- BPE tokenizers use shared cache (single-threaded only)

## Development Commands

### Rust Development

```bash
# Build with default features
cargo build

# Build with protobuf compilation
cargo build --features proto-compile

# Build with rustls instead of native TLS
cargo build --no-default-features --features rustls-tls

# Run all tests
cargo test

# Run a specific test
cargo test test_bert_uncased

# Format code
cargo fmt --manifest-path ./main/Cargo.toml

# Lint code
cargo clippy
```

Format all code before committing to git.

### Python Development

```bash
# Requires nightly Rust
cd python-bindings

# Install for development
pip install -e .

# Run Python tests
python -m pytest tests/
```

## Testing Patterns

Tests follow a consistent pattern:
1. Download vocabulary files using `cached-path`
2. Create tokenizer instance
3. Test tokenization on various inputs
4. Verify token IDs, offsets, and special tokens

Example test structure:
```rust
let vocab_path = download_test_files(...);
let tokenizer = BertTokenizer::from_file(&vocab_path, ...)?;
let encoding = tokenizer.encode(...);
assert_eq!(encoding.token_ids, expected_ids);
```

## Adding New Tokenizers

1. Create tokenizer implementation in `main/src/tokenizer/`
2. Create vocabulary implementation in `main/src/vocab/`
3. Add module exports in respective `mod.rs` files
4. Create test file in `main/tests/`
5. Add Python wrapper in `python-bindings/src/lib.rs`

## Important Notes

- Vocabulary files must be downloaded manually from HuggingFace Transformers
- SentencePiece models use the same `.model` proto files as the C++ library
- The library prioritizes performance and correctness
- All tokenizers support truncation strategies and special token handling

## Tokenizer References

- GPT-NeoX Tokenizer:
  - Key details about the GPTNeoX tokenizer:
    - The tokenizer is based on byte-level Byte-Pair-Encoding (BPE)
    - It allocates additional tokens to whitespace characters, making the model more suitable for certain tasks like code generation
    - It treats spaces like parts of the tokens (similar to SentencePiece)
    - The tokenizer implementation is available in HuggingFace's transformers library as GPTNeoXTokenizerFast
  - Reference the "Tokenization" section of the article "GPT-NeoX-20B: An Open-Source Autoregressive Language Model" from https://arxiv.org/pdf/2204.06745
