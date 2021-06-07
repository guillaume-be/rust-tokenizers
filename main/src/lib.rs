// Copyright 2018 The HuggingFace Inc. team.
// Copyright 2019 Guillaume Becquin
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//!# High performance tokenizers for Rust
//!
//! This crate contains implementation of common tokenizers used in state-of-the-art language models.
//! It is usd as the reference tokenization crate of [rust-bert](https://docs.rs/rust-bert/), exposing modern transformer-based
//! models such as BERT, RoBERTa, GPT2, BART, XLNet...
//!
//! The following tokenizers have been implemented and validated against a Python reference implementation:
//! - Sentence Piece (unigram model)
//! - BERT
//! - DistilBERT
//! - RoBERTa
//! - GPT
//! - GPT2
//! - CTRL
//! - ProphetNet
//! - XLNet
//! - Pegasus
//! - MBart50
//!
//! The library is structured into vocabularies (for the encoding and decoding of the tokens and registration of special tokens)
//! and tokenizers (splitting the input text into tokens). Generally, a tokenizer will contain a reference vocabulary that may
//! be used as part of the tokenization process (for example, containing a list of subwords or merges).
//!
//! ## Usage example
//!
//! ```no_run
//! # fn main() -> anyhow::Result<()> {
//! use rust_tokenizers::adapters::Example;
//! use rust_tokenizers::tokenizer::{BertTokenizer, Tokenizer, TruncationStrategy};
//! use rust_tokenizers::vocab::{BertVocab, Vocab};
//! let vocab_path = "path/to/vocab";
//! let vocab = BertVocab::from_file(&vocab_path)?;
//!
//! let test_sentence = Example::new_from_string("This is a sample sentence to be tokenized");
//! let bert_tokenizer: BertTokenizer = BertTokenizer::from_existing_vocab(vocab, true, true);
//!
//! println!(
//!     "{:?}",
//!     bert_tokenizer.encode(
//!         &test_sentence.sentence_1,
//!         None,
//!         128,
//!         &TruncationStrategy::LongestFirst,
//!         0
//!     )
//! );
//! # Ok(())
//! # }
//! ```

pub mod tokenizer;
pub mod vocab;

pub mod adapters;
pub mod error;
pub use tokenizer::base_tokenizer::{
    ConsolidatableTokens, ConsolidatedTokenIterator, Mask, Offset, OffsetSize, Token,
    TokenIdsWithOffsets, TokenIdsWithSpecialTokens, TokenRef, TokenTrait, TokenizedInput,
    TokensWithOffsets,
};

#[macro_use]
extern crate lazy_static;
