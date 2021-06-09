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

//!# Tokenizers
//!
//! This module contains the tokenizers to split an input text in a sequence of tokens.
//! These rely on the vocabularies for defining the subtokens a given word should be decomposed to.
//! There are 3 main classes of tokenizers implemented in this crate:
//! - WordPiece tokenizers
//!     - BERT
//!     - DistilBERT
//! - Byte-Pair Encoding tokenizers:
//!     - GPT
//!     - GPT2
//!     - RoBERTa
//!     - CTRL
//! - SentencePiece (Unigram) tokenizers:
//!     - SentencePiece
//!     - ALBERT
//!     - XLMRoBERTa
//!     - XLNet
//!     - T5
//!     - Marian
//!     - Reformer
//!
//! All tokenizers are `Send`, `Sync` and support multi-threaded tokenization and encoding.

mod albert_tokenizer;
pub(crate) mod base_tokenizer;
mod bert_tokenizer;
mod constants;
mod ctrl_tokenizer;
mod gpt2_tokenizer;
mod m2m100_tokenizer;
mod marian_tokenizer;
mod mbart50_tokenizer;
mod openai_gpt_tokenizer;
mod pegasus_tokenizer;
mod prophetnet_tokenizer;
mod reformer_tokenizer;
mod roberta_tokenizer;
mod sentence_piece_bpe_tokenizer;
mod sentence_piece_tokenizer;
mod t5_tokenizer;
pub(crate) mod tokenization_utils;
mod xlm_roberta_tokenizer;
mod xlnet_tokenizer;

pub use albert_tokenizer::AlbertTokenizer;
pub use base_tokenizer::{BaseTokenizer, MultiThreadedTokenizer, Tokenizer, TruncationStrategy};
pub use bert_tokenizer::BertTokenizer;
pub use ctrl_tokenizer::CtrlTokenizer;
pub use gpt2_tokenizer::Gpt2Tokenizer;
pub use m2m100_tokenizer::M2M100Tokenizer;
pub use marian_tokenizer::MarianTokenizer;
pub use mbart50_tokenizer::MBart50Tokenizer;
pub use openai_gpt_tokenizer::OpenAiGptTokenizer;
pub use pegasus_tokenizer::PegasusTokenizer;
pub use prophetnet_tokenizer::ProphetNetTokenizer;
pub use reformer_tokenizer::ReformerTokenizer;
pub use roberta_tokenizer::RobertaTokenizer;
pub use sentence_piece_tokenizer::SentencePieceTokenizer;
pub use t5_tokenizer::T5Tokenizer;
pub use tokenization_utils::truncate_sequences;
pub use xlm_roberta_tokenizer::XLMRobertaTokenizer;
pub use xlnet_tokenizer::XLNetTokenizer;
