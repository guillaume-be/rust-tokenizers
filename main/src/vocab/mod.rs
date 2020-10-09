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

mod albert_vocab;
pub(crate) mod base_vocab;
mod bert_vocab;
pub(crate) mod bpe_vocab;
mod gpt2_vocab;
mod marian_vocab;
mod openai_gpt_vocab;
mod roberta_vocab;
mod sentence_piece_vocab;
mod sentencepiece_proto;
mod t5_vocab;
mod xlm_roberta_vocab;
mod xlnet_vocab;

pub use albert_vocab::AlbertVocab;
pub use base_vocab::{BaseVocab, Vocab};
pub use bert_vocab::BertVocab;
pub use gpt2_vocab::Gpt2Vocab;
pub use marian_vocab::MarianVocab;
pub use openai_gpt_vocab::OpenAiGptVocab;
pub use roberta_vocab::RobertaVocab;
pub use sentence_piece_vocab::{SentencePieceModel, SentencePieceVocab};
pub use t5_vocab::T5Vocab;
pub use xlm_roberta_vocab::XLMRobertaVocab;
pub use xlnet_vocab::XLNetVocab;
