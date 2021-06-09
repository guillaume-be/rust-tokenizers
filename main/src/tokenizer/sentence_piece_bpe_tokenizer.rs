// Copyright 2016 Google Inc.
// Adapter from https://github.com/google/sentencepiece/blob/master/src/bpe_model.cc
// Copyright 2019-2021 Guillaume Becquin
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use crate::error::TokenizerError;
use crate::tokenizer::base_tokenizer::{Token, TokenRef};
use crate::vocab::sentencepiece_proto::sentencepiece_model::ModelProto;
use crate::{Mask, Offset, OffsetSize};
use hashbrown::HashMap;
use protobuf::Message;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::fs::File;
use std::io::Read;

#[derive(Debug, Clone)]
pub struct BpeMergeVocab {
    pub values: HashMap<String, i64>,
}

pub struct SentencePieceBpeTokenizer {
    bpe_ranks: BpeMergeVocab,
}

impl SentencePieceBpeTokenizer {
    pub fn from_file(path: &str) -> Result<SentencePieceBpeTokenizer, TokenizerError> {
        let mut f = File::open(path).map_err(|e| {
            TokenizerError::FileNotFound(format!("{} vocabulary file not found :{}", path, e))
        })?;
        let mut contents = Vec::new();
        let proto = match f.read_to_end(&mut contents) {
            Ok(_) => match ModelProto::parse_from_bytes(contents.as_slice()) {
                Ok(proto_value) => proto_value,
                Err(e) => {
                    return Err(TokenizerError::VocabularyParsingError(e.to_string()));
                }
            },
            Err(e) => {
                return Err(TokenizerError::VocabularyParsingError(e.to_string()));
            }
        };

        let mut values = HashMap::new();
        for (idx, piece) in proto.get_pieces().iter().enumerate() {
            values.insert(piece.get_piece().to_owned(), idx as i64);
        }
        let bpe_ranks = BpeMergeVocab { values };
        Ok(SentencePieceBpeTokenizer { bpe_ranks })
    }

    pub fn vocab(&self) -> &BpeMergeVocab {
        &self.bpe_ranks
    }

    pub fn tokenize_to_tokens(&self, initial_token: TokenRef) -> Vec<Token> {
        let mut sub_tokens = Vec::new();
        if initial_token.mask != Mask::Special && initial_token.mask != Mask::Unknown {
            let mut agenda: BinaryHeap<SymbolPair> = BinaryHeap::new();

            // Pre-populate symbols
            let mut symbols = Vec::with_capacity(initial_token.text.len());
            for (character_start, character) in initial_token.text.char_indices() {
                symbols.push(Symbol {
                    start: character_start,
                    end: character_start + character.len_utf8(),
                });
            }

            // Pre-populate priority queue with bi-grams
            for symbol_pair in symbols.windows(2) {
                agenda = self.maybe_add_new_symbol_pair(
                    symbol_pair[0],
                    symbol_pair[1],
                    initial_token.text,
                    agenda,
                );
            }

            while let Some(symbol_pair) = agenda.pop() {
                let left_index = symbols.iter().position(|x| x == &symbol_pair.left);
                let right_index = symbols.iter().position(|x| x == &symbol_pair.right);

                if left_index.is_none() | right_index.is_none() {
                    continue;
                }
                let left_index = left_index.unwrap();
                let right_index = right_index.unwrap();
                symbols.remove(right_index);
                symbols.remove(left_index);
                symbols.insert(
                    left_index,
                    Symbol {
                        start: symbol_pair.left.start,
                        end: symbol_pair.right.end,
                    },
                );
                if left_index > 0 {
                    agenda = self.maybe_add_new_symbol_pair(
                        symbols[left_index - 1],
                        symbols[left_index],
                        initial_token.text,
                        agenda,
                    );
                }
                if right_index < symbols.len() {
                    agenda = self.maybe_add_new_symbol_pair(
                        symbols[left_index],
                        symbols[left_index + 1],
                        initial_token.text,
                        agenda,
                    );
                }
            }
            for symbol in symbols {
                let begin = symbol.start as OffsetSize + initial_token.offset.begin;
                let end = symbol.end as OffsetSize + initial_token.offset.begin;
                sub_tokens.push(Token {
                    text: initial_token.text[symbol.start..symbol.end].to_string(),
                    offset: Offset { begin, end },
                    reference_offsets: (begin..end).collect::<Vec<_>>(),
                    mask: Default::default(),
                })
            }
        } else {
            sub_tokens.push(initial_token.to_owned().clone());
        }
        sub_tokens
    }

    fn maybe_add_new_symbol_pair(
        &self,
        left: Symbol,
        right: Symbol,
        text_reference: &str,
        mut agenda: BinaryHeap<SymbolPair>,
    ) -> BinaryHeap<SymbolPair> {
        let merged_str = &text_reference[left.start..right.end];
        if let Some(&score) = self.bpe_ranks.values.get(merged_str) {
            agenda.push(SymbolPair { left, right, score })
        }
        agenda
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct Symbol {
    start: usize,
    end: usize,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct SymbolPair {
    left: Symbol,
    right: Symbol,
    score: i64,
}

impl Ord for SymbolPair {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .score
            .cmp(&self.score)
            .then_with(|| other.left.start.cmp(&self.left.start))
    }
}

impl PartialOrd for SymbolPair {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
