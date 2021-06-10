// Copyright 2016 Google Inc.
// Adapted from https://github.com/google/sentencepiece/blob/master/src/bpe_model.cc
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
use crate::tokenizer::tokenization_utils::{is_punctuation, is_whitespace};
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

pub struct SentencePieceBpeModel {
    bpe_ranks: BpeMergeVocab,
}

impl SentencePieceBpeModel {
    pub fn from_file(path: &str) -> Result<SentencePieceBpeModel, TokenizerError> {
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
        Ok(SentencePieceBpeModel { bpe_ranks })
    }

    pub fn tokenize_to_tokens(&self, initial_token: TokenRef) -> Vec<Token> {
        let mut sub_tokens = Vec::new();
        if initial_token.mask != Mask::Special && initial_token.mask != Mask::Unknown {
            let mut agenda: BinaryHeap<SymbolPair> = BinaryHeap::new();

            // Pre-populate symbols
            let mut symbols = Vec::with_capacity(initial_token.text.len());
            for (character_index, (character_start, character)) in
                initial_token.text.char_indices().enumerate()
            {
                symbols.push(Symbol {
                    start_byte: character_start,
                    end_byte: character_start + character.len_utf8(),
                    start_offset: character_index,
                    end_offset: character_index + 1,
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
                        start_byte: symbol_pair.left.start_byte,
                        end_byte: symbol_pair.right.end_byte,
                        start_offset: symbol_pair.left.start_offset,
                        end_offset: symbol_pair.right.end_offset,
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
                sub_tokens.push(Token {
                    text: initial_token.text[symbol.start_byte..symbol.end_byte].to_string(),
                    offset: Offset {
                        begin: symbol.start_offset as OffsetSize + initial_token.offset.begin,
                        end: symbol.end_offset as OffsetSize + initial_token.offset.begin,
                    },
                    reference_offsets: initial_token.reference_offsets
                        [symbol.start_offset..symbol.end_offset]
                        .to_vec(),
                    mask: Default::default(),
                })
            }
        } else {
            sub_tokens.push(initial_token.to_owned().clone());
        }
        self.populate_masks(sub_tokens.as_mut_slice(), '\u{2581}');
        sub_tokens
    }

    fn maybe_add_new_symbol_pair(
        &self,
        left: Symbol,
        right: Symbol,
        text_reference: &str,
        mut agenda: BinaryHeap<SymbolPair>,
    ) -> BinaryHeap<SymbolPair> {
        let merged_str = &text_reference[left.start_byte..right.end_byte];
        if let Some(&score) = self.bpe_ranks.values.get(merged_str) {
            agenda.push(SymbolPair { left, right, score })
        }
        agenda
    }

    /// Populates the `mask` field for a sequence of sub-tokens generated by a SentencePiece model.
    /// These masks are not generated as part of the standard unigram decomposition and must be added
    /// afterwards. Mutates the tokens in-place.
    ///
    /// # Arguments
    /// - tokens (`&mut [Token]`): tokens to get the masks from
    /// - whitespace_char (`char`): whitespace character to identify whether a token is a continuation token or not.
    ///
    /// # Example
    /// ```no_run
    /// use rust_tokenizers::vocab::SentencePieceBpeModel;
    /// use rust_tokenizers::TokenRef;
    /// let path = "path/to/spiece.model";
    /// let sentence_piece_model = SentencePieceBpeModel::from_file(path).unwrap();
    ///
    /// let token = TokenRef::new("hello", &[0, 1, 2, 3]);
    /// let mut sub_tokens = sentence_piece_model.tokenize_to_tokens(token);
    /// let sub_tokens_with_masks = sentence_piece_model.populate_masks(&mut sub_tokens, ' ');
    /// ```
    pub fn populate_masks(&self, tokens: &mut [Token], whitespace_token: char) {
        let mut previous_mask = Mask::None;
        for token in tokens {
            if token.text.chars().count() == 1 {
                let first_char = match token.text.chars().last() {
                    Some(value) => value,
                    None => {
                        token.mask = Mask::Unknown;
                        previous_mask = Mask::Unknown;
                        continue;
                    }
                };
                if is_punctuation(&first_char) {
                    token.mask = Mask::Punctuation;
                    previous_mask = Mask::Punctuation;
                    continue;
                }
                if is_whitespace(&first_char) {
                    token.mask = Mask::Whitespace;
                    previous_mask = Mask::Punctuation;
                    continue;
                }
            }
            if !token.text.starts_with(whitespace_token)
                & !(previous_mask == Mask::Punctuation)
                & !(previous_mask == Mask::Whitespace)
            {
                token.mask = Mask::Continuation;
                previous_mask = Mask::Continuation;
            } else {
                previous_mask = Mask::None;
            }
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct Symbol {
    start_byte: usize,
    end_byte: usize,
    start_offset: usize,
    end_offset: usize,
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
            .then_with(|| other.left.start_byte.cmp(&self.left.start_byte))
    }
}

impl PartialOrd for SymbolPair {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
