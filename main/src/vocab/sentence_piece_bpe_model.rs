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
use std::ops::Index;

#[derive(Debug, Clone)]
pub struct BpeMergeVocab {
    pub values: HashMap<String, i64>,
}

/// # SentencePiece BPE Model
/// Model for SentencePiece BPE tokenizer.
/// This model performs SentencePiece BPE decomposition using a priority queue and consecutive merges.
///
/// Expects a SentencePiece protobuf file when created from file.
pub struct SentencePieceBpeModel {
    bpe_ranks: BpeMergeVocab,
}

impl SentencePieceBpeModel {
    /// Creates a SentencePiece BPE Model from a protobuf file.
    ///
    /// # Example
    /// ```no_run
    /// use rust_tokenizers::vocab::SentencePieceBpeModel;
    /// let path = "path/to/spiece.model";
    ///
    /// let sentence_piece_model = SentencePieceBpeModel::from_file(path);
    /// ```
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

    /// Tokenizes an input sequence into an array of Tokens by merging adjacent symbols present
    /// in the merges list.
    ///
    /// # Example
    /// ```no_run
    /// use rust_tokenizers::vocab::SentencePieceBpeModel;
    /// use rust_tokenizers::TokenRef;
    /// let path = "path/to/spiece.model";
    ///
    /// let sentence_piece_bpe_model = SentencePieceBpeModel::from_file(path).unwrap();
    /// let token = TokenRef::new("hello", &[0, 1, 2, 3]);
    /// let tokenized_output = sentence_piece_bpe_model.tokenize_to_tokens(token);
    /// ```
    pub fn tokenize_to_tokens(&self, initial_token: TokenRef) -> Vec<Token> {
        let mut sub_tokens = Vec::new();
        if initial_token.mask != Mask::Special && initial_token.mask != Mask::Unknown {
            let mut agenda: BinaryHeap<SymbolPair> = BinaryHeap::new();

            // Pre-populate symbols
            let mut symbols = SymbolList::from(initial_token);

            // Pre-populate priority queue with bi-grams
            for symbol_index in 1..symbols.len() {
                self.maybe_add_pair(
                    symbol_index as isize - 1,
                    symbol_index as isize,
                    initial_token.text,
                    &symbols,
                    &mut agenda,
                );
            }

            while let Some(symbol_pair) = agenda.pop() {
                let left_symbol_index = symbol_pair.left;
                let right_symbol_index = symbol_pair.right;
                if left_symbol_index != -1 && right_symbol_index != -1 {
                    let new_symbol = symbols.merge_symbols(
                        left_symbol_index as usize,
                        right_symbol_index as usize,
                        symbol_pair.pair_size,
                    );
                    if let Some(new_symbol) = new_symbol {
                        self.maybe_add_pair(
                            new_symbol.prev,
                            left_symbol_index,
                            initial_token.text,
                            &symbols,
                            &mut agenda,
                        );
                        self.maybe_add_pair(
                            left_symbol_index,
                            new_symbol.next,
                            initial_token.text,
                            &symbols,
                            &mut agenda,
                        );
                    }
                }
            }
            for symbol in symbols.into_iter().flatten() {
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
            sub_tokens.push(initial_token.to_owned());
        }
        self.populate_masks(sub_tokens.as_mut_slice(), '\u{2581}');
        sub_tokens
    }

    fn maybe_add_pair(
        &self,
        left_symbol_index: isize,
        right_symbol_index: isize,
        input_text: &str,
        symbols: &SymbolList,
        agenda: &mut BinaryHeap<SymbolPair>,
    ) {
        if left_symbol_index != -1 && right_symbol_index != -1 {
            if let (Some(left_symbol), Some(right_symbol)) = (
                symbols[left_symbol_index as usize],
                symbols[right_symbol_index as usize],
            ) {
                let merged_text = &input_text[left_symbol.start_byte..right_symbol.end_byte];
                if let Some(&score) = self.bpe_ranks.values.get(merged_text) {
                    agenda.push(SymbolPair {
                        left: left_symbol_index,
                        right: right_symbol_index,
                        score,
                        pair_size: left_symbol.size + right_symbol.size,
                    })
                }
            }
        }
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
struct Symbol {
    start_byte: usize,
    end_byte: usize,
    start_offset: usize,
    end_offset: usize,
    prev: isize,
    next: isize,
    size: usize,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
struct SymbolPair {
    left: isize,
    right: isize,
    score: i64,
    pair_size: usize,
}

impl Ord for SymbolPair {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .score
            .cmp(&self.score)
            .then_with(|| other.left.cmp(&self.left))
    }
}

impl PartialOrd for SymbolPair {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

struct SymbolList {
    symbols: Vec<Option<Symbol>>,
}

impl Index<usize> for SymbolList {
    type Output = Option<Symbol>;

    fn index(&self, index: usize) -> &Option<Symbol> {
        self.symbols.index(index)
    }
}

impl IntoIterator for SymbolList {
    type Item = Option<Symbol>;
    type IntoIter = <Vec<Option<Symbol>> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.symbols.into_iter()
    }
}

impl From<TokenRef<'_>> for SymbolList {
    fn from(token: TokenRef) -> Self {
        let mut symbols = Vec::with_capacity(token.text.len());

        for (index, (character_start, character)) in token.text.char_indices().enumerate() {
            let next = if index == token.text.char_indices().count() - 1 {
                -1
            } else {
                (index + 1) as isize
            };
            symbols.push(Some(Symbol {
                start_byte: character_start,
                end_byte: character_start + character.len_utf8(),
                start_offset: index,
                end_offset: index + 1,
                prev: index as isize - 1,
                next,
                size: 1,
            }));
        }
        Self { symbols }
    }
}

impl SymbolList {
    pub fn len(&self) -> usize {
        self.symbols.len()
    }

    pub fn merge_symbols(
        &mut self,
        symbol_1_index: usize,
        symbol_2_index: usize,
        size_validation: usize,
    ) -> Option<Symbol> {
        if let (Some(left_symbol), Some(right_symbol)) =
            (self[symbol_1_index], self[symbol_2_index])
        {
            if left_symbol.size + right_symbol.size != size_validation {
                return None;
            }
            if right_symbol.next != -1 {
                if let Some(next_next) = self.symbols.get_mut(right_symbol.next as usize).unwrap() {
                    next_next.prev = symbol_1_index as isize;
                }
            }
            let new_symbol = Symbol {
                start_byte: left_symbol.start_byte,
                end_byte: right_symbol.end_byte,
                start_offset: left_symbol.start_offset,
                end_offset: right_symbol.end_offset,
                prev: left_symbol.prev,
                next: right_symbol.next,
                size: left_symbol.size + right_symbol.size,
            };
            self.symbols[symbol_2_index] = None;
            self.symbols[symbol_1_index] = Some(new_symbol);
            Some(new_symbol)
        } else {
            None
        }
    }
}
