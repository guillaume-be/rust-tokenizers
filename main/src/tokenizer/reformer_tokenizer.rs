// Copyright 2020 The Trax Authors and The HuggingFace Inc. team.
// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
// Copyright 2020 Guillaume Becquin
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
use crate::tokenizer::tokenization_utils::{
    bpe, clean_text, decompose_nfkc, fix_mask, is_whitespace, lowercase, split_on_bpe_pairs,
    split_on_special_tokens, whitespace_tokenize, BpeCache,
};
use crate::tokenizer::{MultiThreadedTokenizer, Tokenizer};
use crate::vocab::{BpePairVocab, ReformerVocab, Vocab};
use crate::Mask;
use std::collections::HashMap;
use std::sync::RwLock;

/// # Reformer tokenizer
pub struct ReformerTokenizer {
    vocab: ReformerVocab,
    bpe_ranks: BpePairVocab,
    cache: BpeCache,
    lower_case: bool,
}

impl ReformerTokenizer {
    /// Create a new instance of a `ReformerTokenizer`
    /// Expects a SentencePiece protobuf file as an input.
    ///
    /// # Parameters
    /// - path (`&str`): path to the SentencePiece model file
    /// - lower_case (`bool`): flag indicating if the text should be lower-cased as part of the tokenization
    /// - strip_accents (`bool`): flag indicating if accents should be stripped from the text
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_tokenizers::tokenizer::{SentencePieceTokenizer, Tokenizer};
    /// let lower_case = false;
    /// let tokenizer = SentencePieceTokenizer::from_file("path/to/vocab/file", lower_case).unwrap();
    /// ```
    pub fn from_file(path: &str, lower_case: bool) -> Result<ReformerTokenizer, TokenizerError> {
        let vocab = ReformerVocab::from_file(path)?;
        let bpe_ranks = BpePairVocab::from_sentencepiece_file(path)?;
        let cache = RwLock::new(HashMap::new());
        Ok(ReformerTokenizer {
            vocab,
            bpe_ranks,
            cache,
            lower_case,
        })
    }
}

impl Tokenizer<ReformerVocab> for ReformerTokenizer {
    fn vocab(&self) -> &ReformerVocab {
        &self.vocab
    }

    fn tokenize_to_tokens(&self, text: TokenRef) -> Vec<Token> {
        let mut tokens = split_on_special_tokens(text, &self.vocab)
            .into_iter()
            .map(whitespace_tokenize)
            .flatten()
            .map(|token| token.to_owned())
            .collect::<Vec<Token>>();

        let mut sub_tokens = Vec::new();
        for token in tokens.iter_mut() {
            decompose_nfkc(token);
            clean_text(token, true);
            if !token.text.is_empty() {
                token.text = token.text.replace(|c: char| is_whitespace(&c), "\u{2581}");
                if !token.text.starts_with('\u{2581}') {
                    token.text.insert(0, '\u{2581}');
                    token
                        .reference_offsets
                        .insert(0, token.reference_offsets[0]);
                };

                if token.mask != Mask::Special && token.mask != Mask::Unknown {
                    if self.lower_case {
                        lowercase(token);
                    }
                    sub_tokens.extend(split_on_bpe_pairs(
                        token.as_ref(),
                        bpe,
                        &self.bpe_ranks,
                        &self.cache,
                        false,
                    ));
                } else {
                    sub_tokens.push(token.to_owned());
                }
            }

            //     Consolidate consecutive unknown tokens
            let mut prev_is_unk = false;
            let mut indices_to_remove = vec![];
            for (index, sub_token) in sub_tokens.iter_mut().enumerate() {
                if self.vocab.values.get(&sub_token.text).is_none() {
                    sub_token.mask = Mask::Unknown;
                }
                if sub_token.mask == Mask::Unknown {
                    if prev_is_unk {
                        indices_to_remove.push(index);
                    }
                    prev_is_unk = true;
                } else {
                    prev_is_unk = false;
                }
            }
            for index_to_remove in indices_to_remove.into_iter().rev() {
                sub_tokens.remove(index_to_remove);
            }
        }

        fix_mask(&mut sub_tokens);
        sub_tokens
    }

    fn convert_tokens_to_string(&self, tokens: Vec<String>) -> String {
        tokens
            .into_iter()
            .map(|v| v.replace('\u{2581}', " "))
            .collect::<Vec<String>>()
            .join("")
    }
}

impl MultiThreadedTokenizer<ReformerVocab> for ReformerTokenizer {}
