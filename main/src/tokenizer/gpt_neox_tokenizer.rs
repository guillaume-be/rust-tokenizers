// Copyright 2022 EleutherAI and The HuggingFace Inc. team.
// Copyright 2024 Guillaume Becquin
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
use crate::tokenizer::constants::UNICODE_TO_BYTES;
use crate::tokenizer::tokenization_utils::{
    bpe, fix_mask, split_on_bpe_pairs, split_on_regex, split_on_special_tokens,
};
use crate::tokenizer::tokenization_utils::{lowercase, BpeCache};
use crate::tokenizer::{MultiThreadedTokenizer, Tokenizer};
use crate::vocab::base_vocab::SpecialTokenMap;
use crate::vocab::bpe_vocab::BpePairVocab;
use crate::vocab::{GptNeoXVocab, Vocab};
use crate::{Mask, Offset, Token, TokenRef};
use itertools::Itertools;
use regex::Regex;
use serde_json::Value;
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::iter::Iterator;
use std::path::Path;
use std::sync::RwLock;

/// # GPTNeoX tokenizer
/// GPTNeoX tokenizer performing:
/// - splitting on special characters
/// - whitespace splitting
/// - (optional) lower casing
/// - BPE tokenization
/// - (optional) adding BOS/EOS tokens
pub struct GptNeoXTokenizer {
    vocab: GptNeoXVocab,
    bpe_ranks: BpePairVocab,
    cache: BpeCache,
    pattern_tokenization: Regex,
    lower_case: bool,
    add_prefix_space: bool,
    add_bos_token: bool,
    add_eos_token: bool,
}

impl GptNeoXTokenizer {
    /// Create a new instance of a `GptNeoXTokenizer`
    /// Expects a vocabulary json file and a merges file as an input.
    ///
    /// # Parameters
    /// - vocab_path (`&str`): path to the vocabulary file
    /// - merges_path (`&str`): path to the merges file (use as part of the BPE encoding process)
    /// - lower_case (`bool`): flag indicating if the text should be lower-cased as part of the tokenization
    /// - add_prefix_space (`bool`): whether or not to add an initial space to the input
    /// - add_bos_token (`bool`): whether or not to add a BOS token at the start of sequences
    /// - add_eos_token (`bool`): whether or not to add an EOS token at the end of sequences
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_tokenizers::tokenizer::{GptNeoXTokenizer, Tokenizer};
    /// let lower_case = false;
    /// let add_prefix_space = false;
    /// let add_bos_token = false;
    /// let add_eos_token = false;
    /// let tokenizer = GptNeoXTokenizer::from_file(
    ///     "path/to/vocab/file",
    ///     "path/to/merges/file",
    ///     lower_case,
    ///     add_prefix_space,
    ///     add_bos_token,
    ///     add_eos_token
    /// ).unwrap();
    /// ```
    pub fn from_file<P: AsRef<Path>, M: AsRef<Path>>(
        vocab_path: P,
        merges_path: M,
        lower_case: bool,
        add_prefix_space: bool,
        add_bos_token: bool,
        add_eos_token: bool,
    ) -> Result<GptNeoXTokenizer, TokenizerError> {
        let vocab = GptNeoXVocab::from_file(vocab_path)?;
        let bpe_ranks = BpePairVocab::from_file(merges_path)?;
        let cache = RwLock::new(HashMap::new());
        let pattern_tokenization =
            Regex::new(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+")
                .unwrap();
        Ok(GptNeoXTokenizer {
            vocab,
            bpe_ranks,
            cache,
            pattern_tokenization,
            lower_case,
            add_prefix_space,
            add_bos_token,
            add_eos_token,
        })
    }

    /// Create a new instance of a `GptNeoXTokenizer`
    /// Expects a vocabulary json file and a merges file and special token mapping file as inputs.
    ///
    /// # Parameters
    /// - vocab_path (`&str`): path to the vocabulary file
    /// - merges_path (`&str`): path to the merges file (use as part of the BPE encoding process)
    /// - lower_case (`bool`): flag indicating if the text should be lower-cased as part of the tokenization
    /// - add_prefix_space (`bool`): whether or not to add an initial space to the input
    /// - add_bos_token (`bool`): whether or not to add a BOS token at the start of sequences
    /// - add_eos_token (`bool`): whether or not to add an EOS token at the end of sequences
    /// - special_token_mapping_path (`&str`): path to a special token mapping file to overwrite default special tokens
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_tokenizers::tokenizer::{GptNeoXTokenizer, Tokenizer};
    /// let lower_case = false;
    /// let add_prefix_space = false;
    /// let add_bos_token = false;
    /// let add_eos_token = false;
    /// let tokenizer = GptNeoXTokenizer::from_file_with_special_token_mapping(
    ///     "path/to/vocab/file",
    ///     "path/to/merges/file",
    ///     lower_case,
    ///     add_prefix_space,
    ///     add_bos_token,
    ///     add_eos_token,
    ///     "path/to/special/token/mapping/file",
    /// )
    /// .unwrap();
    /// ```
    pub fn from_file_with_special_token_mapping<V: AsRef<Path>, M: AsRef<Path>, S: AsRef<Path>>(
        vocab_path: V,
        merges_path: M,
        lower_case: bool,
        add_prefix_space: bool,
        add_bos_token: bool,
        add_eos_token: bool,
        special_token_mapping_path: S,
    ) -> Result<GptNeoXTokenizer, TokenizerError> {
        let vocab = GptNeoXVocab::from_file_with_special_token_mapping(
            vocab_path,
            special_token_mapping_path,
        )?;
        let bpe_ranks = BpePairVocab::from_file(merges_path)?;
        let cache = RwLock::new(HashMap::new());
        let pattern_tokenization =
            Regex::new(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+")
                .unwrap();
        Ok(GptNeoXTokenizer {
            vocab,
            bpe_ranks,
            cache,
            pattern_tokenization,
            lower_case,
            add_prefix_space,
            add_bos_token,
            add_eos_token,
        })
    }

    /// Create a new instance of a `GptNeoXTokenizer` from an existing vocabulary and merges
    ///
    /// # Parameters
    /// - vocab (`GptNeoXVocab`): GPTNeoX vocabulary
    /// - merges (`BpePairVocab`): BPE pairs vocabulary
    /// - lower_case (`bool`): flag indicating if the text should be lower-cased as part of the tokenization
    /// - add_prefix_space (`bool`): whether or not to add an initial space to the input
    /// - add_bos_token (`bool`): whether or not to add a BOS token at the start of sequences
    /// - add_eos_token (`bool`): whether or not to add an EOS token at the end of sequences
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_tokenizers::tokenizer::{GptNeoXTokenizer, Tokenizer};
    /// use rust_tokenizers::vocab::{BpePairVocab, GptNeoXVocab, Vocab};
    /// let lower_case = false;
    /// let add_prefix_space = false;
    /// let add_bos_token = false;
    /// let add_eos_token = false;
    /// let vocab = GptNeoXVocab::from_file("path/to/vocab/file").unwrap();
    /// let merges = BpePairVocab::from_file("path/to/merges/file").unwrap();
    ///
    /// let tokenizer = GptNeoXTokenizer::from_existing_vocab_and_merges(
    ///     vocab, merges, lower_case, add_prefix_space, add_bos_token, add_eos_token
    /// );
    /// ```
    pub fn from_existing_vocab_and_merges(
        vocab: GptNeoXVocab,
        merges: BpePairVocab,
        lower_case: bool,
        add_prefix_space: bool,
        add_bos_token: bool,
        add_eos_token: bool,
    ) -> GptNeoXTokenizer {
        let cache = RwLock::new(HashMap::new());
        let pattern_tokenization =
            Regex::new(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+")
                .unwrap();
        GptNeoXTokenizer {
            vocab,
            bpe_ranks: merges,
            cache,
            pattern_tokenization,
            lower_case,
            add_prefix_space,
            add_bos_token,
            add_eos_token,
        }
    }

    /// Create a new instance of a `GptNeoXTokenizer` from a HuggingFace tokenizer.json file
    ///
    /// # Parameters
    /// - tokenizer_json_path (`&str`): path to the HuggingFace tokenizer.json file
    /// - lower_case (`bool`): flag indicating if the text should be lower-cased as part of the tokenization
    /// - add_prefix_space (`bool`): whether or not to add an initial space to the input
    /// - add_bos_token (`bool`): whether or not to add a BOS token at the start of sequences
    /// - add_eos_token (`bool`): whether or not to add an EOS token at the end of sequences
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_tokenizers::tokenizer::{GptNeoXTokenizer, Tokenizer};
    /// let lower_case = false;
    /// let add_prefix_space = false;
    /// let add_bos_token = false;
    /// let add_eos_token = false;
    /// let tokenizer = GptNeoXTokenizer::from_tokenizer_json(
    ///     "path/to/tokenizer.json",
    ///     lower_case,
    ///     add_prefix_space,
    ///     add_bos_token,
    ///     add_eos_token
    /// ).unwrap();
    /// ```
    pub fn from_tokenizer_json<P: AsRef<Path>>(
        tokenizer_json_path: P,
        lower_case: bool,
        add_prefix_space: bool,
        add_bos_token: bool,
        add_eos_token: bool,
    ) -> Result<GptNeoXTokenizer, TokenizerError> {
        let file = File::open(tokenizer_json_path)
            .map_err(|e| TokenizerError::FileNotFound(e.to_string()))?;
        let reader = BufReader::new(file);
        let tokenizer_json: Value = serde_json::from_reader(reader)
            .map_err(|e| TokenizerError::FileNotFound(e.to_string()))?;

        // Extract model section
        let model = tokenizer_json.get("model").ok_or_else(|| {
            TokenizerError::FileNotFound("Missing 'model' section in tokenizer.json".to_string())
        })?;

        // Verify it's a BPE model
        let model_type = model
            .get("type")
            .and_then(|t| t.as_str())
            .ok_or_else(|| TokenizerError::FileNotFound("Missing model type".to_string()))?;

        if model_type != "BPE" {
            return Err(TokenizerError::FileNotFound(format!(
                "Expected BPE model, got {}",
                model_type
            )));
        }

        // Extract vocabulary
        let vocab_json = model
            .get("vocab")
            .ok_or_else(|| TokenizerError::FileNotFound("Missing vocab in model".to_string()))?;

        let vocab_map: HashMap<String, i64> = serde_json::from_value(vocab_json.clone())
            .map_err(|e| TokenizerError::FileNotFound(format!("Failed to parse vocab: {}", e)))?;

        // Extract merges
        let merges_json = model
            .get("merges")
            .ok_or_else(|| TokenizerError::FileNotFound("Missing merges in model".to_string()))?;

        let merges_vec: Vec<String> = serde_json::from_value(merges_json.clone())
            .map_err(|e| TokenizerError::FileNotFound(format!("Failed to parse merges: {}", e)))?;

        // Convert merges to BpePairVocab format
        let mut bpe_ranks = HashMap::new();
        for (idx, merge) in merges_vec.iter().enumerate() {
            let parts: Vec<&str> = merge.split(' ').collect();
            if parts.len() == 2 {
                bpe_ranks.insert((parts[0].to_string(), parts[1].to_string()), idx as i64);
            }
        }

        // Extract special tokens from added_tokens section
        let mut unk_token = "<|endoftext|>".to_string();
        let mut bos_token = None;
        let mut eos_token = None;
        let mut pad_token = None;

        if let Some(added_tokens) = tokenizer_json
            .get("added_tokens")
            .and_then(|t| t.as_array())
        {
            for token in added_tokens {
                if let (Some(content), Some(_id), Some(special)) = (
                    token.get("content").and_then(|c| c.as_str()),
                    token.get("id").and_then(|i| i.as_i64()),
                    token.get("special").and_then(|s| s.as_bool()),
                ) {
                    if special {
                        match content {
                            "<|endoftext|>" => {
                                unk_token = content.to_string();
                                bos_token = Some(content.to_string());
                                eos_token = Some(content.to_string());
                            }
                            "<|padding|>" => {
                                pad_token = Some(content.to_string());
                            }
                            _ => {}
                        }
                    }
                }
            }
        }

        let special_token_map = SpecialTokenMap {
            unk_token,
            pad_token,
            bos_token,
            sep_token: None,
            cls_token: None,
            eos_token,
            mask_token: None,
            additional_special_tokens: None,
        };

        // Create the vocabulary
        let vocab = GptNeoXVocab::from_values_and_special_token_map(vocab_map, special_token_map)?;
        let bpe_vocab = BpePairVocab { values: bpe_ranks };

        Ok(GptNeoXTokenizer::from_existing_vocab_and_merges(
            vocab,
            bpe_vocab,
            lower_case,
            add_prefix_space,
            add_bos_token,
            add_eos_token,
        ))
    }
}

impl Tokenizer<GptNeoXVocab> for GptNeoXTokenizer {
    fn vocab(&self) -> &GptNeoXVocab {
        &self.vocab
    }
    fn vocab_mut(&mut self) -> &mut GptNeoXVocab {
        &mut self.vocab
    }

    fn tokenize_with_offsets(&self, text: &str) -> crate::TokensWithOffsets {
        // Don't filter out whitespace-only input for GPTNeoX
        if text.is_empty() {
            return crate::TokensWithOffsets {
                tokens: vec![],
                offsets: vec![],
                reference_offsets: vec![],
                masks: vec![],
            };
        }
        let initial_offsets =
            (0..text.chars().count() as crate::OffsetSize).collect::<Vec<crate::OffsetSize>>();
        let initial_token = crate::TokenRef::new(text, &initial_offsets);
        let tokens = self.tokenize_to_tokens(initial_token);
        let length = tokens.len();
        let mut texts = Vec::with_capacity(length);
        let mut offsets = Vec::with_capacity(length);
        let mut original_positions = Vec::with_capacity(length);
        let mut masks = Vec::with_capacity(length);

        for token in tokens {
            texts.push(token.text);
            offsets.push(if !token.reference_offsets.is_empty() {
                Some(crate::Offset {
                    begin: *token.reference_offsets.first().unwrap(),
                    end: *token.reference_offsets.last().unwrap() + 1,
                })
            } else {
                None
            });
            original_positions.push(token.reference_offsets);
            masks.push(token.mask);
        }
        crate::TokensWithOffsets {
            tokens: texts,
            offsets,
            reference_offsets: original_positions,
            masks,
        }
    }

    fn tokenize_to_tokens(&self, initial_token: TokenRef) -> Vec<Token> {
        let mut initial_token = initial_token.to_owned();

        // Add prefix space if needed
        if self.add_prefix_space
            && !initial_token.text.is_empty()
            && !initial_token.text.starts_with(' ')
        {
            initial_token.text.insert(0, ' ');
            initial_token.reference_offsets.insert(0, 0);
        }

        let mut tokens = split_on_special_tokens(initial_token.as_ref(), &self.vocab)
            .into_iter()
            .map(|token| token.to_owned())
            .collect::<Vec<Token>>();

        let mut sub_tokens = Vec::new();
        for token in tokens.iter_mut() {
            if token.mask != Mask::Special && token.mask != Mask::Unknown {
                if self.lower_case {
                    lowercase(token);
                }

                // GPTNeoX uses BPE with ByteLevel encoding, but has special handling for spaces
                // The vocabulary contains both literal space tokens and ByteLevel encoded tokens
                let regex_tokens: Vec<_> =
                    split_on_regex(token.as_ref(), &self.pattern_tokenization);

                for (idx, token) in regex_tokens.iter().enumerate() {
                    // Check if the token exists as-is in the vocabulary (for literal whitespace tokens like spaces or tabs)
                    // But only for tokens that are pure whitespace - we don't want to bypass BPE for regular words
                    if self.vocab.values.contains_key(token.text)
                        && token.text.chars().all(|c| c.is_whitespace())
                    {
                        // This is a literal whitespace token (spaces, tabs, etc.)
                        sub_tokens.push(Token {
                            text: token.text.to_string(),
                            offset: token.offset,
                            reference_offsets: token.reference_offsets.to_vec(),
                            mask: token.mask,
                        });
                    } else if token.text == "\t\t" {
                        // Special case: double tab should be split into two separate tab tokens
                        // This matches the Python behavior
                        for i in 0..2 {
                            let offset_begin = token.offset.begin + i as u32;
                            let offset_end = offset_begin + 1;
                            sub_tokens.extend(split_on_bpe_pairs(
                                TokenRef {
                                    text: "\t",
                                    offset: Offset {
                                        begin: offset_begin,
                                        end: offset_end,
                                    },
                                    reference_offsets: &token.reference_offsets[i..i + 1],
                                    mask: token.mask,
                                },
                                bpe,
                                &self.bpe_ranks,
                                &self.cache,
                                true, // as_bytes = true for ByteLevel encoding
                            ));
                        }
                    } else {
                        // For non-space tokens, check if we need to add prefix space
                        // This happens when add_prefix_space is true and the previous token was a literal space
                        let needs_prefix = self.add_prefix_space
                            && idx > 0
                            && regex_tokens[idx - 1]
                                .text
                                .chars()
                                .all(|c| c.is_whitespace())
                            && self.vocab.values.contains_key(regex_tokens[idx - 1].text);

                        if needs_prefix && !token.text.starts_with(' ') {
                            // Add prefix space before applying BPE
                            let mut modified_text = String::from(" ");
                            modified_text.push_str(token.text);
                            let mut modified_offsets = vec![token.offset.begin];
                            modified_offsets.extend_from_slice(token.reference_offsets);

                            let modified_token = TokenRef {
                                text: &modified_text,
                                offset: token.offset,
                                reference_offsets: &modified_offsets,
                                mask: token.mask,
                            };

                            sub_tokens.extend(split_on_bpe_pairs(
                                modified_token,
                                bpe,
                                &self.bpe_ranks,
                                &self.cache,
                                true, // as_bytes = true for ByteLevel encoding
                            ));
                        } else {
                            // Apply BPE with ByteLevel encoding normally
                            sub_tokens.extend(split_on_bpe_pairs(
                                *token,
                                bpe,
                                &self.bpe_ranks,
                                &self.cache,
                                true, // as_bytes = true for ByteLevel encoding
                            ));
                        }
                    }
                }
            } else {
                sub_tokens.push(token.clone());
            }
        }

        fix_mask(&mut sub_tokens);
        sub_tokens
    }

    fn convert_tokens_to_string(&self, tokens: Vec<String>) -> String {
        let tokens = tokens
            .iter()
            .join("")
            .replace(" ##", "")
            .chars()
            .map(|character| {
                if let Some(byte) = UNICODE_TO_BYTES.get(&character) {
                    *byte
                } else {
                    // Handle regular characters that aren't in the ByteLevel mapping
                    // This includes actual spaces and other ASCII characters
                    character as u8
                }
            })
            .collect::<Vec<u8>>();
        String::from_utf8_lossy(tokens.as_slice()).to_string()
    }

    fn build_input_with_special_tokens(
        &self,
        tokens_ids_with_offsets_1: crate::TokenIdsWithOffsets,
        tokens_ids_with_offsets_2: Option<crate::TokenIdsWithOffsets>,
    ) -> crate::TokenIdsWithSpecialTokens {
        let mut output = crate::TokenIdsWithSpecialTokens {
            token_ids: Vec::new(),
            segment_ids: Vec::new(),
            special_tokens_mask: Vec::new(),
            token_offsets: Vec::new(),
            reference_offsets: Vec::new(),
            mask: Vec::new(),
        };

        // Add BOS token if needed
        if self.add_bos_token {
            if let Some(bos_id) = self.vocab.special_values.get(self.vocab.get_bos_value()) {
                output.token_ids.push(*bos_id);
                output.segment_ids.push(0);
                output.special_tokens_mask.push(1);
                output.token_offsets.push(None);
                output.reference_offsets.push(vec![]);
                output.mask.push(Mask::Special);
            }
        }

        // Add first sequence
        output.token_ids.extend(&tokens_ids_with_offsets_1.ids);
        output
            .segment_ids
            .extend(vec![0; tokens_ids_with_offsets_1.ids.len()]);
        output
            .special_tokens_mask
            .extend(vec![0; tokens_ids_with_offsets_1.ids.len()]);
        output
            .token_offsets
            .extend(tokens_ids_with_offsets_1.offsets);
        output
            .reference_offsets
            .extend(tokens_ids_with_offsets_1.reference_offsets);
        output.mask.extend(tokens_ids_with_offsets_1.masks);

        // Add EOS token after first sequence if needed
        if self.add_eos_token {
            if let Some(eos_id) = self.vocab.special_values.get(self.vocab.get_eos_value()) {
                output.token_ids.push(*eos_id);
                output.segment_ids.push(0);
                output.special_tokens_mask.push(1);
                output.token_offsets.push(None);
                output.reference_offsets.push(vec![]);
                output.mask.push(Mask::Special);
            }
        }

        // Add second sequence if provided
        if let Some(tokens_2) = tokens_ids_with_offsets_2 {
            // Add BOS token before second sequence if needed
            if self.add_bos_token {
                if let Some(bos_id) = self.vocab.special_values.get(self.vocab.get_bos_value()) {
                    output.token_ids.push(*bos_id);
                    output.segment_ids.push(1);
                    output.special_tokens_mask.push(1);
                    output.token_offsets.push(None);
                    output.reference_offsets.push(vec![]);
                    output.mask.push(Mask::Special);
                }
            }

            output.token_ids.extend(&tokens_2.ids);
            output.segment_ids.extend(vec![1; tokens_2.ids.len()]);
            output
                .special_tokens_mask
                .extend(vec![0; tokens_2.ids.len()]);
            output.token_offsets.extend(tokens_2.offsets);
            output.reference_offsets.extend(tokens_2.reference_offsets);
            output.mask.extend(tokens_2.masks);

            // Add EOS token after second sequence if needed
            if self.add_eos_token {
                if let Some(eos_id) = self.vocab.special_values.get(self.vocab.get_eos_value()) {
                    output.token_ids.push(*eos_id);
                    output.segment_ids.push(1);
                    output.special_tokens_mask.push(1);
                    output.token_offsets.push(None);
                    output.reference_offsets.push(vec![]);
                    output.mask.push(Mask::Special);
                }
            }
        }

        output
    }
}

impl MultiThreadedTokenizer<GptNeoXVocab> for GptNeoXTokenizer {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::base_tokenizer::TruncationStrategy;
    use crate::vocab::base_vocab::{swap_key_values, SpecialTokenMap};
    use crate::vocab::GptNeoXVocab;
    use crate::{Offset, TokenizedInput};
    use std::collections::HashMap;

    fn generate_test_vocab() -> GptNeoXVocab {
        let values: HashMap<String, i64> = [
            ("t".to_owned(), 0),
            ("h".to_owned(), 1),
            ("a@@".to_owned(), 2),
            ("n".to_owned(), 3),
            ("the".to_owned(), 4),
            ("Ġ".to_owned(), 5),
            ("<|endoftext|>".to_owned(), 6),
            ("o@@".to_owned(), 7),
            ("Ġear".to_owned(), 8),
            ("th".to_owned(), 9),
            ("Ċ".to_owned(), 10),
        ]
        .iter()
        .cloned()
        .collect();

        let special_token_map = SpecialTokenMap {
            unk_token: "<|endoftext|>".to_string(),
            pad_token: None,
            bos_token: Some("<|endoftext|>".to_string()),
            sep_token: None,
            cls_token: None,
            eos_token: Some("<|endoftext|>".to_string()),
            mask_token: None,
            additional_special_tokens: None,
        };

        let special_values: HashMap<String, i64> =
            [("<|endoftext|>".to_owned(), 6)].iter().cloned().collect();

        let indices = swap_key_values(&values);
        let special_indices = swap_key_values(&special_values);

        GptNeoXVocab {
            values,
            indices,
            special_token_map,
            special_values,
            special_indices,
        }
    }

    fn generate_test_merges() -> BpePairVocab {
        let values: HashMap<(String, String), i64> = [
            (("Ġ".to_owned(), "t".to_owned()), 0),
            (("Ġ".to_owned(), "n".to_owned()), 1),
            (("e".to_owned(), "e".to_owned()), 2),
            (("Ġt".to_owned(), "he".to_owned()), 3),
            (("h".to_owned(), "e".to_owned()), 4),
            (("t".to_owned(), "h".to_owned()), 5),
            (("t".to_owned(), "he".to_owned()), 6),
            (("Ġ".to_owned(), "e".to_owned()), 7),
            (("Ġe".to_owned(), "a".to_owned()), 8),
            (("Ġea".to_owned(), "r".to_owned()), 9),
        ]
        .iter()
        .cloned()
        .collect();

        BpePairVocab { values }
    }

    #[test]
    fn test_gpt_neox_tokenizer() {
        //        Given
        let vocab = generate_test_vocab();
        let merges = generate_test_merges();
        let gpt_neox_tokenizer: GptNeoXTokenizer = GptNeoXTokenizer::from_existing_vocab_and_merges(
            vocab, merges, true, false, false, false,
        );
        let test_tuples = [
            ("the Earth", vec!["the", "Ġear", "th"]),
            ("", vec![]),
            (" ", vec!["Ġ"]),
            ("   t", vec!["Ġ", "Ġ", "Ġ", "t"]),
            ("t ", vec!["t", "Ġ"]),
            (" \n ", vec!["Ġ", "Ċ", "Ġ"]),
        ];
        let source_texts: Vec<&str> = test_tuples.iter().map(|v| v.0).collect();
        let expected_results: Vec<Vec<&str>> = test_tuples.iter().map(|v| v.1.clone()).collect();

        //        When & Then
        for (source_text, expected_result) in test_tuples.iter() {
            assert_eq!(gpt_neox_tokenizer.tokenize(source_text), *expected_result);
        }

        assert_eq!(
            MultiThreadedTokenizer::tokenize_list(&gpt_neox_tokenizer, &source_texts),
            expected_results
        );
    }

    #[test]
    fn test_gpt_neox_tokenizer_with_prefix_space() {
        //        Given
        let vocab = generate_test_vocab();
        let merges = generate_test_merges();
        let gpt_neox_tokenizer: GptNeoXTokenizer = GptNeoXTokenizer::from_existing_vocab_and_merges(
            vocab, merges, true, true, false, false,
        );
        let test_tuples = [
            ("the Earth", vec!["Ġthe", "Ġear", "th"]),
            ("", vec![]),
            (" ", vec!["Ġ"]),
            ("   t", vec!["Ġ", "Ġ", "Ġ", "t"]),
        ];

        //        When & Then
        for (source_text, expected_result) in test_tuples.iter() {
            assert_eq!(gpt_neox_tokenizer.tokenize(source_text), *expected_result);
        }
    }

    #[test]
    fn test_gpt_neox_tokenizer_no_lower_casing() {
        //        Given
        let vocab = generate_test_vocab();
        let merges = generate_test_merges();
        let gpt_neox_tokenizer: GptNeoXTokenizer = GptNeoXTokenizer::from_existing_vocab_and_merges(
            vocab, merges, false, false, false, false,
        );
        let test_tuples = [
            ("the Earth", vec!["the", "Ġ", "E", "a", "r", "th"]),
            ("", vec![]),
            (" ", vec!["Ġ"]),
            ("   t", vec!["Ġ", "Ġ", "Ġ", "t"]),
            (" \n ", vec!["Ġ", "Ċ", "Ġ"]),
        ];
        let source_texts: Vec<&str> = test_tuples.iter().map(|v| v.0).collect();
        let expected_results: Vec<Vec<&str>> = test_tuples.iter().map(|v| v.1.clone()).collect();

        //        When & Then
        for (source_text, expected_result) in test_tuples.iter() {
            assert_eq!(gpt_neox_tokenizer.tokenize(source_text), *expected_result);
        }

        assert_eq!(
            MultiThreadedTokenizer::tokenize_list(&gpt_neox_tokenizer, &source_texts),
            expected_results
        );
    }

    #[test]
    fn test_encode_with_bos_eos() {
        //        Given
        let vocab = generate_test_vocab();
        let merges = generate_test_merges();
        let gpt_neox_tokenizer: GptNeoXTokenizer = GptNeoXTokenizer::from_existing_vocab_and_merges(
            vocab, merges, true, false, true, true,
        );
        let truncation_strategy = TruncationStrategy::LongestFirst;
        let test_tuples = [(
            "the earth",
            TokenizedInput {
                token_ids: vec![6, 4, 8, 9, 6],
                segment_ids: vec![0, 0, 0, 0, 0],
                special_tokens_mask: vec![1, 0, 0, 0, 1],
                overflowing_tokens: vec![],
                num_truncated_tokens: 0,
                token_offsets: vec![
                    None,
                    Some(Offset { begin: 0, end: 3 }),
                    Some(Offset { begin: 3, end: 7 }),
                    Some(Offset { begin: 7, end: 9 }),
                    None,
                ],
                reference_offsets: vec![
                    vec![],
                    vec![0, 1, 2],
                    vec![3, 4, 5, 6],
                    vec![7, 8],
                    vec![],
                ],
                mask: vec![
                    Mask::Special,
                    Mask::None,
                    Mask::Begin,
                    Mask::Continuation,
                    Mask::Special,
                ],
            },
        )];

        //        When & Then
        for (source_text, expected_result) in test_tuples.iter() {
            assert_eq!(
                gpt_neox_tokenizer.encode(source_text, None, 128, &truncation_strategy, 0),
                *expected_result
            );
        }
    }

    #[test]
    fn test_decode() {
        //        Given
        let vocab = generate_test_vocab();
        let merges = generate_test_merges();
        let gpt_neox_tokenizer: GptNeoXTokenizer = GptNeoXTokenizer::from_existing_vocab_and_merges(
            vocab, merges, true, false, false, false,
        );
        let skip_special_tokens = false;
        let clean_up_tokenization_spaces = false;
        let test_tuples = [(vec![4, 8, 9], "the earth")];
        let source_ids: Vec<Vec<i64>> = test_tuples.iter().map(|v| v.0.clone()).collect_vec();
        let expected_results: Vec<&str> = test_tuples.iter().map(|v| v.1).collect_vec();

        //        When & Then
        for (source_ids, expected_result) in test_tuples.iter() {
            assert_eq!(
                gpt_neox_tokenizer.decode(
                    source_ids,
                    skip_special_tokens,
                    clean_up_tokenization_spaces
                ),
                *expected_result
            );
        }
        assert_eq!(
            Tokenizer::decode_list(
                &gpt_neox_tokenizer,
                &source_ids,
                skip_special_tokens,
                clean_up_tokenization_spaces
            ),
            expected_results
        );
    }
}
