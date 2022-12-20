// Copyright 2018 The Open AI Team Authors
// Copyright 2018 The HuggingFace Inc. team.
// Copyright 2019-2020 Guillaume Becquin
// Copyright 2020 Maarten van Gompel
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
    bpe, fix_mask, split_on_bpe_pairs, split_on_regex_with_lookahead, split_on_special_tokens,
};
use crate::tokenizer::tokenization_utils::{lowercase, BpeCache};
use crate::tokenizer::{MultiThreadedTokenizer, Tokenizer};
use crate::vocab::bpe_vocab::BpePairVocab;
use crate::vocab::{Gpt2Vocab, Vocab};
use crate::{Mask, Token, TokenRef};
use itertools::Itertools;
use regex::Regex;
use std::collections::HashMap;
use std::iter::Iterator;
use std::path::Path;
use std::sync::RwLock;

/// # GPT2 tokenizer
/// GPT2 tokenizer performing:
/// - splitting on special characters
/// - whitespace splitting
/// - (optional) lower casing
/// - BPE tokenization
pub struct Gpt2Tokenizer {
    vocab: Gpt2Vocab,
    bpe_ranks: BpePairVocab,
    cache: BpeCache,
    pattern_lookahead: Regex,
    pattern_tokenization: Regex,
    lower_case: bool,
}

impl Gpt2Tokenizer {
    /// Create a new instance of a `Gpt2Tokenizer`
    /// Expects a vocabulary json file and a merges file as an input.
    ///
    /// # Parameters
    /// - vocab_path (`&str`): path to the vocabulary file
    /// - merges_path (`&str`): path to the merges file (use as part of the BPE encoding process)
    /// - lower_case (`bool`): flag indicating if the text should be lower-cased as part of the tokenization
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_tokenizers::tokenizer::{Gpt2Tokenizer, Tokenizer};
    /// let lower_case = false;
    /// let tokenizer =
    ///     Gpt2Tokenizer::from_file("path/to/vocab/file", "path/to/merges/file", lower_case).unwrap();
    /// ```
    pub fn from_file<P: AsRef<Path>, M: AsRef<Path>>(
        vocab_path: P,
        merges_path: M,
        lower_case: bool,
    ) -> Result<Gpt2Tokenizer, TokenizerError> {
        let vocab = Gpt2Vocab::from_file(vocab_path)?;
        let bpe_ranks = BpePairVocab::from_file(merges_path)?;
        let cache = RwLock::new(HashMap::new());
        let pattern_lookahead = Regex::new(r"\s+\S").unwrap();
        let pattern_tokenization =
            Regex::new(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+")
                .unwrap();
        Ok(Gpt2Tokenizer {
            vocab,
            bpe_ranks,
            cache,
            pattern_lookahead,
            pattern_tokenization,
            lower_case,
        })
    }

    /// Create a new instance of a `Gpt2Tokenizer`
    /// Expects a vocabulary json file and a merges file and special token mapping file as inputs.
    ///
    /// # Parameters
    /// - vocab_path (`&str`): path to the vocabulary file
    /// - merges_path (`&str`): path to the merges file (use as part of the BPE encoding process)
    /// - lower_case (`bool`): flag indicating if the text should be lower-cased as part of the tokenization
    /// - special_token_mapping_path (`&str`): path to a special token mapping file to overwrite default special tokens
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_tokenizers::tokenizer::{Gpt2Tokenizer, Tokenizer};
    /// let lower_case = false;
    /// let tokenizer = Gpt2Tokenizer::from_file_with_special_token_mapping(
    ///     "path/to/vocab/file",
    ///     "path/to/merges/file",
    ///     lower_case,
    ///     "path/to/special/token/mapping/file",
    /// )
    /// .unwrap();
    /// ```
    pub fn from_file_with_special_token_mapping<V: AsRef<Path>, M: AsRef<Path>, S: AsRef<Path>>(
        vocab_path: V,
        merges_path: M,
        lower_case: bool,
        special_token_mapping_path: S,
    ) -> Result<Gpt2Tokenizer, TokenizerError> {
        let vocab = Gpt2Vocab::from_file_with_special_token_mapping(
            vocab_path,
            special_token_mapping_path,
        )?;
        let bpe_ranks = BpePairVocab::from_file(merges_path)?;
        let cache = RwLock::new(HashMap::new());
        let pattern_lookahead = Regex::new(r"\s+\S").unwrap();
        let pattern_tokenization =
            Regex::new(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+")
                .unwrap();
        Ok(Gpt2Tokenizer {
            vocab,
            bpe_ranks,
            cache,
            pattern_lookahead,
            pattern_tokenization,
            lower_case,
        })
    }

    /// Create a new instance of a `Gpt2Tokenizer` from an existing vocabulary and merges
    ///
    /// # Parameters
    /// - vocab (`Gpt2Vocab`): GPT-like vocabulary
    /// - merges (`BpePairVocab`): BPE pairs vocabulary
    /// - lower_case (`bool`): flag indicating if the text should be lower-cased as part of the tokenization
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_tokenizers::tokenizer::{Gpt2Tokenizer, Tokenizer};
    /// use rust_tokenizers::vocab::{BpePairVocab, Gpt2Vocab, Vocab};
    /// let lower_case = false;
    /// let vocab = Gpt2Vocab::from_file("path/to/vocab/file").unwrap();
    /// let merges = BpePairVocab::from_file("path/to/merges/file").unwrap();
    ///
    /// let tokenizer = Gpt2Tokenizer::from_existing_vocab_and_merges(vocab, merges, lower_case);
    /// ```
    pub fn from_existing_vocab_and_merges(
        vocab: Gpt2Vocab,
        merges: BpePairVocab,
        lower_case: bool,
    ) -> Gpt2Tokenizer {
        let cache = RwLock::new(HashMap::new());
        let pattern_lookahead = Regex::new(r"\s+\S").unwrap();
        let pattern_tokenization =
            Regex::new(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+")
                .unwrap();
        Gpt2Tokenizer {
            vocab,
            bpe_ranks: merges,
            cache,
            pattern_lookahead,
            pattern_tokenization,
            lower_case,
        }
    }
}

impl Tokenizer<Gpt2Vocab> for Gpt2Tokenizer {
    fn vocab(&self) -> &Gpt2Vocab {
        &self.vocab
    }

    fn tokenize_to_tokens(&self, initial_token: TokenRef) -> Vec<Token> {
        let mut tokens = split_on_special_tokens(initial_token, &self.vocab)
            .into_iter()
            .map(|token| token.to_owned())
            .collect::<Vec<Token>>();

        let mut sub_tokens = Vec::new();
        for token in tokens.iter_mut() {
            if token.mask != Mask::Special && token.mask != Mask::Unknown {
                if self.lower_case {
                    lowercase(token);
                }
                for token in split_on_regex_with_lookahead(
                    token.as_ref(),
                    &self.pattern_lookahead,
                    &self.pattern_tokenization,
                ) {
                    sub_tokens.extend(split_on_bpe_pairs(
                        token,
                        bpe,
                        &self.bpe_ranks,
                        &self.cache,
                        true,
                    ));
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
            .trim()
            .chars()
            .map(|character| *UNICODE_TO_BYTES.get(&character).unwrap())
            .collect::<Vec<u8>>();
        String::from_utf8_lossy(tokens.as_slice()).to_string()
    }
}

impl MultiThreadedTokenizer<Gpt2Vocab> for Gpt2Tokenizer {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::base_tokenizer::TruncationStrategy;
    use crate::vocab::base_vocab::{swap_key_values, SpecialTokenMap};
    use crate::vocab::Gpt2Vocab;
    use crate::{Offset, TokenizedInput};
    use std::collections::HashMap;

    fn generate_test_vocab() -> Gpt2Vocab {
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

        Gpt2Vocab {
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
    fn test_gpt2_tokenizer() {
        //        Given
        let vocab = generate_test_vocab();
        let merges = generate_test_merges();
        let gpt2_tokenizer: Gpt2Tokenizer =
            Gpt2Tokenizer::from_existing_vocab_and_merges(vocab, merges, true);
        let test_tuples = [
            ("the Earth", vec!["the", "Ġear", "th"]),
            ("", vec![]),
            (" ", vec![]),
            ("   t", vec!["Ġ", "Ġ", "Ġt"]),
            ("t ", vec!["t", "Ġ"]),
            (" \n ", vec![]),
        ];
        let source_texts: Vec<&str> = test_tuples.iter().map(|v| v.0).collect();
        let expected_results: Vec<Vec<&str>> = test_tuples.iter().map(|v| v.1.clone()).collect();

        //        When & Then
        for (source_text, expected_result) in test_tuples.iter() {
            assert_eq!(gpt2_tokenizer.tokenize(source_text), *expected_result);
        }

        assert_eq!(
            MultiThreadedTokenizer::tokenize_list(&gpt2_tokenizer, &source_texts),
            expected_results
        );
    }

    #[test]
    fn test_gpt2_tokenizer_no_lower_casing() {
        //        Given
        let vocab = generate_test_vocab();
        let merges = generate_test_merges();
        let gpt2_tokenizer: Gpt2Tokenizer =
            Gpt2Tokenizer::from_existing_vocab_and_merges(vocab, merges, false);
        let test_tuples = [
            ("the Earth", vec!["the", "Ġ", "E", "a", "r", "th"]),
            ("", vec![]),
            (" ", vec![]),
            ("   t", vec!["Ġ", "Ġ", "Ġt"]),
            (" \n ", vec![]),
        ];
        let source_texts: Vec<&str> = test_tuples.iter().map(|v| v.0).collect();
        let expected_results: Vec<Vec<&str>> = test_tuples.iter().map(|v| v.1.clone()).collect();

        //        When & Then
        for (source_text, expected_result) in test_tuples.iter() {
            assert_eq!(gpt2_tokenizer.tokenize(source_text), *expected_result);
        }

        assert_eq!(
            MultiThreadedTokenizer::tokenize_list(&gpt2_tokenizer, &source_texts),
            expected_results
        );
    }

    #[test]
    fn test_encode() {
        //        Given
        let vocab = generate_test_vocab();
        let merges = generate_test_merges();
        let gpt2_tokenizer: Gpt2Tokenizer =
            Gpt2Tokenizer::from_existing_vocab_and_merges(vocab, merges, true);
        let truncation_strategy = TruncationStrategy::LongestFirst;
        let test_tuples = [
            (
                "the earth",
                TokenizedInput {
                    token_ids: vec![4, 8, 9],
                    segment_ids: vec![0, 0, 0],
                    special_tokens_mask: vec![0, 0, 0],
                    overflowing_tokens: vec![],
                    num_truncated_tokens: 0,
                    token_offsets: vec![
                        Some(Offset { begin: 0, end: 3 }),
                        Some(Offset { begin: 3, end: 7 }),
                        Some(Offset { begin: 7, end: 9 }),
                    ],
                    reference_offsets: vec![vec![0, 1, 2], vec![3, 4, 5, 6], vec![7, 8]],
                    mask: vec![Mask::None, Mask::Begin, Mask::Continuation],
                },
            ),
            (
                " ",
                TokenizedInput {
                    token_ids: vec![],
                    segment_ids: vec![],
                    special_tokens_mask: vec![],
                    overflowing_tokens: vec![],
                    num_truncated_tokens: 0,
                    token_offsets: vec![],
                    reference_offsets: vec![],
                    mask: vec![],
                },
            ),
            (
                "",
                TokenizedInput {
                    token_ids: vec![],
                    segment_ids: vec![],
                    special_tokens_mask: vec![],
                    overflowing_tokens: vec![],
                    num_truncated_tokens: 0,
                    token_offsets: vec![],
                    reference_offsets: vec![],
                    mask: vec![],
                },
            ),
        ];
        let source_texts: Vec<&str> = test_tuples.iter().map(|v| v.0).collect();
        let expected_results: Vec<TokenizedInput> =
            test_tuples.iter().map(|v| v.1.clone()).collect();

        //        When & Then
        for (source_text, expected_result) in test_tuples.iter() {
            assert_eq!(
                gpt2_tokenizer.encode(source_text, None, 128, &truncation_strategy, 0),
                *expected_result
            );
        }
        assert_eq!(
            MultiThreadedTokenizer::encode_list(
                &gpt2_tokenizer,
                &source_texts,
                128,
                &truncation_strategy,
                0
            ),
            expected_results
        );
    }

    #[test]
    fn test_decode() {
        //        Given
        let vocab = generate_test_vocab();
        let merges = generate_test_merges();
        let gpt2_tokenizer: Gpt2Tokenizer =
            Gpt2Tokenizer::from_existing_vocab_and_merges(vocab, merges, true);
        let skip_special_tokens = false;
        let clean_up_tokenization_spaces = false;
        let test_tuples = [(vec![4, 8, 9], "the earth")];
        let source_ids: Vec<Vec<i64>> = test_tuples.iter().map(|v| v.0.clone()).collect_vec();
        let expected_results: Vec<&str> = test_tuples.iter().map(|v| v.1).collect_vec();

        //        When & Then
        for (source_ids, expected_result) in test_tuples.iter() {
            assert_eq!(
                gpt2_tokenizer.decode(
                    source_ids,
                    skip_special_tokens,
                    clean_up_tokenization_spaces
                ),
                *expected_result
            );
        }
        assert_eq!(
            Tokenizer::decode_list(
                &gpt2_tokenizer,
                &source_ids,
                skip_special_tokens,
                clean_up_tokenization_spaces
            ),
            expected_results
        );
    }
}
