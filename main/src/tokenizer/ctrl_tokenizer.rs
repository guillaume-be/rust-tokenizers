// Copyright 2018 Salesforce
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
use crate::tokenizer::tokenization_utils::{
    ctrl_bpe, fix_mask, lowercase, split_on_bpe_pairs, split_on_regex, split_on_special_tokens,
    BpeCache,
};
use crate::tokenizer::{MultiThreadedTokenizer, Tokenizer};
use crate::vocab::bpe_vocab::BpePairVocab;
use crate::vocab::{OpenAiGptVocab, Vocab};
use crate::{Mask, Token, TokenRef};
use regex::Regex;
use std::collections::HashMap;
use std::path::Path;
use std::sync::RwLock;

/// # CTRL tokenizer
/// CTRL tokenizer performing:
/// - splitting on special characters
/// - whitespace splitting
/// - (optional) lower casing
/// - BPE tokenization
pub struct CtrlTokenizer {
    vocab: OpenAiGptVocab,
    bpe_ranks: BpePairVocab,
    cache: BpeCache,
    regex_pattern: Regex,
    lower_case: bool,
}

impl CtrlTokenizer {
    /// Create a new instance of a `CtrlTokenizer`
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
    /// use rust_tokenizers::tokenizer::{CtrlTokenizer, Tokenizer};
    /// let lower_case = false;
    /// let tokenizer =
    ///     CtrlTokenizer::from_file("path/to/vocab/file", "path/to/merges/file", lower_case).unwrap();
    /// ```
    pub fn from_file<V: AsRef<Path>, M: AsRef<Path>>(
        vocab_path: V,
        merges_path: M,
        lower_case: bool,
    ) -> Result<CtrlTokenizer, TokenizerError> {
        let vocab = OpenAiGptVocab::from_file(vocab_path)?;
        let bpe_ranks = BpePairVocab::from_file(merges_path)?;
        let cache = RwLock::new(HashMap::new());
        let regex_pattern = Regex::new(r"\S+\n?").unwrap();
        Ok(CtrlTokenizer {
            vocab,
            bpe_ranks,
            cache,
            regex_pattern,
            lower_case,
        })
    }

    /// Create a new instance of a `CtrlTokenizer`
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
    /// use rust_tokenizers::tokenizer::{CtrlTokenizer, Tokenizer};
    /// let lower_case = false;
    /// let tokenizer = CtrlTokenizer::from_file_with_special_token_mapping(
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
    ) -> Result<CtrlTokenizer, TokenizerError> {
        let vocab = OpenAiGptVocab::from_file_with_special_token_mapping(
            vocab_path,
            special_token_mapping_path,
        )?;
        let bpe_ranks = BpePairVocab::from_file(merges_path)?;
        let cache = RwLock::new(HashMap::new());
        let regex_pattern = Regex::new(r"\S+\n?").unwrap();
        Ok(CtrlTokenizer {
            vocab,
            bpe_ranks,
            cache,
            regex_pattern,
            lower_case,
        })
    }

    /// Create a new instance of a `CtrlTokenizer` from an existing vocabulary and merges
    ///
    /// # Parameters
    /// - vocab (`OpenAiGptVocab`): GPT-like vocabulary
    /// - merges (`BpePairVocab`): BPE pairs vocabulary
    /// - lower_case (`bool`): flag indicating if the text should be lower-cased as part of the tokenization
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_tokenizers::tokenizer::{CtrlTokenizer, Tokenizer};
    /// use rust_tokenizers::vocab::{BpePairVocab, OpenAiGptVocab, Vocab};
    /// let lower_case = false;
    /// let vocab = OpenAiGptVocab::from_file("path/to/vocab/file").unwrap();
    /// let merges = BpePairVocab::from_file("path/to/merges/file").unwrap();
    ///
    /// let tokenizer = CtrlTokenizer::from_existing_vocab_and_merges(vocab, merges, lower_case);
    /// ```
    pub fn from_existing_vocab_and_merges(
        vocab: OpenAiGptVocab,
        merges: BpePairVocab,
        lower_case: bool,
    ) -> CtrlTokenizer {
        let cache = RwLock::new(HashMap::new());
        let regex_pattern = Regex::new(r"\S+\n?").unwrap();
        CtrlTokenizer {
            vocab,
            bpe_ranks: merges,
            cache,
            regex_pattern,
            lower_case,
        }
    }
}

impl Tokenizer<OpenAiGptVocab> for CtrlTokenizer {
    fn vocab(&self) -> &OpenAiGptVocab {
        &self.vocab
    }
    fn vocab_mut(&mut self) -> &mut OpenAiGptVocab {
        &mut self.vocab
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
                for token in split_on_regex(token.as_ref(), &self.regex_pattern) {
                    sub_tokens.extend(split_on_bpe_pairs(
                        token,
                        ctrl_bpe,
                        &self.bpe_ranks,
                        &self.cache,
                        false,
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
        tokens.join(" ").replace("@@ ", "").trim().to_owned()
    }
}

impl MultiThreadedTokenizer<OpenAiGptVocab> for CtrlTokenizer {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::base_tokenizer::{Offset, TokenizedInput, TruncationStrategy};
    use crate::vocab::base_vocab::{swap_key_values, SpecialTokenMap};
    use crate::vocab::OpenAiGptVocab;
    use crate::Mask;
    use itertools::Itertools;
    use std::collections::HashMap;

    fn generate_test_vocab() -> OpenAiGptVocab {
        let values: HashMap<String, i64> = [
            ("t".to_owned(), 0),
            ("h".to_owned(), 1),
            ("a@@".to_owned(), 2),
            ("n".to_owned(), 3),
            ("the".to_owned(), 4),
            ("r@@".to_owned(), 5),
            ("<unk>".to_owned(), 6),
            ("o@@".to_owned(), 8),
        ]
        .iter()
        .cloned()
        .collect();

        let special_token_map = SpecialTokenMap {
            unk_token: "<unk>".to_string(),
            pad_token: None,
            bos_token: None,
            sep_token: None,
            cls_token: None,
            eos_token: None,
            mask_token: None,
            additional_special_tokens: None,
        };
        let special_values: HashMap<String, i64> =
            [("<unk>".to_owned(), 6)].iter().cloned().collect();

        let indices = swap_key_values(&values);
        let special_indices = swap_key_values(&special_values);

        OpenAiGptVocab {
            values,
            indices,
            special_token_map,
            special_values,
            special_indices,
        }
    }

    fn generate_test_merges() -> BpePairVocab {
        let values: HashMap<(String, String), i64> = [
            (("t".to_owned(), "h".to_owned()), 0),
            (("a".to_owned(), "n".to_owned()), 1),
            (("i".to_owned(), "n".to_owned()), 2),
            (("th".to_owned(), "e</w>".to_owned()), 3),
            (("e".to_owned(), "r".to_owned()), 4),
            (("r".to_owned(), "e".to_owned()), 5),
            (("l".to_owned(), "l".to_owned()), 6),
        ]
        .iter()
        .cloned()
        .collect();

        BpePairVocab { values }
    }

    #[test]
    fn test_ctrl_tokenizer() {
        //        Given
        let vocab = generate_test_vocab();
        let merges = generate_test_merges();
        let ctrl_tokenizer: CtrlTokenizer =
            CtrlTokenizer::from_existing_vocab_and_merges(vocab, merges, true);
        let test_tuples = [
            ("The Earth", vec!["the", "e@@", "a@@", "r@@", "t@@", "h"]),
            (
                "Hello, world!",
                vec![
                    "h@@", "e@@", "ll@@", "o@@", ",", "w@@", "o@@", "r@@", "l@@", "d@@", "!",
                ],
            ),
            ("", vec![]),
            (" ", vec![]),
            (" \n ", vec![]),
        ];
        let source_texts: Vec<&str> = test_tuples.iter().map(|v| v.0).collect();
        let expected_results: Vec<Vec<&str>> = test_tuples.iter().map(|v| v.1.clone()).collect();

        //        When & Then
        for (source_text, expected_result) in test_tuples.iter() {
            assert_eq!(ctrl_tokenizer.tokenize(source_text), *expected_result);
        }

        assert_eq!(
            MultiThreadedTokenizer::tokenize_list(&ctrl_tokenizer, &source_texts),
            expected_results
        );
    }

    #[test]
    fn test_ctrl_tokenizer_no_lower_casing() {
        //        Given
        let vocab = generate_test_vocab();
        let merges = generate_test_merges();
        let ctrl_tokenizer: CtrlTokenizer =
            CtrlTokenizer::from_existing_vocab_and_merges(vocab, merges, false);
        let test_tuples = [
            ("the Earth", vec!["the", "E@@", "a@@", "r@@", "t@@", "h"]),
            (
                "Hello, world!",
                vec![
                    "H@@", "e@@", "ll@@", "o@@", ",", "w@@", "o@@", "r@@", "l@@", "d@@", "!",
                ],
            ),
            ("", vec![]),
            (" ", vec![]),
            (" \n ", vec![]),
        ];
        let source_texts: Vec<&str> = test_tuples.iter().map(|v| v.0).collect();
        let expected_results: Vec<Vec<&str>> = test_tuples.iter().map(|v| v.1.clone()).collect();

        //        When & Then
        for (source_text, expected_result) in test_tuples.iter() {
            assert_eq!(ctrl_tokenizer.tokenize(source_text), *expected_result);
        }

        assert_eq!(
            MultiThreadedTokenizer::tokenize_list(&ctrl_tokenizer, &source_texts),
            expected_results
        );
    }

    #[test]
    fn test_encode() {
        //        Given
        let vocab = generate_test_vocab();
        let merges = generate_test_merges();
        let ctrl_tokenizer: CtrlTokenizer =
            CtrlTokenizer::from_existing_vocab_and_merges(vocab, merges, false);
        let truncation_strategy = TruncationStrategy::LongestFirst;
        let test_tuples = [
            (
                "the earth",
                TokenizedInput {
                    token_ids: vec![4, 6, 2, 5, 6, 1],
                    segment_ids: vec![0, 0, 0, 0, 0, 0],
                    special_tokens_mask: vec![0, 0, 0, 0, 0, 0],
                    overflowing_tokens: vec![],
                    num_truncated_tokens: 0,
                    token_offsets: vec![
                        Some(Offset { begin: 0, end: 3 }),
                        Some(Offset { begin: 4, end: 5 }),
                        Some(Offset { begin: 5, end: 6 }),
                        Some(Offset { begin: 6, end: 7 }),
                        Some(Offset { begin: 7, end: 8 }),
                        Some(Offset { begin: 8, end: 9 }),
                    ],
                    reference_offsets: vec![
                        vec![0, 1, 2],
                        vec![4],
                        vec![5],
                        vec![6],
                        vec![7],
                        vec![8],
                    ],
                    mask: vec![
                        Mask::None,
                        Mask::Begin,
                        Mask::Continuation,
                        Mask::Continuation,
                        Mask::Continuation,
                        Mask::Continuation,
                    ],
                },
            ),
            (
                "Hello, world!",
                TokenizedInput {
                    token_ids: vec![6, 6, 6, 8, 6, 6, 8, 5, 6, 6, 6],
                    segment_ids: vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    special_tokens_mask: vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    overflowing_tokens: vec![],
                    num_truncated_tokens: 0,
                    token_offsets: vec![
                        Some(Offset { begin: 0, end: 1 }),
                        Some(Offset { begin: 1, end: 2 }),
                        Some(Offset { begin: 2, end: 4 }),
                        Some(Offset { begin: 4, end: 5 }),
                        Some(Offset { begin: 5, end: 6 }),
                        Some(Offset { begin: 7, end: 8 }),
                        Some(Offset { begin: 8, end: 9 }),
                        Some(Offset { begin: 9, end: 10 }),
                        Some(Offset { begin: 10, end: 11 }),
                        Some(Offset { begin: 11, end: 12 }),
                        Some(Offset { begin: 12, end: 13 }),
                    ],
                    reference_offsets: vec![
                        vec![0],
                        vec![1],
                        vec![2, 3],
                        vec![4],
                        vec![5],
                        vec![7],
                        vec![8],
                        vec![9],
                        vec![10],
                        vec![11],
                        vec![12],
                    ],
                    mask: vec![
                        Mask::Begin,
                        Mask::Continuation,
                        Mask::Continuation,
                        Mask::Continuation,
                        Mask::Continuation,
                        Mask::Begin,
                        Mask::Continuation,
                        Mask::Continuation,
                        Mask::Continuation,
                        Mask::Continuation,
                        Mask::Continuation,
                    ],
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
                ctrl_tokenizer.encode(source_text, None, 128, &truncation_strategy, 0),
                *expected_result
            );
        }
        assert_eq!(
            MultiThreadedTokenizer::encode_list(
                &ctrl_tokenizer,
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
        let ctrl_tokenizer: CtrlTokenizer =
            CtrlTokenizer::from_existing_vocab_and_merges(vocab, merges, false);
        let skip_special_tokens = false;
        let clean_up_tokenization_spaces = false;
        let test_tuples = [(vec![4, 6, 2, 5, 6, 1], "the <unk> ar<unk> h")];
        let source_ids: Vec<Vec<i64>> = test_tuples.iter().map(|v| v.0.clone()).collect_vec();
        let expected_results: Vec<&str> = test_tuples.iter().map(|v| v.1).collect_vec();

        //        When & Then
        for (source_ids, expected_result) in test_tuples.iter() {
            assert_eq!(
                ctrl_tokenizer.decode(
                    source_ids,
                    skip_special_tokens,
                    clean_up_tokenization_spaces
                ),
                *expected_result
            );
        }
        assert_eq!(
            Tokenizer::decode_list(
                &ctrl_tokenizer,
                &source_ids,
                skip_special_tokens,
                clean_up_tokenization_spaces
            ),
            expected_results
        );
    }

    #[test]
    fn test_decode_skip_special_tokens() {
        //        Given
        let vocab = generate_test_vocab();
        let merges = generate_test_merges();
        let ctrl_tokenizer: CtrlTokenizer =
            CtrlTokenizer::from_existing_vocab_and_merges(vocab, merges, false);
        let skip_special_tokens = true;
        let clean_up_tokenization_spaces = true;
        let test_tuples = [(vec![4, 6, 2, 5, 6, 1], "the arh")];
        let source_ids: Vec<Vec<i64>> = test_tuples.iter().map(|v| v.0.clone()).collect_vec();
        let expected_results: Vec<&str> = test_tuples.iter().map(|v| v.1).collect_vec();

        //        When & Then
        for (source_ids, expected_result) in test_tuples.iter() {
            assert_eq!(
                ctrl_tokenizer.decode(
                    source_ids,
                    skip_special_tokens,
                    clean_up_tokenization_spaces
                ),
                *expected_result
            );
        }
        assert_eq!(
            Tokenizer::decode_list(
                &ctrl_tokenizer,
                &source_ids,
                skip_special_tokens,
                clean_up_tokenization_spaces
            ),
            expected_results
        );
    }
}
