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
use crate::tokenizer::base_tokenizer::{
    Mask, Offset, OffsetSize, Token, TokenIdsWithOffsets, TokenIdsWithSpecialTokens, TokenRef,
    Tokenizer,
};
use crate::tokenizer::constants::UNICODE_TO_BYTES;
use crate::tokenizer::tokenization_utils::{
    bpe, fix_mask, is_whitespace, split_on_bpe_pairs, split_on_regex_with_lookahead,
    split_on_special_tokens,
};
use crate::tokenizer::tokenization_utils::{lowercase, BpeCache};
use crate::tokenizer::MultiThreadedTokenizer;
use crate::vocab::bpe_vocab::BpePairVocab;
use crate::vocab::{RobertaVocab, Vocab};
use itertools::Itertools;
use regex::Regex;
use std::collections::HashMap;
use std::iter::Iterator;
use std::sync::RwLock;

/// # RoBERTa tokenizer
/// RoBERTa tokenizer performing:
/// - splitting on special characters
/// - whitespace splitting
/// - (optional) lower casing
/// - BPE tokenization
pub struct RobertaTokenizer {
    vocab: RobertaVocab,
    bpe_ranks: BpePairVocab,
    cache: BpeCache,
    pattern_lookahead: Regex,
    pattern_tokenization: Regex,
    lower_case: bool,
    add_prefix_space: bool,
}

impl RobertaTokenizer {
    /// Create a new instance of a `RobertaTokenizer`
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
    /// use rust_tokenizers::tokenizer::{RobertaTokenizer, Tokenizer};
    /// let lower_case = false;
    /// let add_prefix_space = true;
    /// let tokenizer = RobertaTokenizer::from_file(
    ///     "path/to/vocab/file",
    ///     "path/to/merges/file",
    ///     lower_case,
    ///     add_prefix_space,
    /// )
    /// .unwrap();
    /// ```
    pub fn from_file(
        vocab_path: &str,
        merges_path: &str,
        lower_case: bool,
        add_prefix_space: bool,
    ) -> Result<RobertaTokenizer, TokenizerError> {
        let vocab = RobertaVocab::from_file(vocab_path)?;
        let bpe_ranks = BpePairVocab::from_file(merges_path)?;
        let cache = RwLock::new(HashMap::new());
        let pattern_lookahead = Regex::new(r"\s+\S").unwrap();
        let pattern_tokenization =
            Regex::new(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+")
                .unwrap();
        Ok(RobertaTokenizer {
            vocab,
            bpe_ranks,
            cache,
            pattern_lookahead,
            pattern_tokenization,
            lower_case,
            add_prefix_space,
        })
    }

    /// Create a new instance of a `RobertaTokenizer` from an existing vocabulary and merges
    ///
    /// # Parameters
    /// - vocab (`RobertaVocab`): GPT-like vocabulary
    /// - merges (`BpePairVocab`): BPE pairs vocabulary
    /// - lower_case (`bool`): flag indicating if the text should be lower-cased as part of the tokenization
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_tokenizers::tokenizer::{RobertaTokenizer, Tokenizer};
    /// use rust_tokenizers::vocab::{BpePairVocab, RobertaVocab, Vocab};
    /// let lower_case = false;
    /// let add_prefix_space = true;
    /// let vocab = RobertaVocab::from_file("path/to/vocab/file").unwrap();
    /// let merges = BpePairVocab::from_file("path/to/merges/file").unwrap();
    ///
    /// let tokenizer = RobertaTokenizer::from_existing_vocab_and_merges(
    ///     vocab,
    ///     merges,
    ///     lower_case,
    ///     add_prefix_space,
    /// );
    /// ```
    pub fn from_existing_vocab_and_merges(
        vocab: RobertaVocab,
        merges: BpePairVocab,
        lower_case: bool,
        add_prefix_space: bool,
    ) -> RobertaTokenizer {
        let cache = RwLock::new(HashMap::new());
        let pattern_lookahead = Regex::new(r"\s+\S").unwrap();
        let pattern_tokenization =
            Regex::new(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+")
                .unwrap();
        RobertaTokenizer {
            vocab,
            bpe_ranks: merges,
            cache,
            pattern_lookahead,
            pattern_tokenization,
            lower_case,
            add_prefix_space,
        }
    }
}

impl Tokenizer<RobertaVocab> for RobertaTokenizer {
    fn vocab(&self) -> &RobertaVocab {
        &self.vocab
    }

    fn tokenize_to_tokens(&self, initial_token: TokenRef) -> Vec<Token> {
        if initial_token.text.is_empty() {
            return vec![];
        }
        let mut initial_token: Token = initial_token.to_owned();
        if !is_whitespace(&initial_token.text.chars().next().unwrap()) & self.add_prefix_space {
            initial_token.text.insert(0, ' ');
            initial_token.reference_offsets.insert(0, 0);
        };
        let mut tokens: Vec<Token> = split_on_special_tokens(initial_token.as_ref(), &self.vocab)
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

        String::from_utf8_lossy(&tokens).to_string()
    }

    fn build_input_with_special_tokens(
        &self,
        tokens_ids_with_offsets_1: TokenIdsWithOffsets,
        tokens_ids_with_offsets_2: Option<TokenIdsWithOffsets>,
    ) -> TokenIdsWithSpecialTokens {
        let mut output: Vec<i64> = vec![];
        let mut token_segment_ids: Vec<i8> = vec![];
        let mut special_tokens_mask: Vec<i8> = vec![];
        let mut offsets: Vec<Option<Offset>> = vec![];
        let mut original_offsets: Vec<Vec<OffsetSize>> = vec![];
        let mut mask: Vec<Mask> = vec![];
        special_tokens_mask.push(1);
        special_tokens_mask.extend(vec![0; tokens_ids_with_offsets_1.ids.len()]);
        special_tokens_mask.push(1);
        token_segment_ids.extend(vec![0; tokens_ids_with_offsets_1.ids.len() + 2]);
        output.push(self.vocab.token_to_id(RobertaVocab::cls_value()));
        output.extend(tokens_ids_with_offsets_1.ids);
        output.push(self.vocab.token_to_id(RobertaVocab::sep_value()));
        offsets.push(None);
        offsets.extend(tokens_ids_with_offsets_1.offsets);
        offsets.push(None);
        original_offsets.push(vec![]);
        original_offsets.extend(tokens_ids_with_offsets_1.reference_offsets);
        original_offsets.push(vec![]);
        mask.push(Mask::Special);
        mask.extend(tokens_ids_with_offsets_1.masks);
        mask.push(Mask::Special);
        if let Some(tokens_ids_with_offsets_2_value) = tokens_ids_with_offsets_2 {
            let length = tokens_ids_with_offsets_2_value.ids.len();
            special_tokens_mask.push(1);
            special_tokens_mask.extend(vec![0; length]);
            special_tokens_mask.push(1);
            token_segment_ids.push(0);
            token_segment_ids.extend(vec![1; length + 1]);
            output.push(self.vocab.token_to_id(RobertaVocab::sep_value()));
            output.extend(tokens_ids_with_offsets_2_value.ids);
            output.push(self.vocab.token_to_id(RobertaVocab::sep_value()));
            offsets.push(None);
            offsets.extend(tokens_ids_with_offsets_2_value.offsets);
            original_offsets.extend(tokens_ids_with_offsets_2_value.reference_offsets);
            offsets.push(None);
            original_offsets.push(vec![]);
            mask.extend(tokens_ids_with_offsets_2_value.masks);
            mask.push(Mask::Special);
        }
        TokenIdsWithSpecialTokens {
            token_ids: output,
            segment_ids: token_segment_ids,
            special_tokens_mask,
            token_offsets: offsets,
            reference_offsets: original_offsets,
            mask,
        }
    }
}

impl MultiThreadedTokenizer<RobertaVocab> for RobertaTokenizer {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::base_tokenizer::{TokenizedInput, TruncationStrategy};
    use crate::vocab::base_vocab::swap_key_values;
    use crate::vocab::RobertaVocab;
    use std::collections::HashMap;

    fn generate_test_vocab() -> RobertaVocab {
        let values: HashMap<String, i64> = [
            ("t".to_owned(), 0),
            ("h".to_owned(), 1),
            ("a@@".to_owned(), 2),
            ("n".to_owned(), 3),
            ("Ġthe".to_owned(), 4),
            ("Ġ".to_owned(), 5),
            ("<unk>".to_owned(), 6),
            ("o@@".to_owned(), 7),
            ("<s>".to_owned(), 8),
            ("</s>".to_owned(), 9),
            ("<pad>".to_owned(), 10),
            ("<mask>".to_owned(), 11),
            ("Ġear".to_owned(), 12),
            ("th".to_owned(), 13),
        ]
        .iter()
        .cloned()
        .collect();

        let special_values: HashMap<String, i64> = [
            ("<unk>".to_owned(), 6),
            ("<s>".to_owned(), 8),
            ("</s>".to_owned(), 9),
            ("<pad>".to_owned(), 10),
            ("<mask>".to_owned(), 11),
        ]
        .iter()
        .cloned()
        .collect();

        let indices = swap_key_values(&values);
        let special_indices = swap_key_values(&special_values);

        RobertaVocab {
            values,
            indices,
            unknown_value: "<unk>",
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
    fn test_roberta_tokenizer() {
        //        Given
        let vocab = generate_test_vocab();
        let merges = generate_test_merges();
        let roberta_tokenizer: RobertaTokenizer =
            RobertaTokenizer::from_existing_vocab_and_merges(vocab, merges, true, false);
        let test_tuples = [
            (
                "The Earth",
                vec!["the", "Ġear", "th"],
                vec![
                    Some(Offset { begin: 0, end: 3 }),
                    Some(Offset { begin: 3, end: 7 }),
                    Some(Offset { begin: 7, end: 9 }),
                ],
                vec![vec![0, 1, 2], vec![3, 4, 5, 6], vec![7, 8]],
                vec![Mask::None, Mask::Begin, Mask::Continuation],
            ),
            ("", vec![], vec![], vec![], vec![]),
            (
                "✿",
                vec!["â", "ľ", "¿"],
                vec![
                    Some(Offset { begin: 0, end: 1 }),
                    Some(Offset { begin: 0, end: 1 }),
                    Some(Offset { begin: 0, end: 1 }),
                ],
                vec![vec![0], vec![0], vec![0]],
                vec![Mask::Begin, Mask::Continuation, Mask::Continuation],
            ),
        ];
        let source_texts: Vec<&str> = test_tuples.iter().map(|v| v.0).collect();
        let expected_results: Vec<Vec<&str>> = test_tuples.iter().map(|v| v.1.clone()).collect();

        //        When & Then
        for (
            source_text,
            expected_tokens,
            expected_offsets,
            expected_original_offsets,
            expected_mask,
        ) in test_tuples.iter()
        {
            let tokens_with_offsets = roberta_tokenizer.tokenize_with_offsets(*source_text);
            assert_eq!(tokens_with_offsets.tokens, *expected_tokens);
            assert_eq!(tokens_with_offsets.offsets, *expected_offsets);
            assert_eq!(
                tokens_with_offsets.reference_offsets,
                *expected_original_offsets
            );
            assert_eq!(tokens_with_offsets.masks, *expected_mask);
        }

        assert_eq!(
            MultiThreadedTokenizer::tokenize_list(&roberta_tokenizer, &source_texts),
            expected_results
        );
    }

    #[test]
    fn test_roberta_tokenizer_no_lower_casing() {
        //        Given
        let vocab = generate_test_vocab();
        let merges = generate_test_merges();
        let roberta_tokenizer: RobertaTokenizer =
            RobertaTokenizer::from_existing_vocab_and_merges(vocab, merges, false, true);
        let test_tuples = [
            (
                "The Earth",
                vec!["Ġ", "T", "he", "Ġ", "E", "a", "r", "th"],
                vec![
                    Some(Offset { begin: 0, end: 1 }),
                    Some(Offset { begin: 0, end: 1 }),
                    Some(Offset { begin: 1, end: 3 }),
                    Some(Offset { begin: 3, end: 4 }),
                    Some(Offset { begin: 4, end: 5 }),
                    Some(Offset { begin: 5, end: 6 }),
                    Some(Offset { begin: 6, end: 7 }),
                    Some(Offset { begin: 7, end: 9 }),
                ], /* note: first inserted whitespace has offset (0,0), which will map to Option::None in further encoding */
                vec![
                    vec![0],
                    vec![0],
                    vec![1, 2],
                    vec![3],
                    vec![4],
                    vec![5],
                    vec![6],
                    vec![7, 8],
                ],
                vec![
                    Mask::Begin,
                    Mask::Continuation,
                    Mask::Continuation,
                    Mask::Begin,
                    Mask::Continuation,
                    Mask::Continuation,
                    Mask::Continuation,
                    Mask::Continuation,
                ],
            ),
            ("", vec![], vec![], vec![], vec![]),
            (
                "✿",
                vec!["Ġ", "â", "ľ", "¿"],
                vec![
                    Some(Offset { begin: 0, end: 1 }),
                    Some(Offset { begin: 0, end: 1 }),
                    Some(Offset { begin: 0, end: 1 }),
                    Some(Offset { begin: 0, end: 1 }),
                ],
                vec![vec![0], vec![0], vec![0], vec![0]],
                vec![
                    Mask::Begin,
                    Mask::Continuation,
                    Mask::Continuation,
                    Mask::Continuation,
                ],
            ),
        ];
        let source_texts: Vec<&str> = test_tuples.iter().map(|v| v.0).collect();
        let expected_results: Vec<Vec<&str>> = test_tuples.iter().map(|v| v.1.clone()).collect();

        //        When & Then
        for (
            source_text,
            expected_tokens,
            expected_offsets,
            expected_original_offsets,
            expected_mask,
        ) in test_tuples.iter()
        {
            let tokens_with_offsets = roberta_tokenizer.tokenize_with_offsets(*source_text);
            assert_eq!(tokens_with_offsets.tokens, *expected_tokens);
            assert_eq!(tokens_with_offsets.offsets, *expected_offsets);
            assert_eq!(
                tokens_with_offsets.reference_offsets,
                *expected_original_offsets
            );
            assert_eq!(tokens_with_offsets.masks, *expected_mask);
        }

        assert_eq!(
            MultiThreadedTokenizer::tokenize_list(&roberta_tokenizer, &source_texts),
            expected_results
        );
    }

    #[test]
    fn test_encode() {
        //        Given
        let vocab = generate_test_vocab();
        let merges = generate_test_merges();
        let roberta_tokenizer: RobertaTokenizer =
            RobertaTokenizer::from_existing_vocab_and_merges(vocab, merges, true, true);
        let truncation_strategy = TruncationStrategy::LongestFirst;
        let test_tuples = [
            (
                "the earth",
                TokenizedInput {
                    token_ids: vec![8, 4, 12, 13, 9],
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
                        vec![0, 0, 1, 2],
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
            ),
            (
                "✿",
                TokenizedInput {
                    token_ids: vec![8, 5, 6, 6, 6, 9],
                    segment_ids: vec![0, 0, 0, 0, 0, 0],
                    special_tokens_mask: vec![1, 0, 0, 0, 0, 1],
                    overflowing_tokens: vec![],
                    num_truncated_tokens: 0,
                    token_offsets: vec![
                        None,
                        Some(Offset { begin: 0, end: 1 }),
                        Some(Offset { begin: 0, end: 1 }),
                        Some(Offset { begin: 0, end: 1 }),
                        Some(Offset { begin: 0, end: 1 }),
                        None,
                    ],
                    reference_offsets: vec![vec![], vec![0], vec![0], vec![0], vec![0], vec![]],
                    mask: vec![
                        Mask::Special,
                        Mask::Begin,
                        Mask::Continuation,
                        Mask::Continuation,
                        Mask::Continuation,
                        Mask::Special,
                    ],
                },
            ),
            (
                "",
                TokenizedInput {
                    token_ids: vec![8, 9],
                    segment_ids: vec![0, 0],
                    special_tokens_mask: vec![1, 1],
                    overflowing_tokens: vec![],
                    num_truncated_tokens: 0,
                    token_offsets: vec![None, None],
                    reference_offsets: vec![vec![], vec![]],
                    mask: vec![Mask::Special, Mask::Special],
                },
            ),
        ];
        let source_texts: Vec<&str> = test_tuples.iter().map(|v| v.0).collect();
        let expected_results: Vec<TokenizedInput> =
            test_tuples.iter().map(|v| v.1.clone()).collect();

        //        When & Then
        for (source_text, expected_result) in test_tuples.iter() {
            assert_eq!(
                roberta_tokenizer.encode(source_text, None, 128, &truncation_strategy, 0),
                *expected_result
            );
        }
        assert_eq!(
            MultiThreadedTokenizer::encode_list(
                &roberta_tokenizer,
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
        let roberta_tokenizer: RobertaTokenizer =
            RobertaTokenizer::from_existing_vocab_and_merges(vocab, merges, true, true);
        let skip_special_tokens = false;
        let clean_up_tokenization_spaces = false;
        let test_tuples = [(vec![8, 4, 12, 13, 9], "<s> the earth</s>")];
        let source_ids: Vec<Vec<i64>> = test_tuples.iter().map(|v| v.0.clone()).collect_vec();
        let expected_results: Vec<&str> = test_tuples.iter().map(|v| v.1).collect_vec();

        //        When & Then
        for (source_ids, expected_result) in test_tuples.iter() {
            assert_eq!(
                roberta_tokenizer.decode(
                    source_ids,
                    skip_special_tokens,
                    clean_up_tokenization_spaces,
                ),
                *expected_result
            );
        }
        assert_eq!(
            Tokenizer::decode_list(
                &roberta_tokenizer,
                &source_ids,
                skip_special_tokens,
                clean_up_tokenization_spaces,
            ),
            expected_results
        );
    }
}
