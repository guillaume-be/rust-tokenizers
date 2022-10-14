// Copyright 2020 The Microsoft Authors and The HuggingFace Inc. team.
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

use std::path::Path;

use crate::error::TokenizerError;
use crate::tokenizer::base_tokenizer::{
    BaseTokenizer, Mask, MultiThreadedTokenizer, Offset, OffsetSize, Token, TokenIdsWithOffsets,
    TokenIdsWithSpecialTokens, TokenRef, Tokenizer,
};
use crate::tokenizer::tokenization_utils::tokenize_wordpiece;
use crate::vocab::{ProphetNetVocab, Vocab};

/// # ProphetNet tokenizer
/// ProphetNet tokenizer performing:
/// - BaseTokenizer tokenization (see `BaseTokenizer` for more details)
/// - WordPiece tokenization
pub struct ProphetNetTokenizer {
    vocab: ProphetNetVocab,
    base_tokenizer: BaseTokenizer<ProphetNetVocab>,
}

impl ProphetNetTokenizer {
    /// Create a new instance of a `ProphetNetTokenizer`.
    /// Expects a vocabulary flat-file as an input.
    ///
    /// # Parameters
    /// - path (`&str`): path to the vocabulary file
    /// - lower_case (`bool`): flag indicating if the text should be lower-cased as part of the tokenization
    /// - strip_accents (`bool`): flag indicating if accents should be stripped from the text
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_tokenizers::tokenizer::{ProphetNetTokenizer, Tokenizer};
    ///
    /// let strip_accents = false;
    /// let lower_case = false;
    /// let path = std::path::Path::new("path/to/vocab/file");
    /// let tokenizer =
    ///     ProphetNetTokenizer::from_file(&path, lower_case, strip_accents).unwrap();
    /// ```
    pub fn from_file(
        path: &Path,
        lower_case: bool,
        strip_accents: bool,
    ) -> Result<ProphetNetTokenizer, TokenizerError> {
        let vocab = ProphetNetVocab::from_file(path)?;
        let base_tokenizer =
            BaseTokenizer::from_existing_vocab(vocab.clone(), lower_case, strip_accents);
        Ok(ProphetNetTokenizer {
            vocab,
            base_tokenizer,
        })
    }

    /// Create a new instance of a `ProphetNetTokenizer`.
    /// Expects a vocabulary flat-file and special token mapping file as inputs.
    ///
    /// # Parameters
    /// - path (`&str`): path to the vocabulary file
    /// - lower_case (`bool`): flag indicating if the text should be lower-cased as part of the tokenization
    /// - strip_accents (`bool`): flag indicating if accents should be stripped from the text
    /// - special_token_mapping_path (`&str`): path to a special token mapping file to overwrite default special tokens
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_tokenizers::tokenizer::{ProphetNetTokenizer, Tokenizer};
    /// use std::path::Path;
    ///
    /// let strip_accents = false;
    /// let lower_case = false;
    /// let tokenizer = ProphetNetTokenizer::from_file_with_special_token_mapping(
    ///     &Path::new("path/to/vocab/file"),
    ///     lower_case,
    ///     strip_accents,
    ///     &Path::new("path/to/special/token/mapping/file"),
    /// )
    /// .unwrap();
    /// ```
    pub fn from_file_with_special_token_mapping(
        path: &Path,
        lower_case: bool,
        strip_accents: bool,
        special_token_mapping_path: &Path,
    ) -> Result<ProphetNetTokenizer, TokenizerError> {
        let vocab = ProphetNetVocab::from_file_with_special_token_mapping(
            path,
            special_token_mapping_path,
        )?;
        let base_tokenizer =
            BaseTokenizer::from_existing_vocab(vocab.clone(), lower_case, strip_accents);
        Ok(ProphetNetTokenizer {
            vocab,
            base_tokenizer,
        })
    }

    /// Create a new instance of a `ProphetNetTokenizer` from an existing vocabulary
    ///
    /// # Parameters
    /// - vocab (`ProphetNetVocab`): ProphetNet vocabulary
    /// - lower_case (`bool`): flag indicating if the text should be lower-cased as part of the tokenization
    /// - strip_accents (`bool`): flag indicating if accents should be stripped from the text
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_tokenizers::tokenizer::{ProphetNetTokenizer, Tokenizer};
    /// use rust_tokenizers::vocab::{ProphetNetVocab, Vocab};
    /// let strip_accents = false;
    /// let lower_case = false;
    /// let path = std::path::Path::new("path/to/vocab/file");
    /// let vocab = ProphetNetVocab::from_file(&path).unwrap();
    ///
    /// let tokenizer = ProphetNetTokenizer::from_existing_vocab(vocab, lower_case, strip_accents);
    /// ```
    pub fn from_existing_vocab(
        vocab: ProphetNetVocab,
        lower_case: bool,
        strip_accents: bool,
    ) -> ProphetNetTokenizer {
        let base_tokenizer =
            BaseTokenizer::from_existing_vocab(vocab.clone(), lower_case, strip_accents);
        ProphetNetTokenizer {
            vocab,
            base_tokenizer,
        }
    }
}

impl Tokenizer<ProphetNetVocab> for ProphetNetTokenizer {
    fn vocab(&self) -> &ProphetNetVocab {
        &self.vocab
    }

    fn tokenize_to_tokens(&self, initial_token: TokenRef) -> Vec<Token> {
        //the base tokenizers does most of the work, we simply add a wordpiece tokenizer on top
        self.base_tokenizer
            .tokenize_to_tokens(initial_token)
            .into_iter()
            .flat_map(|token| tokenize_wordpiece(token.as_ref(), &self.vocab, 100))
            .collect()
    }

    fn convert_tokens_to_string(&self, tokens: Vec<String>) -> String {
        tokens.join(" ").replace(" ##", "").trim().to_owned()
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
        special_tokens_mask.extend(vec![0; tokens_ids_with_offsets_1.ids.len()]);
        special_tokens_mask.push(1);
        token_segment_ids.extend(vec![0; tokens_ids_with_offsets_1.ids.len() + 1]);
        output.extend(tokens_ids_with_offsets_1.ids);
        output.push(self.vocab.token_to_id(self.vocab.get_sep_value()));
        offsets.extend(tokens_ids_with_offsets_1.offsets);
        offsets.push(None);
        original_offsets.extend(tokens_ids_with_offsets_1.reference_offsets);
        original_offsets.push(vec![]);
        mask.extend(tokens_ids_with_offsets_1.masks);
        mask.push(Mask::Special);
        if let Some(tokens_ids_with_offsets_2_value) = tokens_ids_with_offsets_2 {
            let length = tokens_ids_with_offsets_2_value.ids.len();
            special_tokens_mask.extend(vec![0; length]);
            special_tokens_mask.push(1);
            token_segment_ids.extend(vec![1; length + 1]);
            output.extend(tokens_ids_with_offsets_2_value.ids);
            output.push(self.vocab.token_to_id(self.vocab.get_sep_value()));
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

impl MultiThreadedTokenizer<ProphetNetVocab> for ProphetNetTokenizer {}

//==============================
// Unit tests
//==============================
#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::base_tokenizer::TruncationStrategy;
    use crate::vocab::base_vocab::{swap_key_values, SpecialTokenMap};
    use crate::TokenizedInput;
    use itertools::Itertools;
    use std::collections::{HashMap, HashSet};

    fn generate_test_vocab() -> ProphetNetVocab {
        let values: HashMap<String, i64> = [
            ("hello".to_owned(), 0),
            ("world".to_owned(), 1),
            ("[UNK]".to_owned(), 2),
            ("!".to_owned(), 3),
            ("[CLS]".to_owned(), 4),
            ("[SEP]".to_owned(), 5),
            ("[MASK]".to_owned(), 6),
            ("中".to_owned(), 7),
            ("华".to_owned(), 8),
            ("人".to_owned(), 9),
            ("[PAD]".to_owned(), 10),
            ("una".to_owned(), 11),
            ("##ffa".to_owned(), 12),
            ("##ble".to_owned(), 13),
            ("[X_SEP]".to_owned(), 14),
        ]
        .iter()
        .cloned()
        .collect();

        let special_token_map = SpecialTokenMap {
            unk_token: "[UNK]".to_string(),
            pad_token: Some("[PAD]".to_string()),
            bos_token: None,
            sep_token: Some("[SEP]".to_string()),
            cls_token: Some("[CLS]".to_string()),
            eos_token: None,
            mask_token: Some("[MASK]".to_string()),
            additional_special_tokens: Some(HashSet::from(["[X_SEP".into()])),
        };

        let special_values: HashMap<String, i64> = [
            ("[UNK]".to_owned(), 2),
            ("[CLS]".to_owned(), 4),
            ("[SEP]".to_owned(), 5),
            ("[MASK]".to_owned(), 6),
            ("[PAD]".to_owned(), 10),
            ("[X_SEP]".to_owned(), 14),
        ]
        .iter()
        .cloned()
        .collect();

        let indices = swap_key_values(&values);
        let special_indices = swap_key_values(&special_values);

        ProphetNetVocab {
            values,
            indices,
            special_token_map,
            special_values,
            special_indices,
        }
    }

    #[test]
    fn test_prophetneg_tokenizer() {
        //        Given
        let vocab = generate_test_vocab();
        let bert_tokenizer: ProphetNetTokenizer =
            ProphetNetTokenizer::from_existing_vocab(vocab, true, true);
        let test_tuples = [
            ("Hello [MASK] world!", vec!["hello", "[MASK]", "world", "!"]),
            (
                "Hello, unaffable world!",
                vec!["hello", "[UNK]", "una", "##ffa", "##ble", "world", "!"],
            ),
            (
                "[UNK]中华人民共和国 [PAD] asdf",
                vec![
                    "[UNK]", "中", "华", "人", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[PAD]", "[UNK]",
                ],
            ),
        ];
        let source_texts: Vec<&str> = test_tuples.iter().map(|v| v.0).collect();
        let expected_results: Vec<Vec<&str>> = test_tuples.iter().map(|v| v.1.clone()).collect();

        //        When & Then
        for (source_text, expected_result) in test_tuples.iter() {
            assert_eq!(bert_tokenizer.tokenize(*source_text), *expected_result);
        }

        assert_eq!(
            Tokenizer::tokenize_list(&bert_tokenizer, &source_texts),
            expected_results
        );
        assert_eq!(
            MultiThreadedTokenizer::tokenize_list(&bert_tokenizer, &source_texts),
            expected_results
        );
    }

    #[test]
    fn test_prophetnet_tokenizer_no_lower_casing() {
        //        Given
        let vocab = generate_test_vocab();
        let bert_tokenizer: ProphetNetTokenizer =
            ProphetNetTokenizer::from_existing_vocab(vocab, false, false);
        let test_tuples = [
            ("Hello [MASK] world!", vec!["[UNK]", "[MASK]", "world", "!"]),
            (
                "Hello, unaffable world!",
                vec!["[UNK]", "[UNK]", "una", "##ffa", "##ble", "world", "!"],
            ),
            (
                "[UNK]中华人民共和国 [PAD] asdf",
                vec![
                    "[UNK]", "中", "华", "人", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[PAD]", "[UNK]",
                ],
            ),
        ];
        let source_texts: Vec<&str> = test_tuples.iter().map(|v| v.0).collect();
        let expected_results: Vec<Vec<&str>> = test_tuples.iter().map(|v| v.1.clone()).collect();

        //        When & Then
        for (source_text, expected_result) in test_tuples.iter() {
            assert_eq!(bert_tokenizer.tokenize(*source_text), *expected_result);
        }

        assert_eq!(
            Tokenizer::tokenize_list(&bert_tokenizer, &source_texts),
            expected_results
        );
        assert_eq!(
            MultiThreadedTokenizer::tokenize_list(&bert_tokenizer, &source_texts),
            expected_results
        );
    }

    #[test]
    fn test_encode() {
        //        Given
        let vocab = generate_test_vocab();
        let bert_tokenizer: ProphetNetTokenizer =
            ProphetNetTokenizer::from_existing_vocab(vocab, true, true);
        let truncation_strategy = TruncationStrategy::LongestFirst;
        let test_tuples = [
            (
                "hello[MASK] world!",
                TokenizedInput {
                    token_ids: vec![0, 6, 1, 3, 5],
                    segment_ids: vec![0, 0, 0, 0, 0],
                    special_tokens_mask: vec![0, 0, 0, 0, 1],
                    overflowing_tokens: vec![],
                    num_truncated_tokens: 0,
                    token_offsets: vec![
                        Some(Offset { begin: 0, end: 5 }),
                        Some(Offset { begin: 5, end: 11 }),
                        Some(Offset { begin: 12, end: 17 }),
                        Some(Offset { begin: 17, end: 18 }),
                        None,
                    ],
                    reference_offsets: vec![
                        vec![0, 1, 2, 3, 4],
                        vec![5, 6, 7, 8, 9, 10],
                        vec![12, 13, 14, 15, 16],
                        vec![17],
                        vec![],
                    ],
                    mask: vec![
                        Mask::None,
                        Mask::Special,
                        Mask::None,
                        Mask::Punctuation,
                        Mask::Special,
                    ],
                },
            ),
            (
                "hello, unaffable world!",
                TokenizedInput {
                    token_ids: vec![0, 2, 11, 12, 13, 1, 3, 5],
                    segment_ids: vec![0, 0, 0, 0, 0, 0, 0, 0],
                    special_tokens_mask: vec![0, 0, 0, 0, 0, 0, 0, 1],
                    overflowing_tokens: vec![],
                    num_truncated_tokens: 0,
                    token_offsets: vec![
                        Some(Offset { begin: 0, end: 5 }),
                        Some(Offset { begin: 5, end: 6 }),
                        Some(Offset { begin: 7, end: 10 }),
                        Some(Offset { begin: 10, end: 13 }),
                        Some(Offset { begin: 13, end: 16 }),
                        Some(Offset { begin: 17, end: 22 }),
                        Some(Offset { begin: 22, end: 23 }),
                        None,
                    ],
                    reference_offsets: vec![
                        vec![0, 1, 2, 3, 4],
                        vec![5],
                        vec![7, 8, 9],
                        vec![10, 11, 12],
                        vec![13, 14, 15],
                        vec![17, 18, 19, 20, 21],
                        vec![22],
                        vec![],
                    ],
                    mask: vec![
                        Mask::None,
                        Mask::Unknown,
                        Mask::Begin,
                        Mask::Continuation,
                        Mask::Continuation,
                        Mask::None,
                        Mask::Punctuation,
                        Mask::Special,
                    ],
                },
            ),
            (
                "[UNK]中华人民共和国 [PAD] asdf",
                TokenizedInput {
                    token_ids: vec![2, 7, 8, 9, 2, 2, 2, 2, 10, 2, 5],
                    segment_ids: vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    special_tokens_mask: vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    overflowing_tokens: vec![],
                    num_truncated_tokens: 0,
                    token_offsets: vec![
                        Some(Offset { begin: 0, end: 5 }),
                        Some(Offset { begin: 5, end: 6 }),
                        Some(Offset { begin: 6, end: 7 }),
                        Some(Offset { begin: 7, end: 8 }),
                        Some(Offset { begin: 8, end: 9 }),
                        Some(Offset { begin: 9, end: 10 }),
                        Some(Offset { begin: 10, end: 11 }),
                        Some(Offset { begin: 11, end: 12 }),
                        Some(Offset { begin: 13, end: 18 }),
                        Some(Offset { begin: 19, end: 23 }),
                        None,
                    ],
                    reference_offsets: vec![
                        vec![0, 1, 2, 3, 4],
                        vec![5],
                        vec![6],
                        vec![7],
                        vec![8],
                        vec![9],
                        vec![10],
                        vec![11],
                        vec![13, 14, 15, 16, 17],
                        vec![19, 20, 21, 22],
                        vec![],
                    ],
                    mask: vec![
                        Mask::Unknown,
                        Mask::CJK,
                        Mask::CJK,
                        Mask::CJK,
                        Mask::Unknown,
                        Mask::Unknown,
                        Mask::Unknown,
                        Mask::Unknown,
                        Mask::Special,
                        Mask::Unknown,
                        Mask::Special,
                    ],
                },
            ),
        ];
        let source_texts: Vec<&str> = test_tuples.iter().map(|v| v.0).collect();
        let expected_results: Vec<TokenizedInput> =
            test_tuples.iter().map(|v| v.1.clone()).collect();

        //        When & Then
        for (source_text, expected_result) in test_tuples.iter() {
            let tokenized_input =
                bert_tokenizer.encode(source_text, None, 128, &truncation_strategy, 0);
            assert_eq!(
                tokenized_input.token_ids.len(),
                tokenized_input.token_offsets.len(),
                "Tokens and offsets must have same length"
            );
            assert_eq!(tokenized_input, *expected_result);
        }
        assert_eq!(
            Tokenizer::encode_list(&bert_tokenizer, &source_texts, 128, &truncation_strategy, 0),
            expected_results
        );
        assert_eq!(
            MultiThreadedTokenizer::encode_list(
                &bert_tokenizer,
                &source_texts,
                128,
                &truncation_strategy,
                0,
            ),
            expected_results
        );
    }

    #[test]
    fn test_encode_sentence_pair() {
        //        Given
        let vocab = generate_test_vocab();
        let bert_tokenizer: ProphetNetTokenizer =
            ProphetNetTokenizer::from_existing_vocab(vocab, true, true);
        let truncation_strategy = TruncationStrategy::LongestFirst;
        let test_tuples = [
//            No truncation required
            (
                ("hello world", "This is the second sentence"),
                TokenizedInput {
                    token_ids: vec!(0, 1, 5, 2, 2, 2, 2, 2, 5),
                    segment_ids: vec!(0, 0, 0, 1, 1, 1, 1, 1, 1),
                    special_tokens_mask: vec!(0, 0, 1, 0, 0, 0, 0, 0, 1),
                    overflowing_tokens: vec!(),
                    num_truncated_tokens: 0,
                    token_offsets: vec!(
                        Some(Offset { begin: 0, end: 5 }), Some(Offset { begin: 6, end: 11 }), None, Some(Offset { begin: 0, end: 4 }), Some(Offset { begin: 5, end: 7 }), Some(Offset { begin: 8, end: 11 }), Some(Offset { begin: 12, end: 18 }), Some(Offset { begin: 19, end: 27 }), None
                    ),
                    reference_offsets: vec!(vec!(0, 1, 2, 3, 4), vec!(6, 7, 8, 9, 10), vec!(), vec!(0, 1, 2, 3), vec!(5, 6), vec!(8, 9, 10), vec!(12, 13, 14, 15, 16, 17), vec!(19, 20, 21, 22, 23, 24, 25, 26), vec!()),
                    mask: vec!(Mask::None, Mask::None, Mask::Special, Mask::Unknown, Mask::Unknown, Mask::Unknown, Mask::Unknown, Mask::Unknown, Mask::Special),
                }
            ),
//            Truncation of sentence 2 (longest)
            (
                ("hello world", "!This is the second sentence!!!"),
                TokenizedInput {
                    token_ids: vec!(0, 1, 5, 3, 2, 2, 2, 2, 2, 5),
                    segment_ids: vec!(0, 0, 0, 1, 1, 1, 1, 1, 1, 1),
                    special_tokens_mask: vec!(0, 0, 1, 0, 0, 0, 0, 0, 0, 1),
                    overflowing_tokens: vec!(3, 3, 3),
                    num_truncated_tokens: 3,
                    token_offsets: vec!(
                        Some(Offset { begin: 0, end: 5 }), Some(Offset { begin: 6, end: 11 }), None, Some(Offset { begin: 0, end: 1 }), Some(Offset { begin: 1, end: 5 }), Some(Offset { begin: 6, end: 8 }), Some(Offset { begin: 9, end: 12 }), Some(Offset { begin: 13, end: 19 }), Some(Offset { begin: 20, end: 28 }), None
                    ),
                    reference_offsets: vec!(vec!(0, 1, 2, 3, 4), vec!(6, 7, 8, 9, 10), vec!(), vec!(0), vec!(1, 2, 3, 4), vec!(6, 7), vec!(9, 10, 11), vec!(13, 14, 15, 16, 17, 18), vec!(20, 21, 22, 23, 24, 25, 26, 27), vec!()),
                    mask: vec!(Mask::None, Mask::None, Mask::Special, Mask::Punctuation, Mask::Unknown, Mask::Unknown, Mask::Unknown, Mask::Unknown, Mask::Unknown, Mask::Special),
                }
            ),
//            Truncation of sentence 1 (longest)
            (
                ("[UNK] hello  hello  hello  hello  hello  hello  hello  hello  hello  hello  hello", "!!!"),
                TokenizedInput {
                    token_ids: vec!(2, 0, 0, 0, 0, 5, 3, 3, 3, 5),
                    segment_ids: vec!(0, 0, 0, 0, 0, 0, 1, 1, 1, 1),
                    special_tokens_mask: vec!(0, 0, 0, 0, 0, 1, 0, 0, 0, 1),
                    overflowing_tokens: vec!(0, 0, 0, 0, 0, 0, 0),
                    num_truncated_tokens: 7,
                    token_offsets: vec!(
                        Some(Offset { begin: 0, end: 5 }), Some(Offset { begin: 6, end: 11 }), Some(Offset { begin: 13, end: 18 }), Some(Offset { begin: 20, end: 25 }), Some(Offset { begin: 27, end: 32 }), None, Some(Offset { begin: 0, end: 1 }), Some(Offset { begin: 1, end: 2 }), Some(Offset { begin: 2, end: 3 }), None
                    ),
                    reference_offsets: vec!(vec!(0, 1, 2, 3, 4), vec!(6, 7, 8, 9, 10), vec!(13, 14, 15, 16, 17), vec!(20, 21, 22, 23, 24), vec!(27, 28, 29, 30, 31), vec!(), vec!(0), vec!(1), vec!(2), vec!()),
                    mask: vec!(Mask::Unknown, Mask::None, Mask::None, Mask::None, Mask::None, Mask::Special, Mask::Punctuation, Mask::Punctuation, Mask::Punctuation, Mask::Special),
                }
            ),
//            Truncation of both sentences (longest)
            (
                ("[UNK] hello  hello  hello  hello  hello", "!!!!!!!!"),
                TokenizedInput {
                    token_ids: vec!(2, 0, 0, 0, 5, 3, 3, 3, 3, 5),
                    segment_ids: vec!(0, 0, 0, 0, 0, 1, 1, 1, 1, 1),
                    special_tokens_mask: vec!(0, 0, 0, 0, 1, 0, 0, 0, 0, 1),
                    overflowing_tokens: vec!(3, 0, 3, 0, 3, 3),
                    num_truncated_tokens: 6,
                    token_offsets: vec!(
                        Some(Offset { begin: 0, end: 5 }), Some(Offset { begin: 6, end: 11 }), Some(Offset { begin: 13, end: 18 }), Some(Offset { begin: 20, end: 25 }), None, Some(Offset { begin: 0, end: 1 }), Some(Offset { begin: 1, end: 2 }), Some(Offset { begin: 2, end: 3 }), Some(Offset { begin: 3, end: 4 }), None
                    ),
                    reference_offsets: vec!(vec!(0, 1, 2, 3, 4), vec!(6, 7, 8, 9, 10), vec!(13, 14, 15, 16, 17), vec!(20, 21, 22, 23, 24), vec!(), vec!(0), vec!(1), vec!(2), vec!(3), vec!()),
                    mask: vec!(Mask::Unknown, Mask::None, Mask::None, Mask::None, Mask::Special, Mask::Punctuation, Mask::Punctuation, Mask::Punctuation, Mask::Punctuation, Mask::Special),
                }
            )
        ];
        let source_texts: Vec<(&str, &str)> = test_tuples.iter().map(|v| v.0).collect();
        let expected_results: Vec<TokenizedInput> =
            test_tuples.iter().map(|v| v.1.clone()).collect();

        //        When & Then
        for (source_text, expected_result) in test_tuples.iter() {
            assert_eq!(
                bert_tokenizer.encode(
                    source_text.0,
                    Some(source_text.1),
                    10,
                    &truncation_strategy,
                    0,
                ),
                *expected_result
            );
        }
        assert_eq!(
            Tokenizer::encode_pair_list(
                &bert_tokenizer,
                &source_texts,
                10,
                &truncation_strategy,
                0,
            ),
            expected_results
        );
        assert_eq!(
            MultiThreadedTokenizer::encode_pair_list(
                &bert_tokenizer,
                &source_texts,
                10,
                &truncation_strategy,
                0,
            ),
            expected_results
        );
    }

    #[test]
    fn test_decode() {
        //        Given
        let vocab = generate_test_vocab();
        let bert_tokenizer: ProphetNetTokenizer =
            ProphetNetTokenizer::from_existing_vocab(vocab, true, true);
        let skip_special_tokens = false;
        let clean_up_tokenization_spaces = false;
        let test_tuples = [
            (vec![0, 1, 3], "hello world !"),
            (
                vec![0, 2, 11, 12, 13, 1, 3, 5],
                "hello [UNK] unaffable world ! [SEP]",
            ),
        ];
        let source_ids: Vec<Vec<i64>> = test_tuples.iter().map(|v| v.0.clone()).collect_vec();
        let expected_results: Vec<&str> = test_tuples.iter().map(|v| v.1).collect_vec();

        //        When & Then
        for (source_ids, expected_result) in test_tuples.iter() {
            assert_eq!(
                bert_tokenizer.decode(
                    source_ids,
                    skip_special_tokens,
                    clean_up_tokenization_spaces,
                ),
                *expected_result
            );
        }
        assert_eq!(
            Tokenizer::decode_list(
                &bert_tokenizer,
                &source_ids,
                skip_special_tokens,
                clean_up_tokenization_spaces,
            ),
            expected_results
        );
        assert_eq!(
            MultiThreadedTokenizer::decode_list(
                &bert_tokenizer,
                &source_ids,
                skip_special_tokens,
                clean_up_tokenization_spaces,
            ),
            expected_results
        );
    }

    #[test]
    fn test_decode_skip_special_tokens() {
        //        Given
        let vocab = generate_test_vocab();
        let bert_tokenizer: ProphetNetTokenizer =
            ProphetNetTokenizer::from_existing_vocab(vocab, true, true);
        let skip_special_tokens = true;
        let clean_up_tokenization_spaces = true;
        let test_tuples = [
            (vec![0, 1, 3], "hello world!"),
            (vec![0, 2, 11, 12, 13, 1, 3, 5], "hello unaffable world!"),
        ];
        let source_ids: Vec<Vec<i64>> = test_tuples.iter().map(|v| v.0.clone()).collect_vec();
        let expected_results: Vec<&str> = test_tuples.iter().map(|v| v.1).collect_vec();

        //        When & Then
        for (source_ids, expected_result) in test_tuples.iter() {
            assert_eq!(
                bert_tokenizer.decode(
                    source_ids,
                    skip_special_tokens,
                    clean_up_tokenization_spaces,
                ),
                *expected_result
            );
        }
        assert_eq!(
            Tokenizer::decode_list(
                &bert_tokenizer,
                &source_ids,
                skip_special_tokens,
                clean_up_tokenization_spaces,
            ),
            expected_results
        );
        assert_eq!(
            MultiThreadedTokenizer::decode_list(
                &bert_tokenizer,
                &source_ids,
                skip_special_tokens,
                clean_up_tokenization_spaces,
            ),
            expected_results
        );
    }
}
