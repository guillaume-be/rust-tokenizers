// Copyright 2018 The Google AI Language Team Authors
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

use crate::preprocessing::tokenizer::base_tokenizer::{MultiThreadedTokenizer, BaseTokenizer, Tokenizer, Offset, TokenRef, Token, Mask};
use std::sync::Arc;
use crate::preprocessing::tokenizer::tokenization_utils::tokenize_wordpiece;
use crate::preprocessing::vocab::base_vocab::Vocab;
use crate::BertVocab;

pub struct BertTokenizer {
    vocab: Arc<BertVocab>,
    base_tokenizer: BaseTokenizer<BertVocab>,
}

impl BertTokenizer {
    pub fn from_file(path: &str, lower_case: bool) -> BertTokenizer {
        let vocab = Arc::new(BertVocab::from_file(path));
        let strip_accents = lower_case;
        let base_tokenizer = BaseTokenizer::from_existing_vocab(vocab.clone(), lower_case, strip_accents);
        BertTokenizer { vocab, base_tokenizer }
    }

    pub fn from_existing_vocab(vocab: Arc<BertVocab>, lower_case: bool) -> BertTokenizer {
        let strip_accents = lower_case;
        let base_tokenizer = BaseTokenizer::from_existing_vocab(vocab.clone(), lower_case, strip_accents);
        BertTokenizer { vocab, base_tokenizer }
    }
}

impl Tokenizer<BertVocab> for BertTokenizer {
    fn vocab(&self) -> &BertVocab {
        self.vocab.as_ref()
    }

    fn tokenize_to_tokens(&self, initial_token: TokenRef) -> Vec<Token> {
        //the base tokenizers does most of the work, we simply add a wordpiece tokenizer on top
        self.base_tokenizer.tokenize_to_tokens(initial_token).into_iter().map(|token| {
            tokenize_wordpiece(token.token_ref(), self.vocab.as_ref(), 100)
        }).flatten().collect()
    }

    fn convert_tokens_to_string(&self, tokens: Vec<String>) -> String {
        tokens.join(" ").replace(" ##", "").trim().to_owned()
    }

    fn build_input_with_special_tokens(&self, tokens_1: Vec<i64>, tokens_2: Option<Vec<i64>>, offsets_1: Vec<Offset>, offsets_2: Option<Vec<Offset>>, mask_1: Vec<Mask>, mask_2: Option<Vec<Mask>>) -> (Vec<i64>, Vec<i8>, Vec<i8>, Vec<Option<Offset>>, Vec<Mask>) {
        let mut output: Vec<i64> = vec!();
        let mut token_segment_ids: Vec<i8> = vec!();
        let mut special_tokens_mask: Vec<i8> = vec!();
        let mut offsets: Vec<Option<Offset>> = vec!();
        let mut mask: Vec<Mask> = vec!();
        special_tokens_mask.push(1);
        special_tokens_mask.extend(vec![0; tokens_1.len()]);
        special_tokens_mask.push(1);
        token_segment_ids.extend(vec![0; tokens_1.len() + 2]);
        output.push(self.vocab.token_to_id(BertVocab::cls_value()));
        output.extend(tokens_1);
        output.push(self.vocab.token_to_id(BertVocab::sep_value()));
        offsets.push(None);
        offsets.extend(offsets_1.into_iter().map(|offset| offset.into_option()).collect::<Vec<Option<Offset>>>());
        offsets.push(None);
        mask.push(Mask::Special);
        mask.extend(mask_1);
        mask.push(Mask::Special);
        if let Some(add_tokens) = tokens_2 {
            let length = add_tokens.len();
            special_tokens_mask.extend(vec![0; length]);
            special_tokens_mask.push(1);
            token_segment_ids.extend(vec![1; length + 1]);
            output.extend(add_tokens);
            output.push(self.vocab.token_to_id(BertVocab::sep_value()));
            if let Some(add_offsets) = offsets_2 {
                offsets.extend(add_offsets.into_iter().map(|offset| offset.into_option()).collect::<Vec<Option<Offset>>>());
            } else {
                offsets.extend(vec![None; length]);
            }
            offsets.push(None);
            if let Some(mask_2) = mask_2 {
                mask.extend(mask_2)
            } else {
                mask.extend(vec![Mask::None; length]);
            }
            mask.push(Mask::Special);
        }
        (output, token_segment_ids, special_tokens_mask, offsets, mask)
    }
}

impl MultiThreadedTokenizer<BertVocab> for BertTokenizer {}


//==============================
// Unit tests
//==============================
#[cfg(test)]
mod tests {
    use super::*;
    use crate::BertVocab;
    use std::collections::HashMap;
    use crate::preprocessing::tokenizer::base_tokenizer::{TruncationStrategy, TokenizedInput};
    use crate::preprocessing::vocab::base_vocab::swap_key_values;
    use itertools::Itertools;

    fn generate_test_vocab() -> BertVocab {
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
            ("##ble".to_owned(), 13)
        ].iter().cloned().collect();

        let special_values: HashMap<String, i64> = [
            ("[UNK]".to_owned(), 2),
            ("[CLS]".to_owned(), 4),
            ("[SEP]".to_owned(), 5),
            ("[MASK]".to_owned(), 6),
            ("[PAD]".to_owned(), 10)
        ].iter().cloned().collect();

        let indices = swap_key_values(&values);
        let special_indices = swap_key_values(&special_values);

        BertVocab { values, indices, unknown_value: "[UNK]", special_values, special_indices }
    }

    #[test]
    fn test_bert_tokenizer() {
//        Given
        let vocab = Arc::new(generate_test_vocab());
        let bert_tokenizer: BertTokenizer = BertTokenizer::from_existing_vocab(vocab, true);
        let test_tuples = [
            (
                "Hello [MASK] world!",
                vec!("hello", "[MASK]", "world", "!")
            ),
            (
                "Hello, unaffable world!",
                vec!("hello", "[UNK]", "una", "##ffa", "##ble", "world", "!")
            ),
            (
                "[UNK]中华人民共和国 [PAD] asdf",
                vec!("[UNK]", "中", "华", "人", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[PAD]", "[UNK]")
            )
        ];
        let source_texts: Vec<&str> = test_tuples.iter().map(|v| v.0).collect();
        let expected_results: Vec<Vec<&str>> = test_tuples.iter().map(|v| v.1.clone()).collect();

//        When & Then
        for (source_text, expected_result) in test_tuples.iter() {
            assert_eq!(bert_tokenizer.tokenize(*source_text), *expected_result);
        }

        assert_eq!(Tokenizer::tokenize_list(&bert_tokenizer, source_texts.clone()), expected_results);
        assert_eq!(MultiThreadedTokenizer::tokenize_list(&bert_tokenizer, source_texts.clone()), expected_results);
    }

    #[test]
    fn test_bert_tokenizer_no_lower_casing() {
//        Given
        let vocab = Arc::new(generate_test_vocab());
        let bert_tokenizer: BertTokenizer = BertTokenizer::from_existing_vocab(vocab, false);
        let test_tuples = [
            (
                "Hello [MASK] world!",
                vec!("[UNK]", "[MASK]", "world", "!")
            ),
            (
                "Hello, unaffable world!",
                vec!("[UNK]", "[UNK]", "una", "##ffa", "##ble", "world", "!")
            ),
            (
                "[UNK]中华人民共和国 [PAD] asdf",
                vec!("[UNK]", "中", "华", "人", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[PAD]", "[UNK]")
            )
        ];
        let source_texts: Vec<&str> = test_tuples.iter().map(|v| v.0).collect();
        let expected_results: Vec<Vec<&str>> = test_tuples.iter().map(|v| v.1.clone()).collect();

//        When & Then
        for (source_text, expected_result) in test_tuples.iter() {
            assert_eq!(bert_tokenizer.tokenize(*source_text), *expected_result);
        }

        assert_eq!(Tokenizer::tokenize_list(&bert_tokenizer, source_texts.clone()), expected_results);
        assert_eq!(MultiThreadedTokenizer::tokenize_list(&bert_tokenizer, source_texts.clone()), expected_results);
    }

    #[test]
    fn test_encode() {
//        Given
        let vocab = Arc::new(generate_test_vocab());
        let bert_tokenizer: BertTokenizer = BertTokenizer::from_existing_vocab(vocab, true);
        let truncation_strategy = TruncationStrategy::LongestFirst;
        let test_tuples = [
            (
                "hello[MASK] world!",
                TokenizedInput {
                    token_ids: vec!(4, 0, 6, 1, 3, 5),
                    segment_ids: vec!(0, 0, 0, 0, 0, 0),
                    special_tokens_mask: vec!(1, 0, 0, 0, 0, 1),
                    overflowing_tokens: vec!(),
                    num_truncated_tokens: 0,
                    token_offsets: vec![None, Some(Offset { begin: 0, end: 5 }), Some(Offset { begin: 5, end: 11 }), Some(Offset { begin: 12, end: 17 }), Some(Offset { begin: 17, end: 18 }), None],
                    mask:
                    vec!(Mask::Special, Mask::None, Mask::Special, Mask::None, Mask::Punctuation, Mask::Special),
                }
            ),
            (
                "hello, unaffable world!",
                TokenizedInput {
                    token_ids: vec!(4, 0, 2, 11, 12, 13, 1, 3, 5),
                    segment_ids: vec!(0, 0, 0, 0, 0, 0, 0, 0, 0),
                    special_tokens_mask: vec!(1, 0, 0, 0, 0, 0, 0, 0, 1),
                    overflowing_tokens: vec!(),
                    num_truncated_tokens: 0,
                    token_offsets: vec!(
                        None, Some(Offset { begin: 0, end: 5 }), Some(Offset { begin: 5, end: 6 }), Some(Offset { begin: 7, end: 10 }), Some(Offset { begin: 10, end: 13 }), Some(Offset { begin: 13, end: 16 }), Some(Offset { begin: 17, end: 22 }), Some(Offset { begin: 22, end: 23 }), None
                    ),
                    mask: vec!(Mask::Special, Mask::None, Mask::Unknown, Mask::Begin, Mask::Continuation, Mask::Continuation, Mask::None, Mask::Punctuation, Mask::Special),
                }
            ),
            (
                "[UNK]中华人民共和国 [PAD] asdf",
                TokenizedInput {
                    token_ids: vec!(4, 2, 7, 8, 9, 2, 2, 2, 2, 10, 2, 5),
                    segment_ids: vec!(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                    special_tokens_mask: vec!(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
                    overflowing_tokens: vec!(),
                    num_truncated_tokens: 0,
                    token_offsets: vec!(
                        None, Some(Offset { begin: 0, end: 5 }), Some(Offset { begin: 5, end: 6 }), Some(Offset { begin: 6, end: 7 }), Some(Offset { begin: 7, end: 8 }), Some(Offset { begin: 8, end: 9 }), Some(Offset { begin: 9, end: 10 }), Some(Offset { begin: 10, end: 11 }), Some(Offset { begin: 11, end: 12 }), Some(Offset { begin: 13, end: 18 }), Some(Offset { begin: 19, end: 23 }), None
                    ),
                    mask: vec!(Mask::Special, Mask::Unknown, Mask::CJK, Mask::CJK, Mask::CJK, Mask::Unknown, Mask::Unknown, Mask::Unknown, Mask::Unknown, Mask::Special, Mask::Unknown, Mask::Special),
                }
            )
        ];
        let source_texts: Vec<&str> = test_tuples.iter().map(|v| v.0).collect();
        let expected_results: Vec<TokenizedInput> = test_tuples.iter().map(|v| v.1.clone()).collect();

//        When & Then
        for (source_text, expected_result) in test_tuples.iter() {
            let tokenized_input = bert_tokenizer.encode(source_text, None, 128, &truncation_strategy, 0);
            assert_eq!(tokenized_input.token_ids.len(), tokenized_input.token_offsets.len(), "Tokens and offsets must have same length");
            assert_eq!(tokenized_input, *expected_result);
        }
        assert_eq!(Tokenizer::encode_list(&bert_tokenizer, source_texts.clone(), 128, &truncation_strategy, 0), expected_results);
        assert_eq!(MultiThreadedTokenizer::encode_list(&bert_tokenizer, source_texts.clone(), 128, &truncation_strategy, 0), expected_results);
    }

    #[test]
    fn test_encode_sentence_pair() {
//        Given
        let vocab = Arc::new(generate_test_vocab());
        let bert_tokenizer: BertTokenizer = BertTokenizer::from_existing_vocab(vocab, true);
        let truncation_strategy = TruncationStrategy::LongestFirst;
        let test_tuples = [
//            No truncation required
            (
                ("hello world", "This is the second sentence"),
                TokenizedInput {
                    token_ids: vec!(4, 0, 1, 5, 2, 2, 2, 2, 2, 5),
                    segment_ids: vec!(0, 0, 0, 0, 1, 1, 1, 1, 1, 1),
                    special_tokens_mask: vec!(1, 0, 0, 1, 0, 0, 0, 0, 0, 1),
                    overflowing_tokens: vec!(),
                    num_truncated_tokens: 0,
                    token_offsets: vec!(
                        None, Some(Offset { begin: 0, end: 5 }), Some(Offset { begin: 6, end: 11 }), None, Some(Offset { begin: 0, end: 4 }), Some(Offset { begin: 5, end: 7 }), Some(Offset { begin: 8, end: 11 }), Some(Offset { begin: 12, end: 18 }), Some(Offset { begin: 19, end: 27 }), None
                    ),
                    mask: vec!(Mask::Special, Mask::None, Mask::None, Mask::Special, Mask::Unknown, Mask::Unknown, Mask::Unknown, Mask::Unknown, Mask::Unknown, Mask::Special),
                }
            ),
//            Truncation of sentence 2 (longest)
            (
                ("hello world", "!This is the second sentence!!!"),
                TokenizedInput {
                    token_ids: vec!(4, 0, 1, 5, 3, 2, 2, 2, 2, 5),
                    segment_ids: vec!(0, 0, 0, 0, 1, 1, 1, 1, 1, 1),
                    special_tokens_mask: vec!(1, 0, 0, 1, 0, 0, 0, 0, 0, 1),
                    overflowing_tokens: vec!(),
                    num_truncated_tokens: 4,
                    token_offsets: vec!(
                        None, Some(Offset { begin: 0, end: 5 }), Some(Offset { begin: 6, end: 11 }), None, Some(Offset { begin: 0, end: 1 }), Some(Offset { begin: 1, end: 5 }), Some(Offset { begin: 6, end: 8 }), Some(Offset { begin: 9, end: 12 }), Some(Offset { begin: 13, end: 19 }), None
                    ),
                    mask: vec!(Mask::Special, Mask::None, Mask::None, Mask::Special, Mask::Punctuation, Mask::Unknown, Mask::Unknown, Mask::Unknown, Mask::Unknown, Mask::Special),
                }
            ),
//            Truncation of sentence 1 (longest)
            (
                ("[UNK] hello  hello  hello  hello  hello  hello  hello  hello  hello  hello  hello", "!!!"),
                TokenizedInput {
                    token_ids: vec!(4, 2, 0, 0, 0, 5, 3, 3, 3, 5),
                    segment_ids: vec!(0, 0, 0, 0, 0, 0, 1, 1, 1, 1),
                    special_tokens_mask: vec!(1, 0, 0, 0, 0, 1, 0, 0, 0, 1),
                    overflowing_tokens: vec!(0, 0, 0, 0, 0, 0, 0, 0),
                    num_truncated_tokens: 8,
                    token_offsets: vec!(
                        None, Some(Offset { begin: 0, end: 5 }), Some(Offset { begin: 6, end: 11 }), Some(Offset { begin: 13, end: 18 }), Some(Offset { begin: 20, end: 25 }), None, Some(Offset { begin: 0, end: 1 }), Some(Offset { begin: 1, end: 2 }), Some(Offset { begin: 2, end: 3 }), None
                    ),
                    mask: vec!(Mask::Special, Mask::Unknown, Mask::None, Mask::None, Mask::None, Mask::Special, Mask::Punctuation, Mask::Punctuation, Mask::Punctuation, Mask::Special),
                }
            ),
//            Truncation of both sentences (longest)
            (
                ("[UNK] hello  hello  hello  hello  hello", "!!!!!!!!"),
                TokenizedInput {
                    token_ids: vec!(4, 2, 0, 0, 5, 3, 3, 3, 3, 5),
                    segment_ids: vec!(0, 0, 0, 0, 0, 1, 1, 1, 1, 1),
                    special_tokens_mask: vec!(1, 0, 0, 0, 1, 0, 0, 0, 0, 1),
                    overflowing_tokens: vec!(0, 0, 0),
                    num_truncated_tokens: 7,
                    token_offsets: vec!(
                        None, Some(Offset { begin: 0, end: 5 }), Some(Offset { begin: 6, end: 11 }), Some(Offset { begin: 13, end: 18 }), None, Some(Offset { begin: 0, end: 1 }), Some(Offset { begin: 1, end: 2 }), Some(Offset { begin: 2, end: 3 }), Some(Offset { begin: 3, end: 4 }), None
                    ),
                    mask: vec!(Mask::Special, Mask::Unknown, Mask::None, Mask::None, Mask::Special, Mask::Punctuation, Mask::Punctuation, Mask::Punctuation, Mask::Punctuation, Mask::Special),
                }
            )
        ];
        let source_texts: Vec<(&str, &str)> = test_tuples.iter().map(|v| v.0).collect();
        let expected_results: Vec<TokenizedInput> = test_tuples.iter().map(|v| v.1.clone()).collect();

//        When & Then
        for (source_text, expected_result) in test_tuples.iter() {
            assert_eq!(bert_tokenizer.encode(source_text.0, Some(source_text.1), 10, &truncation_strategy, 0),
                       *expected_result);
        }
        assert_eq!(Tokenizer::encode_pair_list(&bert_tokenizer, source_texts.clone(), 10, &truncation_strategy, 0), expected_results);
        assert_eq!(MultiThreadedTokenizer::encode_pair_list(&bert_tokenizer, source_texts.clone(), 10, &truncation_strategy, 0), expected_results);
    }

    #[test]
    fn test_decode() {
//        Given
        let vocab = Arc::new(generate_test_vocab());
        let bert_tokenizer: BertTokenizer = BertTokenizer::from_existing_vocab(vocab, true);
        let skip_special_tokens = false;
        let clean_up_tokenization_spaces = false;
        let test_tuples = [
            (
                vec!(0, 1, 3),
                "hello world !",
            ),
            (
                vec!(4, 0, 2, 11, 12, 13, 1, 3, 5),
                "[CLS] hello [UNK] unaffable world ! [SEP]",
            )
        ];
        let source_ids: Vec<Vec<i64>> = test_tuples.iter().map(|v| v.0.clone()).collect_vec();
        let expected_results: Vec<&str> = test_tuples.iter().map(|v| v.1.clone()).collect_vec();

//        When & Then
        for (source_ids, expected_result) in test_tuples.iter() {
            assert_eq!(bert_tokenizer.decode(source_ids.clone(), skip_special_tokens, clean_up_tokenization_spaces),
                       *expected_result);
        }
        assert_eq!(Tokenizer::decode_list(&bert_tokenizer, source_ids.clone(), skip_special_tokens, clean_up_tokenization_spaces), expected_results);
        assert_eq!(MultiThreadedTokenizer::decode_list(&bert_tokenizer, source_ids.clone(), skip_special_tokens, clean_up_tokenization_spaces), expected_results);
    }

    #[test]
    fn test_decode_skip_special_tokens() {
//        Given
        let vocab = Arc::new(generate_test_vocab());
        let bert_tokenizer: BertTokenizer = BertTokenizer::from_existing_vocab(vocab, true);
        let skip_special_tokens = true;
        let clean_up_tokenization_spaces = true;
        let test_tuples = [
            (
                vec!(0, 1, 3),
                "hello world!",
            ),
            (
                vec!(4, 0, 2, 11, 12, 13, 1, 3, 5),
                "hello unaffable world!",
            )
        ];
        let source_ids: Vec<Vec<i64>> = test_tuples.iter().map(|v| v.0.clone()).collect_vec();
        let expected_results: Vec<&str> = test_tuples.iter().map(|v| v.1.clone()).collect_vec();

//        When & Then
        for (source_ids, expected_result) in test_tuples.iter() {
            assert_eq!(bert_tokenizer.decode(source_ids.clone(), skip_special_tokens, clean_up_tokenization_spaces),
                       *expected_result);
        }
        assert_eq!(Tokenizer::decode_list(&bert_tokenizer, source_ids.clone(), skip_special_tokens, clean_up_tokenization_spaces), expected_results);
        assert_eq!(MultiThreadedTokenizer::decode_list(&bert_tokenizer, source_ids.clone(), skip_special_tokens, clean_up_tokenization_spaces), expected_results);
    }
}
