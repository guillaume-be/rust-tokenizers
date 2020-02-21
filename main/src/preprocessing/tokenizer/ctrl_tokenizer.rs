// Copyright 2018 Salesforce
// Copyright 2018 The HuggingFace Inc. team.
// Copyright 2019 Guillaume Becquin
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use crate::OpenAiGptVocab;
use crate::preprocessing::vocab::base_vocab::Vocab;
use crate::preprocessing::tokenizer::base_tokenizer::Tokenizer;
use std::collections::HashMap;
use crate::preprocessing::tokenizer::tokenization_utils::{ctrl_bpe, split_on_special_tokens};
use std::rc::Rc;
use std::cell::RefCell;
use crate::preprocessing::vocab::bpe_vocab::BpePairVocab;
use regex::Regex;
use itertools::Itertools;


pub struct CtrlTokenizer {
    vocab: Rc<OpenAiGptVocab>,
    bpe_ranks: Rc<BpePairVocab>,
    cache: RefCell<HashMap<String, Vec<String>>>,
    regex_pattern: Regex,
    lower_case: bool,
}

impl CtrlTokenizer {
    pub fn from_file(vocab_path: &str, merges_path: &str, lower_case: bool) -> CtrlTokenizer {
        let vocab = Rc::new(OpenAiGptVocab::from_file(vocab_path));
        let bpe_ranks = Rc::new(BpePairVocab::from_file(merges_path));
        let cache = RefCell::new(HashMap::new());
        let regex_pattern = Regex::new(r"\S+\n?").unwrap();
        CtrlTokenizer { vocab, bpe_ranks, cache, regex_pattern, lower_case }
    }

    pub fn from_existing_vocab_and_merges(vocab: Rc<OpenAiGptVocab>, merges: Rc<BpePairVocab>, lower_case: bool) -> CtrlTokenizer {
        let cache = RefCell::new(HashMap::new());
        let regex_pattern = Regex::new(r"\S+\n?").unwrap();
        CtrlTokenizer { vocab, bpe_ranks: merges, cache, regex_pattern, lower_case }
    }
}

impl Tokenizer<OpenAiGptVocab> for CtrlTokenizer {
    fn vocab(&self) -> &OpenAiGptVocab {
        self.vocab.as_ref()
    }

    fn tokenize(&self, text: &str) -> Vec<String> {
        let mut tokenized_text: Vec<String> = Vec::with_capacity(text.len());
        let temp_text = split_on_special_tokens(text, self.vocab.as_ref());
        let temp_text = temp_text
            .into_iter()
            .map(|v| if self.lower_case { v.to_lowercase() } else { v.to_owned() })
            .collect_vec();

        for text in temp_text {
            if !self.vocab.special_values.contains_key(text.as_str()) {
                for word in self.regex_pattern.find_iter(text.as_ref()) {
                    let cached: bool = match self.cache.borrow().get(word.as_str()) {
                        Some(value) => {
                            tokenized_text.extend(value.clone());
                            true
                        }
                        None => false
                    };
                    if !cached {
                        let bpe_output = ctrl_bpe(word.as_str(), self.bpe_ranks.as_ref());
                        self.cache.borrow_mut().insert(word.as_str().to_owned(), bpe_output.clone());
                        tokenized_text.extend(bpe_output);
                    }
                };
            } else {
                tokenized_text.push(text);
            }
        }
        tokenized_text
    }

    fn convert_tokens_to_string(&self, tokens: Vec<String>) -> String {
        tokens.join(" ").replace("@@ ", "").trim().to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::OpenAiGptVocab;
    use std::collections::HashMap;
    use crate::preprocessing::tokenizer::base_tokenizer::{TruncationStrategy, TokenizedInput};
    use crate::preprocessing::vocab::base_vocab::swap_key_values;

    fn generate_test_vocab() -> OpenAiGptVocab {
        let values: HashMap<String, i64> = [
            ("t".to_owned(), 0),
            ("h".to_owned(), 1),
            ("a@@".to_owned(), 2),
            ("n".to_owned(), 3),
            ("the".to_owned(), 4),
            ("r@@".to_owned(), 5),
            ("<unk>".to_owned(), 6),
            ("o@@".to_owned(), 8)
        ].iter().cloned().collect();

        let special_values: HashMap<String, i64> = [
            ("<unk>".to_owned(), 6),
        ].iter().cloned().collect();

        let indices = swap_key_values(&values);
        let special_indices = swap_key_values(&special_values);

        OpenAiGptVocab { values, indices, unknown_value: "<unk>", special_values, special_indices }
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
        ].iter().cloned().collect();


        BpePairVocab { values }
    }

    #[test]
    fn test_ctrl_tokenizer() {
//        Given
        let vocab = Rc::new(generate_test_vocab());
        let merges = Rc::new(generate_test_merges());
        let ctrl_tokenizer: CtrlTokenizer = CtrlTokenizer::from_existing_vocab_and_merges(vocab, merges, true);
        let test_tuples = [
            (
                "The Earth",
                vec!("the", "e@@", "a@@", "r@@", "t@@", "h")
            ),
            (
                "Hello, world!",
                vec!("h@@", "e@@", "ll@@", "o@@", ",", "w@@", "o@@", "r@@", "l@@", "d@@", "!")
            ),
            (
                "",
                vec!()
            ),
            (
                " ",
                vec!("<unk>")
            ),
            (
                " \n ",
                vec!("<unk>")
            ),
        ];
        let source_texts: Vec<&str> = test_tuples.iter().map(|v| v.0).collect();
        let expected_results: Vec<Vec<&str>> = test_tuples.iter().map(|v| v.1.clone()).collect();

//        When & Then
        for (source_text, expected_result) in test_tuples.iter() {
            assert_eq!(ctrl_tokenizer.tokenize(*source_text), *expected_result);
        }

        assert_eq!(ctrl_tokenizer.tokenize_list(source_texts.clone()), expected_results);
    }

    #[test]
    fn test_ctrl_tokenizer_no_lower_casing() {
//        Given
        let vocab = Rc::new(generate_test_vocab());
        let merges = Rc::new(generate_test_merges());
        let ctrl_tokenizer: CtrlTokenizer = CtrlTokenizer::from_existing_vocab_and_merges(vocab, merges, false);
        let test_tuples = [
            (
                "the Earth",
                vec!("the", "E@@", "a@@", "r@@", "t@@", "h")
            ),
            (
                "Hello, world!",
                vec!("H@@", "e@@", "ll@@", "o@@", ",", "w@@", "o@@", "r@@", "l@@", "d@@", "!")
            ),
            (
                "",
                vec!()
            ),
            (
                " ",
                vec!("<unk>")
            ),
            (
                " \n ",
                vec!("<unk>")
            ),
        ];
        let source_texts: Vec<&str> = test_tuples.iter().map(|v| v.0).collect();
        let expected_results: Vec<Vec<&str>> = test_tuples.iter().map(|v| v.1.clone()).collect();

//        When & Then
        for (source_text, expected_result) in test_tuples.iter() {
            assert_eq!(ctrl_tokenizer.tokenize(*source_text), *expected_result);
        }

        assert_eq!(ctrl_tokenizer.tokenize_list(source_texts.clone()), expected_results);
    }

    #[test]
    fn test_encode() {
//        Given
        let vocab = Rc::new(generate_test_vocab());
        let merges = Rc::new(generate_test_merges());
        let ctrl_tokenizer: CtrlTokenizer = CtrlTokenizer::from_existing_vocab_and_merges(vocab, merges, false);
        let truncation_strategy = TruncationStrategy::LongestFirst;
        let test_tuples = [
            (
                "the earth",
                TokenizedInput { token_ids: vec!(4, 6, 2, 5, 6, 1), segment_ids: vec!(0, 0, 0, 0, 0, 0), special_tokens_mask: vec!(0, 0, 0, 0, 0, 0), overflowing_tokens: vec!(), num_truncated_tokens: 0 }
            ),
            (
                "Hello, world!",
                TokenizedInput { token_ids: vec!(6, 6, 6, 8, 6, 6, 8, 5, 6, 6, 6), segment_ids: vec!(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), special_tokens_mask: vec!(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), overflowing_tokens: vec!(), num_truncated_tokens: 0 }
            ),
            (
                "",
                TokenizedInput { token_ids: vec!(), segment_ids: vec!(), special_tokens_mask: vec!(), overflowing_tokens: vec!(), num_truncated_tokens: 0 }
            )
        ];
        let source_texts: Vec<&str> = test_tuples.iter().map(|v| v.0).collect();
        let expected_results: Vec<TokenizedInput> = test_tuples.iter().map(|v| v.1.clone()).collect();

//        When & Then
        for (source_text, expected_result) in test_tuples.iter() {
            assert_eq!(ctrl_tokenizer.encode(source_text, None, 128, &truncation_strategy, 0),
                       *expected_result);
        }
        assert_eq!(ctrl_tokenizer.encode_list(source_texts.clone(), 128, &truncation_strategy, 0), expected_results);
    }

    #[test]
    fn test_decode() {
//        Given
        let vocab = Rc::new(generate_test_vocab());
        let merges = Rc::new(generate_test_merges());
        let ctrl_tokenizer: CtrlTokenizer = CtrlTokenizer::from_existing_vocab_and_merges(vocab, merges, false);
        let skip_special_tokens = false;
        let clean_up_tokenization_spaces = false;
        let test_tuples = [
            (
                vec!(4, 6, 2, 5, 6, 1),
                "the <unk> ar<unk> h",
            )
        ];
        let source_ids: Vec<Vec<i64>> = test_tuples.iter().map(|v| v.0.clone()).collect_vec();
        let expected_results: Vec<&str> = test_tuples.iter().map(|v| v.1.clone()).collect_vec();

//        When & Then
        for (source_ids, expected_result) in test_tuples.iter() {
            assert_eq!(ctrl_tokenizer.decode(source_ids.clone(), skip_special_tokens, clean_up_tokenization_spaces),
                       *expected_result);
        }
        assert_eq!(Tokenizer::decode_list(&ctrl_tokenizer, source_ids.clone(), skip_special_tokens, clean_up_tokenization_spaces), expected_results);
    }

    #[test]
    fn test_decode_skip_special_tokens() {
//        Given
        let vocab = Rc::new(generate_test_vocab());
        let merges = Rc::new(generate_test_merges());
        let ctrl_tokenizer: CtrlTokenizer = CtrlTokenizer::from_existing_vocab_and_merges(vocab, merges, false);
        let skip_special_tokens = true;
        let clean_up_tokenization_spaces = true;
        let test_tuples = [
            (
                vec!(4, 6, 2, 5, 6, 1),
                "the arh",
            )
        ];
        let source_ids: Vec<Vec<i64>> = test_tuples.iter().map(|v| v.0.clone()).collect_vec();
        let expected_results: Vec<&str> = test_tuples.iter().map(|v| v.1.clone()).collect_vec();

//        When & Then
        for (source_ids, expected_result) in test_tuples.iter() {
            assert_eq!(ctrl_tokenizer.decode(source_ids.clone(), skip_special_tokens, clean_up_tokenization_spaces),
                       *expected_result);
        }
        assert_eq!(Tokenizer::decode_list(&ctrl_tokenizer, source_ids.clone(), skip_special_tokens, clean_up_tokenization_spaces), expected_results);
    }
}