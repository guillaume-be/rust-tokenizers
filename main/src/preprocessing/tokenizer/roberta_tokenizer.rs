// Copyright 2018 The Open AI Team Authors
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

use crate::RobertaVocab;
use crate::preprocessing::vocab::base_vocab::Vocab;
use crate::preprocessing::tokenizer::base_tokenizer::Tokenizer;
use std::collections::HashMap;
use crate::preprocessing::tokenizer::tokenization_utils::{bpe, split_on_special_tokens, is_whitespace};
use std::rc::Rc;
use std::cell::RefCell;
use crate::preprocessing::vocab::bpe_vocab::BpePairVocab;
use regex::Regex;
use crate::preprocessing::tokenizer::constants::{BYTES_TO_UNICODE, UNICODE_TO_BYTES};
use std::iter::Iterator;
use itertools::Itertools;

pub struct RobertaTokenizer {
    vocab: Rc<RobertaVocab>,
    bpe_ranks: Rc<BpePairVocab>,
    cache: RefCell<HashMap<String, Vec<String>>>,
    pattern_lookahead: Regex,
    pattern_tokenization: Regex,
    lower_case: bool,
}

impl RobertaTokenizer {
    pub fn from_file(vocab_path: &str, merges_path: &str, lower_case: bool) -> RobertaTokenizer {
        let vocab = Rc::new(RobertaVocab::from_file(vocab_path));
        let bpe_ranks = Rc::new(BpePairVocab::from_file(merges_path));
        let cache = RefCell::new(HashMap::new());
        let pattern_lookahead = Regex::new(r"\s+\S").unwrap();
        let pattern_tokenization = Regex::new(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+").unwrap();
        RobertaTokenizer { vocab, bpe_ranks, cache, pattern_lookahead, pattern_tokenization, lower_case }
    }

    pub fn from_existing_vocab_and_merges(vocab: Rc<RobertaVocab>, merges: Rc<BpePairVocab>, lower_case: bool) -> RobertaTokenizer {
        let cache = RefCell::new(HashMap::new());
        let pattern_lookahead = Regex::new(r"\s+\S").unwrap();
        let pattern_tokenization = Regex::new(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+").unwrap();
        RobertaTokenizer { vocab, bpe_ranks: merges, cache, pattern_lookahead, pattern_tokenization, lower_case }
    }
}

impl Tokenizer<RobertaVocab> for RobertaTokenizer {
    fn vocab(&self) -> &RobertaVocab {
        self.vocab.as_ref()
    }

    fn tokenize(&self, text: &str) -> Vec<String> {
        if text.len() == 0 {
            return vec!();
        }
        let mut tokenized_text: Vec<String> = Vec::with_capacity(text.len() + 1);
        let temp_text = if !is_whitespace(&text.chars().next().unwrap()) {
            let text = format!("{}{}", ' ', text);
            let temp_text = split_on_special_tokens(text.as_str(), self.vocab.as_ref());
            temp_text
                .into_iter()
                .map(|v| if self.lower_case { v.to_lowercase() } else { v.to_owned() })
                .collect_vec()
        } else {
            let temp_text = split_on_special_tokens(text, self.vocab.as_ref());
            temp_text
                .into_iter()
                .map(|v| if self.lower_case { v.to_lowercase() } else { v.to_owned() })
                .collect_vec()
        };

//        Rust regex's library does not include lookahead, decomposing the process in 2 steps
        for text in temp_text {
            if !self.vocab.special_values.contains_key(text.as_str()) {
                let mut sub_words: Vec<&str> = vec!();
                let mut splits: Vec<&str> = vec!();

                let mut i: usize = 0;
                let mut end: usize;
                for hit in self.pattern_lookahead.find_iter(text.as_str()) {
                    end = hit.end() - 1 - hit.as_str().chars().last().unwrap().len_utf8();
                    splits.push(&text[i..end]);
                    i = end;
                }
                splits.push(&text[i..]);

                for sub_word in splits {
                    for hit in self.pattern_tokenization.find_iter(sub_word) {
                        sub_words.push(hit.as_str());
                    }
                }

                for word in sub_words {
                    let word: String = word.as_bytes().iter().map(|v| BYTES_TO_UNICODE.get(&v).unwrap()).collect();
                    let cached: bool = match self.cache.borrow().get(&word) {
                        Some(value) => {
                            tokenized_text.extend(value.clone());
                            true
                        }
                        None => false
                    };
                    if !cached {
                        let bpe_output = bpe(word.as_str(), self.bpe_ranks.as_ref());
                        self.cache.borrow_mut().insert(word.to_owned(), bpe_output.clone());
                        tokenized_text.extend(bpe_output);
                    }
                };
            } else {
                tokenized_text.push(text);
            }
        }
        tokenized_text
    }

    fn build_input_with_special_tokens(&self, tokens_1: Vec<i64>, tokens_2: Option<Vec<i64>>) -> (Vec<i64>, Vec<i8>, Vec<i8>) {
        let mut output: Vec<i64> = vec!();
        let mut token_segment_ids: Vec<i8> = vec!();
        let mut special_tokens_mask: Vec<i8> = vec!();
        special_tokens_mask.push(1);
        special_tokens_mask.extend(vec![0; tokens_1.len()]);
        special_tokens_mask.push(1);
        token_segment_ids.extend(vec![0; tokens_1.len() + 2]);
        output.push(self.vocab.token_to_id(RobertaVocab::cls_value()));
        output.extend(tokens_1);
        output.push(self.vocab.token_to_id(RobertaVocab::sep_value()));
        if let Some(add_tokens) = tokens_2 {
            special_tokens_mask.push(1);
            special_tokens_mask.extend(vec![0; add_tokens.len()]);
            special_tokens_mask.push(1);
            token_segment_ids.push(0);
            token_segment_ids.extend(vec![1; add_tokens.len() + 1]);
            output.push(self.vocab.token_to_id(RobertaVocab::sep_value()));
            output.extend(add_tokens);
            output.push(self.vocab.token_to_id(RobertaVocab::sep_value()));
        }
        (output, token_segment_ids, special_tokens_mask)
    }

    fn convert_tokens_to_string(&self, tokens: Vec<String>) -> String {
        let tokens = tokens
            .iter()
            .join("")
            .replace(" ##", "")
            .trim()
            .chars()
            .map(|character| UNICODE_TO_BYTES.get(&character).unwrap().clone())
            .collect_vec();

        String::from_utf8(tokens).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RobertaVocab;
    use std::collections::HashMap;
    use crate::preprocessing::tokenizer::base_tokenizer::{TruncationStrategy, TokenizedInput};
    use crate::preprocessing::vocab::base_vocab::swap_key_values;

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
        ].iter().cloned().collect();

        let special_values: HashMap<String, i64> = [
            ("<unk>".to_owned(), 6),
            ("<s>".to_owned(), 8),
            ("</s>".to_owned(), 9),
            ("<pad>".to_owned(), 10),
            ("<mask>".to_owned(), 11),
        ].iter().cloned().collect();

        let indices = swap_key_values(&values);
        let special_indices = swap_key_values(&special_values);

        RobertaVocab { values, indices, unknown_value: "<unk>", special_values, special_indices }
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
        ].iter().cloned().collect();


        BpePairVocab { values }
    }

    #[test]
    fn test_roberta_tokenizer() {
//        Given
        let vocab = Rc::new(generate_test_vocab());
        let merges = Rc::new(generate_test_merges());
        let roberta_tokenizer: RobertaTokenizer = RobertaTokenizer::from_existing_vocab_and_merges(vocab, merges, true);
        let test_tuples = [
            (
                "The Earth",
                vec!("Ġthe", "Ġear", "th")
            ),
            (
                "",
                vec!()
            ),
            (
                "✿",
                vec!("Ġ", "â", "ľ", "¿")
            ),
        ];
        let source_texts: Vec<&str> = test_tuples.iter().map(|v| v.0).collect();
        let expected_results: Vec<Vec<&str>> = test_tuples.iter().map(|v| v.1.clone()).collect();

//        When & Then
        for (source_text, expected_result) in test_tuples.iter() {
            assert_eq!(roberta_tokenizer.tokenize(*source_text), *expected_result);
        }

        assert_eq!(roberta_tokenizer.tokenize_list(source_texts.clone()), expected_results);
    }

    #[test]
    fn test_roberta_tokenizer_no_lower_casing() {
//        Given
        let vocab = Rc::new(generate_test_vocab());
        let merges = Rc::new(generate_test_merges());
        let roberta_tokenizer: RobertaTokenizer = RobertaTokenizer::from_existing_vocab_and_merges(vocab, merges, false);
        let test_tuples = [
            (
                "The Earth",
                vec!("Ġ", "T", "he", "Ġ", "E", "a", "r", "th")
            ),
            (
                "",
                vec!()
            ),
            (
                "✿",
                vec!("Ġ", "â", "ľ", "¿")
            ),
        ];
        let source_texts: Vec<&str> = test_tuples.iter().map(|v| v.0).collect();
        let expected_results: Vec<Vec<&str>> = test_tuples.iter().map(|v| v.1.clone()).collect();

//        When & Then
        for (source_text, expected_result) in test_tuples.iter() {
            assert_eq!(roberta_tokenizer.tokenize(*source_text), *expected_result);
        }

        assert_eq!(roberta_tokenizer.tokenize_list(source_texts.clone()), expected_results);
    }


    #[test]
    fn test_encode() {
//        Given
        let vocab = Rc::new(generate_test_vocab());
        let merges = Rc::new(generate_test_merges());
        let roberta_tokenizer: RobertaTokenizer = RobertaTokenizer::from_existing_vocab_and_merges(vocab, merges, true);
        let truncation_strategy = TruncationStrategy::LongestFirst;
        let test_tuples = [
            (
                "the earth",
                TokenizedInput { token_ids: vec!(8, 4, 12, 13, 9), segment_ids: vec!(0, 0, 0, 0, 0), special_tokens_mask: vec!(1, 0, 0, 0, 1), overflowing_tokens: vec!(), num_truncated_tokens: 0 }
            ),
            (
                "✿",
                TokenizedInput { token_ids: vec!(8, 5, 6, 6, 6, 9), segment_ids: vec!(0, 0, 0, 0, 0, 0), special_tokens_mask: vec!(1, 0, 0, 0, 0, 1), overflowing_tokens: vec!(), num_truncated_tokens: 0 }
            ),
            (
                "",
                TokenizedInput { token_ids: vec!(8, 9), segment_ids: vec!(0, 0), special_tokens_mask: vec!(1, 1), overflowing_tokens: vec!(), num_truncated_tokens: 0 }
            )
        ];
        let source_texts: Vec<&str> = test_tuples.iter().map(|v| v.0).collect();
        let expected_results: Vec<TokenizedInput> = test_tuples.iter().map(|v| v.1.clone()).collect();

//        When & Then
        for (source_text, expected_result) in test_tuples.iter() {
            assert_eq!(roberta_tokenizer.encode(source_text, None, 128, &truncation_strategy, 0),
                       *expected_result);
        }
        assert_eq!(roberta_tokenizer.encode_list(source_texts.clone(), 128, &truncation_strategy, 0), expected_results);
    }

    #[test]
    fn test_decode() {
//        Given
        let vocab = Rc::new(generate_test_vocab());
        let merges = Rc::new(generate_test_merges());
        let roberta_tokenizer: RobertaTokenizer = RobertaTokenizer::from_existing_vocab_and_merges(vocab, merges, true);
        let skip_special_tokens = false;
        let clean_up_tokenization_spaces = false;
        let test_tuples = [
            (
                vec!(8, 4, 12, 13, 9),
                "<s> the earth</s>",
            )
        ];
        let source_ids: Vec<Vec<i64>> = test_tuples.iter().map(|v| v.0.clone()).collect_vec();
        let expected_results: Vec<&str> = test_tuples.iter().map(|v| v.1.clone()).collect_vec();

//        When & Then
        for (source_ids, expected_result) in test_tuples.iter() {
            assert_eq!(roberta_tokenizer.decode(source_ids.clone(), skip_special_tokens, clean_up_tokenization_spaces),
                       *expected_result);
        }
        assert_eq!(Tokenizer::decode_list(&roberta_tokenizer, source_ids.clone(), skip_special_tokens, clean_up_tokenization_spaces), expected_results);
    }
}