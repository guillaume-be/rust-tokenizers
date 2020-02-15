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

use crate::Gpt2Vocab;
use crate::preprocessing::vocab::base_vocab::Vocab;
use crate::preprocessing::tokenizer::base_tokenizer::Tokenizer;
use std::collections::HashMap;
use crate::preprocessing::tokenizer::tokenization_utils::{bpe, split_on_special_tokens};
use std::rc::Rc;
use std::cell::RefCell;
use crate::preprocessing::vocab::bpe_vocab::BpePairVocab;
use regex::Regex;
use crate::preprocessing::tokenizer::constants::BYTES_TO_UNICODE;

pub struct Gpt2Tokenizer {
    vocab: Rc<Gpt2Vocab>,
    bpe_ranks: Rc<BpePairVocab>,
    cache: RefCell<HashMap<String, Vec<String>>>,
    pattern_lookahead: Regex,
    pattern_tokenization: Regex,
}

impl Gpt2Tokenizer {
    pub fn from_file(vocab_path: &str, merges_path: &str) -> Gpt2Tokenizer {
        let vocab = Rc::new(Gpt2Vocab::from_file(vocab_path));
        let bpe_ranks = Rc::new(BpePairVocab::from_file(merges_path));
        let cache = RefCell::new(HashMap::new());
        let pattern_lookahead = Regex::new(r"\s+\S").unwrap();
        let pattern_tokenization = Regex::new(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+").unwrap();
        Gpt2Tokenizer { vocab, bpe_ranks, cache, pattern_lookahead, pattern_tokenization }
    }

    pub fn from_existing_vocab_and_merges(vocab: Rc<Gpt2Vocab>, merges: Rc<BpePairVocab>) -> Gpt2Tokenizer {
        let cache = RefCell::new(HashMap::new());
        let pattern_lookahead = Regex::new(r"\s+\S").unwrap();
        let pattern_tokenization = Regex::new(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+").unwrap();
        Gpt2Tokenizer { vocab, bpe_ranks: merges, cache, pattern_lookahead, pattern_tokenization }
    }
}

impl Tokenizer<Gpt2Vocab> for Gpt2Tokenizer {
    fn vocab(&self) -> &Gpt2Vocab {
        &self.vocab
    }

    fn tokenize(&self, text: &str) -> Vec<String> {
        let mut tokenized_text: Vec<String> = Vec::with_capacity(text.len());
        let temp_text = split_on_special_tokens(text, self.vocab.as_ref());

//        Rust regex's library does not include lookahead, decomposing the process in 2 steps
        for text in temp_text {
            if !self.vocab.special_values.contains_key(text) {
                let mut sub_words: Vec<&str> = vec!();
                let mut splits: Vec<&str> = vec!();

                let mut i: usize = 0;
                let mut end: usize;
                for hit in self.pattern_lookahead.find_iter(text) {
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
                        let bpe_output = bpe(&word, &self.bpe_ranks);
                        self.cache.borrow_mut().insert(word.to_owned(), bpe_output.clone());
                        tokenized_text.extend(bpe_output);
                    }
                };
            } else {
                tokenized_text.push(text.to_owned());
            }
        }
        tokenized_text
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Gpt2Vocab;
    use std::collections::HashMap;
    use crate::preprocessing::tokenizer::base_tokenizer::{TruncationStrategy, TokenizedInput};
    use crate::preprocessing::vocab::base_vocab::swap_key_values;

    fn generate_test_vocab() -> Gpt2Vocab {
        let values: HashMap<String, i64> = [
            ("t".to_owned(), 0),
            ("h".to_owned(), 1),
            ("a@@".to_owned(), 2),
            ("n".to_owned(), 3),
            ("the".to_owned(), 4),
            ("Ġ".to_owned(), 5),
            ("<|endoftext|>".to_owned(), 6),
            ("o@@".to_owned(), 7)
        ].iter().cloned().collect();

        let special_values: HashMap<String, i64> = [
            ("<|endoftext|>".to_owned(), 6),
        ].iter().cloned().collect();

        let indices = swap_key_values(&values);
        let special_indices = swap_key_values(&special_values);

        Gpt2Vocab { values, indices, unknown_value: "<|endoftext|>", special_values, special_indices }
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
        ].iter().cloned().collect();


        BpePairVocab { values }
    }

    #[test]
    fn test_ctrl_tokenizer() {
//        Given
        let vocab = Rc::new(generate_test_vocab());
        let merges = Rc::new(generate_test_merges());
        let gpt2_tokenizer: Gpt2Tokenizer = Gpt2Tokenizer::from_existing_vocab_and_merges(vocab, merges);
        let test_tuples = [
            (
                "the earth",
                vec!("the", "Ġ", "e", "a", "r", "th")
            ),
            (
                "",
                vec!()
            ),
            (
                " ",
                vec!("<|endoftext|>")
            ),
            (
                " \n ",
                vec!("<|endoftext|>")
            ),
        ];
        let source_texts: Vec<&str> = test_tuples.iter().map(|v| v.0).collect();
        let expected_results: Vec<Vec<&str>> = test_tuples.iter().map(|v| v.1.clone()).collect();

//        When & Then
        for (source_text, expected_result) in test_tuples.iter() {
            assert_eq!(gpt2_tokenizer.tokenize(*source_text), *expected_result);
        }

        assert_eq!(gpt2_tokenizer.tokenize_list(source_texts.clone()), expected_results);
    }


    #[test]
    fn test_encode() {
//        Given
        let vocab = Rc::new(generate_test_vocab());
        let merges = Rc::new(generate_test_merges());
        let gpt2_tokenizer: Gpt2Tokenizer = Gpt2Tokenizer::from_existing_vocab_and_merges(vocab, merges);
        let truncation_strategy = TruncationStrategy::LongestFirst;
        let test_tuples = [
            (
                "the earth",
                TokenizedInput { token_ids: vec!(4, 5, 6, 6, 6, 6), segment_ids: vec!(0, 0, 0, 0, 0, 0), special_tokens_mask: vec!(0, 0, 0, 0, 0, 0), overflowing_tokens: vec!(), num_truncated_tokens: 0 }
            ),
            (
                " ",
                TokenizedInput { token_ids: vec!(6), segment_ids: vec!(0), special_tokens_mask: vec!(0), overflowing_tokens: vec!(), num_truncated_tokens: 0 }
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
            assert_eq!(gpt2_tokenizer.encode(source_text, None, 128, &truncation_strategy, 0),
                       *expected_result);
        }
        assert_eq!(gpt2_tokenizer.encode_list(source_texts.clone(), 128, &truncation_strategy, 0), expected_results);
    }
}