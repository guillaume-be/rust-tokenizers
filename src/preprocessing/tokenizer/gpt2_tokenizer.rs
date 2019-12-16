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
use crate::preprocessing::tokenizer::tokenization_utils::bpe;
use std::rc::Rc;
use std::cell::RefCell;
use crate::preprocessing::vocab::bpe_vocab::BpePairVocab;
use regex::Regex;
use crate::preprocessing::tokenizer::constants::BYTES_TO_UNICODE;

pub struct Gpt2Tokenizer {
    vocab: Rc<Gpt2Vocab>,
    bpe_ranks: Rc<BpePairVocab>,
    cache: RefCell<HashMap<String, Vec<String>>>,
    pattern: Regex
}

impl Gpt2Tokenizer {
    pub fn from_file(vocab_path: &str, merges_path: &str) -> Gpt2Tokenizer {
        let vocab = Rc::new(Gpt2Vocab::from_file(vocab_path));
        let bpe_ranks = Rc::new(BpePairVocab::from_file(merges_path));
        let cache = RefCell::new(HashMap::new());
        let pattern = Regex::new(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+").unwrap();
        Gpt2Tokenizer { vocab, bpe_ranks, cache, pattern }
    }

    pub fn from_existing_vocab_and_merges(vocab: Rc<Gpt2Vocab>, merges: Rc<BpePairVocab>) -> Gpt2Tokenizer {
        let cache = RefCell::new(HashMap::new());
//        ToDo: handle lookahead and avoid grouping all matches that contain a string as the next group
        let pattern = Regex::new(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+").unwrap();
        Gpt2Tokenizer { vocab, bpe_ranks: merges, cache, pattern }
    }
}

impl Tokenizer<Gpt2Vocab> for Gpt2Tokenizer {
    fn vocab(&self) -> &Gpt2Vocab {
        &self.vocab
    }

    fn tokenize(&self, text: &str) -> Vec<String> {
        let mut tokenized_text: Vec<String> = Vec::with_capacity(text.len());
        for word in self.pattern.find_iter(text.as_ref()) {
            let word = word.as_str();
            let word: String = word.chars().map(|v| BYTES_TO_UNICODE.get(&(v as u32)).unwrap()).into_iter().collect();
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
        tokenized_text
    }
}

//#[cfg(test)]
//mod tests {
//    use super::*;
//    use crate::CtrlVocab;
//    use std::collections::HashMap;
//    use crate::preprocessing::tokenizer::base_tokenizer::{TruncationStrategy, TokenizedInput};
//
//    fn generate_test_vocab() -> CtrlVocab {
//        let values: HashMap<String, i64> = [
//            ("t".to_owned(), 0),
//            ("h".to_owned(), 1),
//            ("a@@".to_owned(), 2),
//            ("n".to_owned(), 3),
//            ("the".to_owned(), 4),
//            ("r@@".to_owned(), 5),
//            ("<unk>".to_owned(), 6),
//            ("o@@".to_owned(), 8)
//        ].iter().cloned().collect();
//
//        let special_values: HashMap<String, i64> = [
//            ("<unk>".to_owned(), 7),
//        ].iter().cloned().collect();
//
//        CtrlVocab { values, unknown_value: "<unk>", special_values }
//    }
//
//    fn generate_test_merges() -> BpePairVocab {
//        let values: HashMap<(String, String), i64> = [
//            (("t".to_owned(), "h".to_owned()), 0),
//            (("a".to_owned(), "n".to_owned()), 1),
//            (("i".to_owned(), "n".to_owned()), 2),
//            (("th".to_owned(), "e</w>".to_owned()), 3),
//            (("e".to_owned(), "r".to_owned()), 4),
//            (("r".to_owned(), "e".to_owned()), 5),
//            (("l".to_owned(), "l".to_owned()), 6),
//        ].iter().cloned().collect();
//
//
//        BpePairVocab { values }
//    }
//
//    #[test]
//    fn test_ctrl_tokenizer() {
////        Given
//        let vocab = Rc::new(generate_test_vocab());
//        let merges = Rc::new(generate_test_merges());
//        let ctrl_tokenizer: CtrlTokenizer = CtrlTokenizer::from_existing_vocab_and_merges(vocab, merges);
//        let test_tuples = [
//            (
//                "the earth",
//                vec!("the", "e@@", "a@@", "r@@", "t@@", "h")
//            ),
//            (
//                "Hello, world!",
//                vec!("H@@", "e@@", "ll@@", "o@@", ",", "w@@", "o@@", "r@@", "l@@", "d@@", "!")
//            ),
//            (
//                "",
//                vec!("")
//            ),
//            (
//                " ",
//                vec!("")
//            ),
//            (
//                " \n ",
//                vec!("")
//            ),
//        ];
//        let source_texts: Vec<&str> = test_tuples.iter().map(|v| v.0).collect();
//        let expected_results: Vec<Vec<&str>> = test_tuples.iter().map(|v| v.1.clone()).collect();
//
////        When & Then
//        for (source_text, expected_result) in test_tuples.iter() {
//            assert_eq!(ctrl_tokenizer.tokenize(*source_text), *expected_result);
//        }
//
//        assert_eq!(ctrl_tokenizer.tokenize_list(source_texts.clone()), expected_results);
//    }
//
//    #[test]
//    fn test_encode() {
////        Given
//        let vocab = Rc::new(generate_test_vocab());
//        let merges = Rc::new(generate_test_merges());
//        let ctrl_tokenizer: CtrlTokenizer = CtrlTokenizer::from_existing_vocab_and_merges(vocab, merges);
//        let truncation_strategy = TruncationStrategy::LongestFirst;
//        let test_tuples = [
//            (
//                "the earth",
//                TokenizedInput { token_ids: vec!(4, 6, 2, 5, 6, 1), segment_ids: vec!(0, 0, 0, 0, 0, 0), special_tokens_mask: vec!(0, 0, 0, 0, 0, 0), overflowing_tokens: vec!(), num_truncated_tokens: 0 }
//            ),
//            (
//                "Hello, world!",
//                TokenizedInput { token_ids: vec!(6, 6, 6, 8, 6, 6, 8, 5, 6, 6, 6), segment_ids: vec!(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), special_tokens_mask: vec!(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), overflowing_tokens: vec!(), num_truncated_tokens: 0 }
//            ),
//            (
//                "",
//                TokenizedInput { token_ids: vec!(6), segment_ids: vec!(0), special_tokens_mask: vec!(0), overflowing_tokens: vec!(), num_truncated_tokens: 0 }
//            )
//        ];
//        let source_texts: Vec<&str> = test_tuples.iter().map(|v| v.0).collect();
//        let expected_results: Vec<TokenizedInput> = test_tuples.iter().map(|v| v.1.clone()).collect();
//
////        When & Then
//        for (source_text, expected_result) in test_tuples.iter() {
//            assert_eq!(ctrl_tokenizer.encode(source_text, None, 128, &truncation_strategy, 0),
//                       *expected_result);
//        }
//        assert_eq!(ctrl_tokenizer.encode_list(source_texts.clone(), 128, &truncation_strategy, 0), expected_results);
//    }
//}