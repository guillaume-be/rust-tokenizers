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

use crate::Gpt2Vocab;
use crate::preprocessing::vocab::base_vocab::Vocab;
use crate::preprocessing::tokenizer::base_tokenizer::{Tokenizer, Token, TokenRef, Mask,Offset};
use std::collections::HashMap;
use crate::preprocessing::tokenizer::tokenization_utils::{bpe, split_on_special_tokens, split_on_regex_with_lookahead, split_on_bpe_pairs, fix_mask};
use std::rc::Rc;
use std::cell::RefCell;
use crate::preprocessing::vocab::bpe_vocab::BpePairVocab;
use regex::Regex;
use crate::preprocessing::tokenizer::constants::{UNICODE_TO_BYTES};
use std::iter::Iterator;
use itertools::Itertools;

pub struct Gpt2Tokenizer {
    vocab: Rc<Gpt2Vocab>,
    bpe_ranks: Rc<BpePairVocab>,
    cache: RefCell<HashMap<String, Vec<Token>>>,
    pattern_lookahead: Regex,
    pattern_tokenization: Regex,
    lower_case: bool,
}

impl Gpt2Tokenizer {
    pub fn from_file(vocab_path: &str, merges_path: &str, lower_case: bool) -> Gpt2Tokenizer {
        let vocab = Rc::new(Gpt2Vocab::from_file(vocab_path));
        let bpe_ranks = Rc::new(BpePairVocab::from_file(merges_path));
        let cache = RefCell::new(HashMap::new());
        let pattern_lookahead = Regex::new(r"\s+\S").unwrap();
        let pattern_tokenization = Regex::new(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+").unwrap();
        Gpt2Tokenizer { vocab, bpe_ranks, cache, pattern_lookahead, pattern_tokenization, lower_case }
    }

    pub fn from_existing_vocab_and_merges(vocab: Rc<Gpt2Vocab>, merges: Rc<BpePairVocab>, lower_case: bool) -> Gpt2Tokenizer {
        let cache = RefCell::new(HashMap::new());
        let pattern_lookahead = Regex::new(r"\s+\S").unwrap();
        let pattern_tokenization = Regex::new(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+").unwrap();
        Gpt2Tokenizer { vocab, bpe_ranks: merges, cache, pattern_lookahead, pattern_tokenization, lower_case }
    }
}

impl Tokenizer<Gpt2Vocab> for Gpt2Tokenizer {
    fn vocab(&self) -> &Gpt2Vocab {
        self.vocab.as_ref()
    }

    fn tokenize_to_tokens<'a>(&self, initial_token: TokenRef<'a>) -> Vec<Token> {
        let tokens: Vec<Token> = split_on_special_tokens(initial_token, self.vocab.as_ref())
            .into_iter()
            .map(|token| {
                // v-- this is where the token gets owned, all steps above handle TokenRefs (dealing with &str)
                let mut token = token.owned_token();
                if token.mask != Mask::Special && token.mask != Mask::Unknown {
                    //apply the necessary transformations to the actual tokens (unless it's a special value)
                    if self.lower_case {
                        token.text = token.text.to_lowercase();
                    }
                }

                split_on_regex_with_lookahead(token.token_ref(), &self.pattern_lookahead, &self.pattern_tokenization).into_iter().map(|token| token.owned_token()).collect::<Vec<Token>>()
            })
            .flatten()
            .map(|token: Token| {
                split_on_bpe_pairs(token.token_ref(), bpe, &self.bpe_ranks, &self.cache)
            })
            .flatten()
            .collect();

        fix_mask(tokens)
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
        String::from_utf8_lossy(&tokens).to_string()
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
            ("o@@".to_owned(), 7),
            ("Ġear".to_owned(), 8),
            ("th".to_owned(), 9),
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
            (("Ġ".to_owned(), "e".to_owned()), 7),
            (("Ġe".to_owned(), "a".to_owned()), 8),
            (("Ġea".to_owned(), "r".to_owned()), 9),
        ].iter().cloned().collect();


        BpePairVocab { values }
    }

    #[test]
    fn test_gpt2_tokenizer() {
//        Given
        let vocab = Rc::new(generate_test_vocab());
        let merges = Rc::new(generate_test_merges());
        let gpt2_tokenizer: Gpt2Tokenizer = Gpt2Tokenizer::from_existing_vocab_and_merges(vocab, merges, true);
        let test_tuples = [
            (
                "the Earth",
                vec!("the", "Ġear", "th")
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
    fn test_gpt2_tokenizer_no_lower_casing() {
//        Given
        let vocab = Rc::new(generate_test_vocab());
        let merges = Rc::new(generate_test_merges());
        let gpt2_tokenizer: Gpt2Tokenizer = Gpt2Tokenizer::from_existing_vocab_and_merges(vocab, merges, false);
        let test_tuples = [
            (
                "the Earth",
                vec!("the", "Ġ", "E", "a", "r", "th")
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
        let gpt2_tokenizer: Gpt2Tokenizer = Gpt2Tokenizer::from_existing_vocab_and_merges(vocab, merges, true);
        let truncation_strategy = TruncationStrategy::LongestFirst;
        let test_tuples = [
            (
                "the earth",
                TokenizedInput { token_ids: vec!(4, 8, 9), segment_ids: vec!(0, 0, 0), special_tokens_mask: vec!(0, 0, 0), overflowing_tokens: vec!(), num_truncated_tokens: 0, token_offsets: vec!(
                    Some(Offset { begin: 0, end: 3 }), Some(Offset { begin: 3, end: 7 }), Some(Offset { begin: 7, end: 9 })
                    ),
                    mask: vec!(Mask::None, Mask::Begin, Mask::Continuation)
                }
            ),
            (
                " ",
                TokenizedInput { token_ids: vec!(6), segment_ids: vec!(0), special_tokens_mask: vec!(0), overflowing_tokens: vec!(), num_truncated_tokens: 0, token_offsets: vec!( Some(Offset { begin: 0, end: 1 }) ), mask: vec!(Mask::None) }
            ),
            (
                "",
                TokenizedInput { token_ids: vec!(), segment_ids: vec!(), special_tokens_mask: vec!(), overflowing_tokens: vec!(), num_truncated_tokens: 0, token_offsets: vec!(), mask: vec!(Mask::None) }
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

    #[test]
    fn test_decode() {
//        Given
        let vocab = Rc::new(generate_test_vocab());
        let merges = Rc::new(generate_test_merges());
        let gpt2_tokenizer: Gpt2Tokenizer = Gpt2Tokenizer::from_existing_vocab_and_merges(vocab, merges, true);
        let skip_special_tokens = false;
        let clean_up_tokenization_spaces = false;
        let test_tuples = [
            (
                vec!(4, 8, 9),
                "the earth",
            )
        ];
        let source_ids: Vec<Vec<i64>> = test_tuples.iter().map(|v| v.0.clone()).collect_vec();
        let expected_results: Vec<&str> = test_tuples.iter().map(|v| v.1.clone()).collect_vec();

//        When & Then
        for (source_ids, expected_result) in test_tuples.iter() {
            assert_eq!(gpt2_tokenizer.decode(source_ids.clone(), skip_special_tokens, clean_up_tokenization_spaces),
                       *expected_result);
        }
        assert_eq!(Tokenizer::decode_list(&gpt2_tokenizer, source_ids.clone(), skip_special_tokens, clean_up_tokenization_spaces), expected_results);
    }
}
