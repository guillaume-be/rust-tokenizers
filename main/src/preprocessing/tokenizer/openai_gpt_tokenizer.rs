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

use crate::OpenAiGptVocab;
use crate::preprocessing::vocab::base_vocab::Vocab;
use crate::preprocessing::tokenizer::base_tokenizer::{Tokenizer, BaseTokenizer, Mask, Token, TokenRef};
use std::collections::HashMap;
use crate::preprocessing::tokenizer::tokenization_utils::{split_on_bpe_pairs, openai_gpt_bpe, fix_mask};
use std::rc::Rc;
use std::cell::RefCell;
use crate::preprocessing::vocab::bpe_vocab::BpePairVocab;
use std::sync::Arc;

pub struct OpenAiGptTokenizer {
    vocab: Arc<OpenAiGptVocab>,
    base_tokenizer: BaseTokenizer<OpenAiGptVocab>,
    bpe_ranks: Rc<BpePairVocab>,
    cache: RefCell<HashMap<String, Vec<Token>>>,
}

impl OpenAiGptTokenizer {
    pub fn from_file(vocab_path: &str, merges_path: &str, lower_case: bool) -> OpenAiGptTokenizer {
        let vocab = Arc::new(OpenAiGptVocab::from_file(vocab_path));
        let base_tokenizer = BaseTokenizer::from_existing_vocab(vocab.clone(), lower_case, true);
        let bpe_ranks = Rc::new(BpePairVocab::from_file(merges_path));
        let cache = RefCell::new(HashMap::new());
        OpenAiGptTokenizer { vocab, base_tokenizer, bpe_ranks, cache }
    }

    pub fn from_existing_vocab_and_merges(vocab: Arc<OpenAiGptVocab>, merges: Rc<BpePairVocab>, lower_case: bool) -> OpenAiGptTokenizer {
        let base_tokenizer = BaseTokenizer::from_existing_vocab(vocab.clone(), lower_case, true);
        let cache = RefCell::new(HashMap::new());
        OpenAiGptTokenizer { vocab, base_tokenizer, bpe_ranks: merges, cache }
    }
}

impl Tokenizer<OpenAiGptVocab> for OpenAiGptTokenizer {
    fn vocab(&self) -> &OpenAiGptVocab {
        self.vocab.as_ref()
    }

    fn tokenize_to_tokens(&self, initial_token: TokenRef) -> Vec<Token> {
        let tokens: Vec<Token> = self.base_tokenizer.tokenize_to_tokens(initial_token).into_iter().map(|token| {
            if token.mask != Mask::Special && token.mask != Mask::Unknown {
                split_on_bpe_pairs(token.token_ref(), openai_gpt_bpe, (&self.bpe_ranks).as_ref(), &self.cache, true)
            } else {
                vec!(token)
            }
        }).flatten().collect();

        fix_mask(tokens)
    }

    fn convert_tokens_to_string(&self, tokens: Vec<String>) -> String {
        tokens.join("").replace("</w>", " ").trim().to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::OpenAiGptVocab;
    use std::collections::HashMap;
    use crate::preprocessing::tokenizer::base_tokenizer::{TruncationStrategy, TokenizedInput, Offset};
    use crate::preprocessing::vocab::base_vocab::swap_key_values;
    use itertools::Itertools;

    fn generate_test_vocab() -> OpenAiGptVocab {
        let values: HashMap<String, i64> = [
            ("t".to_owned(), 0),
            ("h".to_owned(), 1),
            ("a</w>".to_owned(), 2),
            ("n".to_owned(), 3),
            ("the".to_owned(), 4),
            ("Ġ".to_owned(), 5),
            ("<unk>".to_owned(), 6),
            ("o</w>".to_owned(), 7),
            ("the</w>".to_owned(), 8),
            ("rth</w>".to_owned(), 9),
            ("ea".to_owned(), 10),
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
            (("4".to_owned(), "t".to_owned()), 0),
            (("2".to_owned(), "n".to_owned()), 1),
            (("r".to_owned(), "th</w>".to_owned()), 2),
            (("t".to_owned(), "he</w>".to_owned()), 3),
            (("h".to_owned(), "e".to_owned()), 4),
            (("t".to_owned(), "h</w>".to_owned()), 5),
            (("t".to_owned(), "h".to_owned()), 6),
            (("th".to_owned(), "e</w>".to_owned()), 7),
            (("e".to_owned(), "a".to_owned()), 8),
        ].iter().cloned().collect();


        BpePairVocab { values }
    }

    #[test]
    fn test_openai_gpt_tokenizer() {
//        Given
        let vocab = Arc::new(generate_test_vocab());
        let merges = Rc::new(generate_test_merges());
        let openai_gpt_tokenizer: OpenAiGptTokenizer = OpenAiGptTokenizer::from_existing_vocab_and_merges(vocab, merges, true);
        let test_tuples = [
            (
                "The earth",
                vec!("the</w>", "ea", "rth</w>")
            ),
            (
                "",
                vec!()
            ),
            (
                " ",
                vec!()
            ),
            (
                " \n ",
                vec!()
            ),
        ];
        let source_texts: Vec<&str> = test_tuples.iter().map(|v| v.0).collect();
        let expected_results: Vec<Vec<&str>> = test_tuples.iter().map(|v| v.1.clone()).collect();

//        When & Then
        for (source_text, expected_result) in test_tuples.iter() {
            assert_eq!(openai_gpt_tokenizer.tokenize(*source_text), *expected_result);
        }

        assert_eq!(openai_gpt_tokenizer.tokenize_list(source_texts.clone()), expected_results);
    }

    #[test]
    fn test_openai_gpt_tokenizer_no_lower_casing() {
//        Given
        let vocab = Arc::new(generate_test_vocab());
        let merges = Rc::new(generate_test_merges());
        let openai_gpt_tokenizer: OpenAiGptTokenizer = OpenAiGptTokenizer::from_existing_vocab_and_merges(vocab, merges, false);
        let test_tuples = [
            (
                "The Earth",
                vec!("T", "h", "e</w>", "E", "a", "rth</w>")
            ),
            (
                "",
                vec!()
            ),
            (
                " ",
                vec!()
            ),
            (
                " \n ",
                vec!()
            ),
        ];
        let source_texts: Vec<&str> = test_tuples.iter().map(|v| v.0).collect();
        let expected_results: Vec<Vec<&str>> = test_tuples.iter().map(|v| v.1.clone()).collect();

//        When & Then
        for (source_text, expected_result) in test_tuples.iter() {
            assert_eq!(openai_gpt_tokenizer.tokenize(*source_text), *expected_result);
        }

        assert_eq!(openai_gpt_tokenizer.tokenize_list(source_texts.clone()), expected_results);
    }


    #[test]
    fn test_encode() {
//        Given
        let vocab = Arc::new(generate_test_vocab());
        let merges = Rc::new(generate_test_merges());
        let openai_gpt_tokenizer: OpenAiGptTokenizer = OpenAiGptTokenizer::from_existing_vocab_and_merges(vocab, merges, true);
        let truncation_strategy = TruncationStrategy::LongestFirst;
        let test_tuples = [
            (
                "the earth",
                TokenizedInput {
                    token_ids: vec!(8, 10, 9),
                    segment_ids: vec!(0, 0, 0),
                    special_tokens_mask: vec!(0, 0, 0),
                    overflowing_tokens: vec!(),
                    num_truncated_tokens: 0,
                    token_offsets: vec!(Some(Offset { begin: 0, end: 3 }), Some(Offset { begin: 4, end: 6 }), Some(Offset { begin: 6, end: 9 })),
                    mask: vec!(Mask::None, Mask::Begin, Mask::Continuation),
                }
            ),
            (
                " ",
                TokenizedInput {
                    token_ids: vec!(),
                    segment_ids: vec!(),
                    special_tokens_mask: vec!(),
                    overflowing_tokens: vec!(),
                    num_truncated_tokens: 0,
                    token_offsets: vec!(),
                    mask: vec!(),
                }
            ),
            (
                "",
                TokenizedInput {
                    token_ids: vec!(),
                    segment_ids: vec!(),
                    special_tokens_mask: vec!(),
                    overflowing_tokens: vec!(),
                    num_truncated_tokens: 0,
                    token_offsets: vec!(),
                    mask: vec!(),
                }
            )
        ];
        let source_texts: Vec<&str> = test_tuples.iter().map(|v| v.0).collect();
        let expected_results: Vec<TokenizedInput> = test_tuples.iter().map(|v| v.1.clone()).collect();

//        When & Then
        for (source_text, expected_result) in test_tuples.iter() {
            assert_eq!(openai_gpt_tokenizer.encode(source_text, None, 128, &truncation_strategy, 0),
                       *expected_result);
        }
        assert_eq!(openai_gpt_tokenizer.encode_list(source_texts.clone(), 128, &truncation_strategy, 0), expected_results);
    }

    #[test]
    fn test_decode() {
//        Given
        let vocab = Arc::new(generate_test_vocab());
        let merges = Rc::new(generate_test_merges());
        let openai_gpt_tokenizer: OpenAiGptTokenizer = OpenAiGptTokenizer::from_existing_vocab_and_merges(vocab, merges, true);
        let skip_special_tokens = false;
        let clean_up_tokenization_spaces = false;
        let test_tuples = [
            (
                vec!(8, 10, 9),
                "the earth",
            )
        ];
        let source_ids: Vec<Vec<i64>> = test_tuples.iter().map(|v| v.0.clone()).collect_vec();
        let expected_results: Vec<&str> = test_tuples.iter().map(|v| v.1.clone()).collect_vec();

//        When & Then
        for (source_ids, expected_result) in test_tuples.iter() {
            assert_eq!(openai_gpt_tokenizer.decode(source_ids.clone(), skip_special_tokens, clean_up_tokenization_spaces),
                       *expected_result);
        }
        assert_eq!(Tokenizer::decode_list(&openai_gpt_tokenizer, source_ids.clone(), skip_special_tokens, clean_up_tokenization_spaces), expected_results);
    }
}
