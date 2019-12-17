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