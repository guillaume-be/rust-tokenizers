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

use std::sync::Arc;
use crate::CtrlVocab;
use crate::preprocessing::vocab::base_vocab::Vocab;
use crate::preprocessing::tokenizer::base_tokenizer::Tokenizer;
use crate::preprocessing::vocab::ctrl_vocab::{BpePairVocab, BpePair};
use regex::Regex;
use std::collections::HashSet;
use std::ops::Index;


pub struct CtrlTokenizer {
    vocab: Arc<CtrlVocab>,
    bpe_ranks: Arc<BpePairVocab>,
}

impl CtrlTokenizer {
    pub fn from_file(vocab_path: &str, merges_path: &str) -> CtrlTokenizer {
        let vocab = Arc::new(CtrlVocab::from_file(vocab_path));
        let bpe_ranks = Arc::new(BpePairVocab::from_file(merges_path));
        CtrlTokenizer { vocab, bpe_ranks }
    }

    pub fn from_existing_vocab_and_merges(vocab: Arc<CtrlVocab>, merges: Arc<BpePairVocab>) -> CtrlTokenizer {
        CtrlTokenizer { vocab, bpe_ranks: merges }
    }
}

impl Tokenizer<CtrlVocab> for CtrlTokenizer {
    fn vocab(&self) -> &CtrlVocab {
        &self.vocab
    }

    fn tokenize(&self, text: &str) -> Vec<String> {
        let mut tokenized_text: Vec<String> = vec!();
        for word in Regex::new(r"\S+\n?").unwrap().find_iter(text.as_ref()) {
            tokenized_text.push(word.as_str().to_owned());
        };
        tokenized_text
    }
}

pub fn get_pairs(token: &Vec<String>) -> Option<HashSet<BpePair>> {
    let mut token = token.iter();
    if let Some(mut byte_1) = token.next() {
        if let Some(mut byte_2) = token.next() {
            let mut output: HashSet<BpePair> = HashSet::new();
            output.insert(BpePair { byte_1: String::from(byte_1), byte_2: String::from(byte_2) });
            while let Some(byte) = token.next() {
                byte_1 = byte_2;
                byte_2 = byte;
                output.insert(BpePair { byte_1: String::from(byte_1), byte_2: String::from(byte_2) });
            }
            Some(output)
        } else {
            None
        }
    } else {
        None
    }
}

pub fn bpe(token: &str, bpe_ranks: &BpePairVocab) -> String {
    let mut word = token.chars().map(|v| v.to_string()).collect::<Vec<String>>();

    if !word.is_empty() {
        word.last_mut().unwrap().push_str("</w>");
    };

    if let Some(initial_pairs) = get_pairs(&word) {
        let mut pairs = initial_pairs;
        loop {
            let bigram = pairs.iter().min_by_key(|pair|
                match bpe_ranks.byte_pair_to_id(pair) {
                    Some(rank) => *rank,
                    None => i64::max_value()
                }).unwrap();
            if bpe_ranks.byte_pair_to_id(bigram).is_none() { break; }
            let mut new_word: Vec<String> = vec!();
            let mut i = 0;

            while i < word.len() {
                let j = if let Some(index) = &word[i..].iter().position(|r| *r == bigram.byte_1) {
                    index + i
                } else {
                    new_word.extend_from_slice(&word[i..]);
                    break;
                };
                new_word.extend_from_slice(&word[i..j]);
                i = j;
                if (word[i] == bigram.byte_1) & (i < word.len() - 1) & (word[i + 1] == bigram.byte_2) {
                    let mut combined_bytes = bigram.byte_1.clone();
                    combined_bytes.push_str(bigram.byte_2.as_str());
                    new_word.push(combined_bytes);
                    i += 2;
                } else {
                    new_word.push(bigram.byte_1.clone());
                    i += 1;
                }
            }
            word = new_word.clone();
            if word.len() == 1 {
                break;
            }
            pairs = get_pairs(&word).unwrap();
        }
    };
    let word = word.join("@@ ");
    let word = (&word[..word.len() - 4]).to_owned();
    word
}