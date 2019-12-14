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
use std::collections::HashSet;
use regex::Regex;


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
            tokenized_text.extend(bpe(word.as_str(), &self.bpe_ranks));
        };
        tokenized_text
    }
}

pub fn get_pairs(token: &Vec<String>) -> Option<HashSet<BpePair>> {
    match token.len() {
        0 | 1 => None,
        _ => {
            let mut output: HashSet<BpePair> = HashSet::with_capacity(token.len());
            for idx in 0..token.len() - 1 {
                if let [byte_1, byte_2] = &token[idx..idx + 2] {
                    output.insert(BpePair { byte_1: byte_1.to_owned(), byte_2: byte_2.to_owned() });
                }
            }
            Some(output)
        }
    }
}

pub fn bpe(token: &str, bpe_ranks: &BpePairVocab) -> Vec<String> {
    let mut sub_tokens = token.chars().map(|v| v.to_string()).collect::<Vec<String>>();

    if !sub_tokens.is_empty() {
        sub_tokens.last_mut().unwrap().push_str("</w>");
    };

    if let Some(initial_pairs) = get_pairs(&sub_tokens) {
        let mut pairs = initial_pairs;
        loop {
            let bigram = pairs.iter().min_by_key(|pair|
                match bpe_ranks.byte_pair_to_id(pair) {
                    Some(rank) => *rank,
                    None => i64::max_value()
                }).unwrap();
            if bpe_ranks.byte_pair_to_id(bigram).is_none() { break; }
            let mut temp_sub_tokens: Vec<String> = vec!();
            let mut i = 0;

            while i < sub_tokens.len() {
                let j = if let Some(index) = &sub_tokens[i..].iter().position(|r| *r == bigram.byte_1) {
                    index + i
                } else {
                    temp_sub_tokens.extend_from_slice(&sub_tokens[i..]);
                    break;
                };
                temp_sub_tokens.extend_from_slice(&sub_tokens[i..j]);
                i = j;
                if (sub_tokens[i] == bigram.byte_1) & (i < sub_tokens.len() - 1) & (sub_tokens[i + 1] == bigram.byte_2) {
                    let mut combined_bytes = bigram.byte_1.clone();
                    combined_bytes.push_str(bigram.byte_2.as_str());
                    temp_sub_tokens.push(combined_bytes);
                    i += 2;
                } else {
                    temp_sub_tokens.push(bigram.byte_1.clone());
                    i += 1;
                }
            }
            sub_tokens = temp_sub_tokens.clone();
            if sub_tokens.len() == 1 {
                break;
            }
            pairs = get_pairs(&sub_tokens).unwrap();
        }
    };

    let word = sub_tokens.join("@@ ");
    if !word.is_empty() {
        (&word[..word.len() - 4]).split(' ').map(|v| v.to_owned()).collect()
    } else {
        vec!(word)
    }
}