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

use crate::CtrlVocab;
use crate::preprocessing::vocab::base_vocab::Vocab;
use crate::preprocessing::tokenizer::base_tokenizer::Tokenizer;
use crate::preprocessing::vocab::ctrl_vocab::{BpePairVocab, BpePair};
use std::collections::{HashSet, HashMap};
use crate::preprocessing::tokenizer::tokenization_utils::is_whitespace;
use std::rc::Rc;


pub struct CtrlTokenizer {
    vocab: Rc<CtrlVocab>,
    bpe_ranks: Rc<BpePairVocab>,
    cache: HashMap<String, Vec<String>>,
}

impl CtrlTokenizer {
    pub fn from_file(vocab_path: &str, merges_path: &str) -> CtrlTokenizer {
        let vocab = Rc::new(CtrlVocab::from_file(vocab_path));
        let bpe_ranks = Rc::new(BpePairVocab::from_file(merges_path));
        let cache = HashMap::new();
        CtrlTokenizer { vocab, bpe_ranks, cache }
    }

    pub fn from_existing_vocab_and_merges(vocab: Rc<CtrlVocab>, merges: Rc<BpePairVocab>) -> CtrlTokenizer {
        let cache = HashMap::new();
        CtrlTokenizer { vocab, bpe_ranks: merges, cache }
    }

    fn vocab(&self) -> &CtrlVocab {
        &self.vocab
    }

    pub fn tokenize(&mut self, text: &str) -> Vec<String> {
        let mut tokenized_text: Vec<String> = vec!();
        for word in text.trim().split(|v| is_whitespace(&v)) {
            match self.cache.get(word) {
                Some(value) => tokenized_text.extend(value.clone()),
                None => {
                    let bpe_output = bpe(word, &self.bpe_ranks);
                    self.cache.insert(word.to_owned(), bpe_output.clone());
                    tokenized_text.extend(bpe_output);
                }
            }

        };
        tokenized_text
    }
}

//impl Tokenizer<CtrlVocab> for CtrlTokenizer {
//
//}

pub fn get_pairs(token: &Vec<String>) -> Option<HashSet<BpePair>> {
    match token.len() {
        0 | 1 => None,
        _ => {
            let mut output: HashSet<BpePair> = HashSet::with_capacity(token.len());
            for idx in 0..token.len() - 1 {
                if let [byte_1, byte_2] = &token[idx..idx + 2] {
                    output.insert(BpePair { byte_1: byte_1.clone(), byte_2: byte_2.clone() });
                }
            }
            Some(output)
        }
    }
}

pub fn group_common_pairs(tokens: Vec<String>, bpe_ranks: &BpePairVocab) -> (Vec<String>, bool) {
    let mut end_loop: bool = false;
    if let Some(pairs) = get_pairs(&tokens) {
        let bigram = pairs.iter().min_by_key(|pair|
            match bpe_ranks.byte_pair_to_id(pair) {
                Some(rank) => *rank,
                None => i64::max_value()
            }).unwrap();
        if bpe_ranks.byte_pair_to_id(bigram).is_none() {
            return (tokens, true);
        }

        let mut temp_sub_tokens: Vec<String> = vec!();
        let mut i = 0;

        while i < tokens.len() {
            let j = if let Some(index) = &tokens[i..].iter().position(|r| *r == bigram.byte_1) {
                index + i
            } else {
                temp_sub_tokens.extend_from_slice(&tokens[i..]);
                break;
            };
            temp_sub_tokens.extend_from_slice(&tokens[i..j]);
            i = j;
            if (tokens[i] == bigram.byte_1) & (i < tokens.len() - 1) & (tokens[i + 1] == bigram.byte_2) {
                let mut combined_bytes = bigram.byte_1.clone();
                combined_bytes.push_str(bigram.byte_2.as_str());
                temp_sub_tokens.push(combined_bytes);
                i += 2;
            } else {
                temp_sub_tokens.push(bigram.byte_1.clone());
                i += 1;
            }
        }
        if temp_sub_tokens.len() == 1 {
            end_loop = true;
        }
        return (temp_sub_tokens, end_loop);
    } else {
        return (tokens, true);
    }
}

pub fn bpe(token: &str, bpe_ranks: &BpePairVocab) -> Vec<String> {
    let mut sub_tokens = token.chars().map(|v| v.to_string()).collect::<Vec<String>>();

    if !sub_tokens.is_empty() {
        sub_tokens.last_mut().unwrap().push_str("</w>");
    };

    let (mut output, mut end_loop) = (sub_tokens.clone(), false);
    loop {
        output = match group_common_pairs(output, &bpe_ranks) {
            (value, true) => {
                end_loop = true;
                value
            }
            (value, false) => value,
        };
        if end_loop {
            break;
        }
    }

    let word = output.join("@@ ");
    if !word.is_empty() {
        (&word[..word.len() - 4]).split(' ').map(|v| v.to_owned()).collect()
    } else {
        vec!(word)
    }
}