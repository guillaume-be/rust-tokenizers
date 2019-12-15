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
use crate::preprocessing::vocab::ctrl_vocab::BpePairVocab;
use std::collections::HashMap;
use crate::preprocessing::tokenizer::tokenization_utils::{is_whitespace, bpe};
use std::rc::Rc;
use std::cell::RefCell;


pub struct CtrlTokenizer {
    vocab: Rc<CtrlVocab>,
    bpe_ranks: Rc<BpePairVocab>,
    cache: RefCell<HashMap<String, Vec<String>>>,
}

impl CtrlTokenizer {
    pub fn from_file(vocab_path: &str, merges_path: &str) -> CtrlTokenizer {
        let vocab = Rc::new(CtrlVocab::from_file(vocab_path));
        let bpe_ranks = Rc::new(BpePairVocab::from_file(merges_path));
        let cache = RefCell::new(HashMap::new());
        CtrlTokenizer { vocab, bpe_ranks, cache }
    }

    pub fn from_existing_vocab_and_merges(vocab: Rc<CtrlVocab>, merges: Rc<BpePairVocab>) -> CtrlTokenizer {
        let cache = RefCell::new(HashMap::new());
        CtrlTokenizer { vocab, bpe_ranks: merges, cache }
    }
}

impl Tokenizer<CtrlVocab> for CtrlTokenizer {
    fn vocab(&self) -> &CtrlVocab {
        &self.vocab
    }

    fn tokenize(&self, text: &str) -> Vec<String> {
        let mut tokenized_text: Vec<String> = Vec::with_capacity(text.len());
        for word in text.trim().split(|v| is_whitespace(&v)) {
            let cached: bool = match self.cache.borrow().get(word) {
                Some(value) => {
                    tokenized_text.extend(value.clone());
                    true
                }
                None => false
            };
            if !cached {
                let bpe_output = bpe(word, &self.bpe_ranks);
                self.cache.borrow_mut().insert(word.to_owned(), bpe_output.clone());
                tokenized_text.extend(bpe_output);
            }
        };
        tokenized_text
    }
}
