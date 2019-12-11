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
use crate::preprocessing::vocab::ctrl_vocab::BpePairVocab;
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
            tokenized_text.push(word.as_str().to_owned());
        };
        tokenized_text
    }
}