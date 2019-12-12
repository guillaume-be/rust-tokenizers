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

use rust_transformers;
use rust_transformers::preprocessing::vocab::base_vocab::Vocab;
use rust_transformers::preprocessing::adapters::Example;
//use rust_transformers::preprocessing::tokenizer::bert_tokenizer::BertTokenizer;

//use rust_transformers::preprocessing::tokenizer::base_tokenizer::{Tokenizer, TruncationStrategy};
use std::sync::Arc;
use rust_transformers::preprocessing::vocab::ctrl_vocab::BpePairVocab;
use rust_transformers::preprocessing::tokenizer::ctrl_tokenizer::{CtrlTokenizer, bpe};
use rust_transformers::preprocessing::tokenizer::base_tokenizer::Tokenizer;

fn main() {
    let vocab_path = "E:/Coding/rust-transformers/resources/vocab/ctrl-vocab.json";
    let bpe_path = "E:/Coding/rust-transformers/resources/vocab/ctrl-merges.txt";
    let ctrl_vocab = Arc::new(rust_transformers::CtrlVocab::from_file(vocab_path));
    let _bpe_ranks = Arc::new(BpePairVocab::from_file(bpe_path));

    let _test_sentence = Example::new_from_string("[MASK]Reprise �au tout début des années [SEP]1960[SEP] par le commissariat à l'énergie atomique (CEA), cette structure reste, au xxie siècle, l'un des principaux employeurs de main d'œuvre de la commune.");
//    println!("{:?}", _bpe_ranks.pair_to_id("r", "o"));
    let ctrl_tokenizer: CtrlTokenizer = CtrlTokenizer::from_existing_vocab_and_merges(ctrl_vocab.clone(), _bpe_ranks.clone());
    let tokenized_text = ctrl_tokenizer.tokenize(&_test_sentence.sentence_1);

    println!("{:?}", tokenized_text.len());
    println!("{:?}", bpe("he", &_bpe_ranks));


}
