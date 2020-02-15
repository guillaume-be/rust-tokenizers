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

use rust_tokenizers;
use rust_tokenizers::preprocessing::vocab::base_vocab::Vocab;
use rust_tokenizers::preprocessing::tokenizer::base_tokenizer::{TruncationStrategy, Tokenizer};
use std::env;
use rust_tokenizers::preprocessing::adapters::Example;
use std::sync::Arc;
use rust_tokenizers::BertTokenizer;

fn main() {


    let vocab_path = env::var("bert_vocab").expect("`bert_vocab` environment variable not set");
    let vocab = Arc::new(rust_tokenizers::BertVocab::from_file(&vocab_path));


    let _test_sentence = Example::new_from_string("This is a sample sentence to be tokenized");
    let bert_tokenizer: BertTokenizer = BertTokenizer::from_existing_vocab(vocab.clone());

    println!("{:?}", bert_tokenizer.encode(&_test_sentence.sentence_1,
                                           None,
                                           128,
                                           &TruncationStrategy::LongestFirst,
                                           0));
}
