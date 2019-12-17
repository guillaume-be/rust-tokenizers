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
use rust_transformers::preprocessing::tokenizer::base_tokenizer::{TruncationStrategy, Tokenizer};
use std::process;
use std::time::Instant;
use std::rc::Rc;
use rust_transformers::preprocessing::vocab::bpe_vocab::BpePairVocab;
use rust_transformers::preprocessing::tokenizer::gpt2_tokenizer::Gpt2Tokenizer;
use rust_transformers::preprocessing::adapters::Example;
use rust_transformers::preprocessing::tokenizer::ctrl_tokenizer::CtrlTokenizer;

fn main() {
    let _data = match rust_transformers::preprocessing::adapters::read_sst2(
        "E:/Coding/rust-transformers/resources/data/SST-2/train.tsv",
        b'\t') {
        Ok(examples) => {
            examples
        }
        Err(err) => {
            println!("{}", err);
            process::exit(1);
        }
    };

    let vocab_path = "E:/Coding/rust-transformers/resources/vocab/gpt2-vocab.json";
    let bpe_path = "E:/Coding/rust-transformers/resources/vocab/gpt2-merges.txt";
    let ctrlvocab_path = "E:/Coding/rust-transformers/resources/vocab/ctrl-vocab.json";
    let ctrlbpe_path = "E:/Coding/rust-transformers/resources/vocab/ctrl-merges.txt";
    let gpt2_vocab = Rc::new(rust_transformers::Gpt2Vocab::from_file(vocab_path));
    let _bpe_ranks = Rc::new(BpePairVocab::from_file(bpe_path));
    let ctrl_vocab = Rc::new(rust_transformers::CtrlVocab::from_file(ctrlvocab_path));
    let ctrl_bpe_ranks = Rc::new(BpePairVocab::from_file(ctrlbpe_path));

    let _test_sentence = Example::new_from_string(" eprise");
//    println!("{:?}", _bpe_ranks.pair_to_id("r", "o"));
    let gpt2_tokenizer: Gpt2Tokenizer = Gpt2Tokenizer::from_existing_vocab_and_merges(gpt2_vocab.clone(), _bpe_ranks.clone());
    let ctrl_tokenizer: CtrlTokenizer = CtrlTokenizer::from_existing_vocab_and_merges(ctrl_vocab.clone(), ctrl_bpe_ranks.clone());
//    let tokenized_text = gpt2_tokenizer.tokenize(&_test_sentence.sentence_1);
    let _text_list: Vec<&str> = _data.iter().map(|v| v.sentence_1.as_ref()).collect();
    let _before = Instant::now();

//    let _results = gpt2_tokenizer.encode_list(_text_list, 128, &TruncationStrategy::LongestFirst, 0);
//    for text in _text_list{
//        gpt2_tokenizer.tokenize(text);
//    }
    for text in _text_list{
        ctrl_tokenizer.tokenize(text);
    }
//    println!("{:?}", gpt2_tokenizer.tokenize(&_test_sentence.sentence_1));
//    println!("{:?}", ctrl_tokenizer.tokenize(&_test_sentence.sentence_1));
//    println!("{:?}", gpt2_tokenizer.encode(&_test_sentence.sentence_1, None,128, &TruncationStrategy::LongestFirst, 0));
    println!("Elapsed time: {:.2?}", _before.elapsed());
}
