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
use rust_transformers::preprocessing::adapters::Example;
use std::sync::Arc;
use rust_transformers::preprocessing::tokenizer::openai_gpt_tokenizer::OpenAiGptTokenizer;

fn main() {
    let _data = match rust_transformers::preprocessing::adapters::read_sst2(
        "E:/Coding/backup-rust/rust-transformers/resources/data/SST-2/train.tsv",
        b'\t') {
        Ok(examples) => {
            examples
        }
        Err(err) => {
            println!("{}", err);
            process::exit(1);
        }
    };

    let vocab_path = "E:/Coding/backup-rust/rust-transformers/resources/vocab/openai-gpt-vocab.json";
    let bpe_path = "E:/Coding/backup-rust/rust-transformers/resources/vocab/openai-gpt-merges.txt";
    let vocab = Arc::new(rust_transformers::OpenAiGptVocab::from_file(vocab_path));
    let _bpe_ranks = Rc::new(BpePairVocab::from_file(bpe_path));

    let _test_sentence = Example::new_from_string("This is a sample sentence to be tokenized");
    let openai_gpt_tokenizer: OpenAiGptTokenizer = OpenAiGptTokenizer::from_existing_vocab_and_merges(vocab.clone(), _bpe_ranks.clone());

    let _text_list: Vec<&str> = _data.iter().map(|v| v.sentence_1.as_ref()).collect();
    let _before = Instant::now();
    println!("{:?}", openai_gpt_tokenizer.encode(&_test_sentence.sentence_1, None, 128, &TruncationStrategy::LongestFirst, 0));
    println!("Elapsed time: {:.2?}", _before.elapsed());
}
