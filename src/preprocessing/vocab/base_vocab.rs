use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufRead};

pub trait Vocab {
    fn from_file(path: &str) -> Self;

    fn read_vocab_file(path: &str) -> HashMap<String, i64> {
        let f = File::open(path).expect("Could not open vocabulary file.");
        let br = BufReader::new(f);
        let mut data = HashMap::new();
        let mut index = 0;

        for line in br.lines() {
            data.insert(line.unwrap(), index);
            index += 1;
        };
        data
    }
}

pub struct BaseVocab {
    pub values: HashMap<String, i64>
}

impl Vocab for BaseVocab {
    fn from_file(path: &str) -> BaseVocab {
        let data = BaseVocab::read_vocab_file(path);
        BaseVocab {
            values: data
        }
    }
}