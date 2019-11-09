use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufRead};

pub trait Vocab {
    fn from_file(path: &str) -> Self;
//    ToDo: move the file parsing over here
//    fn _read_contents(path: &str) -> HashMap<String, i64>;
}

pub struct BaseVocab {
    pub values: HashMap<String, i64>
}

impl Vocab for BaseVocab {
    fn from_file(path: &str) -> BaseVocab {

        let f = File::open(path).expect("Could not open vocabulary file.");
        let br = BufReader::new(f);
        let mut data = HashMap::new();
        let mut index = 0;

        for line in br.lines() {
            data.insert(line.unwrap(), index);
            index += 1;
    }

        BaseVocab {
            values: data
        }
    }
}