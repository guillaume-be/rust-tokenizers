use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufRead};
use std::error::Error;
use std::process;

pub trait Vocab {
    fn unknown_value() -> &'static str;

    fn values(&self) -> &HashMap<String, i64>;

    fn special_values(&self) -> &HashMap<String, i64>;

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

    fn _token_to_id(&self,
                    token: &str,
                    values: &HashMap<String, i64>,
                    special_values: &HashMap<String, i64>,
                    unknown_value: &str) -> Result<i64, Box<dyn Error>> {
        match special_values.get(token) {
            Some(index) => Ok(*index),
            None => match values.get(token) {
                Some(index) => Ok(*index),
                None => match values.get(unknown_value) {
                    Some(index) => Ok(*index),
                    None => Err("Could not decode token".into())
                }
            }
        }
    }

    fn _register_as_special_value(token: &str,
                                  values: &HashMap<String, i64>,
                                  special_values: &mut HashMap<String, i64>) {
        let token_id = match values.get(token) {
            Some(index) => *index,
            None => panic!("The unknown value could not be found in the vocabulary")
        };
        special_values.insert(String::from(token), token_id);
    }

    fn token_to_id(&self, token: &str) -> i64;
}


pub struct BaseVocab {
    pub values: HashMap<String, i64>,
    pub unknown_value: &'static str,
    pub special_values: HashMap<String, i64>,
}

impl Vocab for BaseVocab {
    fn unknown_value() -> &'static str { "[UNK]" }

    fn values(&self) -> &HashMap<String, i64> {
        &self.values
    }

    fn special_values(&self) -> &HashMap<String, i64> {
        &self.special_values
    }


    fn from_file(path: &str) -> BaseVocab {
        let values = BaseVocab::read_vocab_file(path);
        let mut special_values = HashMap::new();

        let unknown_value = BaseVocab::unknown_value();
        BaseVocab::_register_as_special_value(unknown_value, &values, &mut special_values);

        BaseVocab { values, unknown_value, special_values }
    }

    fn token_to_id(&self, token: &str) -> i64 {
        match self._token_to_id(token, &self.values, &self.special_values, &self.unknown_value) {
            Ok(index) => index,
            Err(err) => {
                println!("{}", err);
                process::exit(1);
            }
        }
    }
}