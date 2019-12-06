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
            data.insert(line.unwrap().trim().to_owned(), index);
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
            None => panic!("The special value {} could not be found in the vocabulary", token)
        };
        special_values.insert(String::from(token), token_id);
    }

    fn token_to_id(&self, token: &str) -> i64;

    fn convert_tokens_to_ids(&self, tokens: Vec<&str>) -> Vec<i64> {
        tokens.iter().map(|v| self.token_to_id(v)).collect()
    }
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

//==============================
// Unit tests
//==============================
#[cfg(test)]
mod tests {
    use super::*;
    use std::io;
    use std::io::Write;

    #[test]
    fn test_create_object() {
//        Given
        let values: HashMap<String, i64> = HashMap::new();
        let special_values: HashMap<String, i64> = HashMap::new();
        let unknown_value = BaseVocab::unknown_value();

//        When
        let base_vocab = BaseVocab {
            values,
            unknown_value,
            special_values,
        };

//        Then
        assert_eq!(base_vocab.unknown_value, "[UNK]");
        assert_eq!(base_vocab.unknown_value, BaseVocab::unknown_value());
        assert_eq!(base_vocab.values, *base_vocab.values());
        assert_eq!(base_vocab.special_values, *base_vocab.special_values());
    }

    #[test]
    fn test_create_object_from_file() -> Result<(), io::Error> {
//        Given
        let mut vocab_file = tempfile::NamedTempFile::new()?;
        write!(vocab_file, "hello \n world \n [UNK] \n !")?;
        let path = vocab_file.into_temp_path();
        let target_values: HashMap<String, i64> = [
            ("hello".to_owned(), 0),
            ("world".to_owned(), 1),
            ("[UNK]".to_owned(), 2),
            ("!".to_owned(), 3)
        ].iter().cloned().collect();

        let special_values: HashMap<String, i64> = [
            ("[UNK]".to_owned(), 2)
        ].iter().cloned().collect();

//        When
        let base_vocab = BaseVocab::from_file(path.to_path_buf().to_str().unwrap());

//        Then
        assert_eq!(base_vocab.unknown_value, "[UNK]");
        assert_eq!(base_vocab.values, target_values);
        assert_eq!(base_vocab.special_values, special_values);
        drop(path);
        Ok(())
    }

    #[test]
    #[should_panic]
    fn test_create_object_from_file_without_unknown_token() {
//        Given
        let mut vocab_file = tempfile::NamedTempFile::new().unwrap();
        write!(vocab_file, "hello \n world \n !").unwrap();
        let path = vocab_file.into_temp_path();

//        When & Then
        let _base_vocab = BaseVocab::from_file(path.to_path_buf().to_str().unwrap());
    }

    #[test]
    fn test_encode_tokens() -> Result<(), io::Error> {
//        Given
        let mut vocab_file = tempfile::NamedTempFile::new()?;
        write!(vocab_file, "hello \n world \n [UNK] \n !")?;
        let path = vocab_file.into_temp_path();
        let base_vocab = BaseVocab::from_file(path.to_path_buf().to_str().unwrap());

//        When & Then
        assert_eq!(base_vocab.token_to_id("hello"), 0);
        assert_eq!(base_vocab.token_to_id("world"), 1);
        assert_eq!(base_vocab.token_to_id("!"), 3);
        assert_eq!(base_vocab.token_to_id("[UNK]"), 2);
        assert_eq!(base_vocab.token_to_id("oov_value"), 2);

        drop(path);
        Ok(())
    }
}