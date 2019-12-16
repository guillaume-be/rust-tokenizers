// Copyright 2018 The Open AI Team Authors
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

use std::collections::HashMap;
use crate::preprocessing::vocab::base_vocab::Vocab;
use std::process;
use std::fs::File;
use std::io::BufReader;

pub struct Gpt2Vocab {
    pub values: HashMap<String, i64>,
    pub unknown_value: &'static str,
    pub special_values: HashMap<String, i64>,
}

impl Vocab for Gpt2Vocab {
    fn unknown_value() -> &'static str { "<|endoftext|>" }

    fn values(&self) -> &HashMap<String, i64> {
        &self.values
    }

    fn special_values(&self) -> &HashMap<String, i64> {
        &self.special_values
    }

    fn from_file(path: &str) -> Gpt2Vocab {
        let f = File::open(path).expect("Could not open vocabulary file.");
        let br = BufReader::new(f);
        let values: HashMap<String, i64> = serde_json::from_reader(br).expect("could not parse vocabulary");
        let mut special_values = HashMap::new();
        let unknown_value = Gpt2Vocab::unknown_value();
        Gpt2Vocab::_register_as_special_value(unknown_value, &values, &mut special_values);

        Gpt2Vocab { values, unknown_value, special_values }
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
    fn test_create_vocab() {
//        Given
        let values: HashMap<String, i64> = HashMap::new();
        let special_values: HashMap<String, i64> = HashMap::new();
        let unknown_value = Gpt2Vocab::unknown_value();

//        When
        let ctrl_vocab = Gpt2Vocab {
            values,
            unknown_value,
            special_values,
        };

//        Then
        assert_eq!(ctrl_vocab.unknown_value, "<|endoftext|>");
        assert_eq!(ctrl_vocab.unknown_value, Gpt2Vocab::unknown_value());
        assert_eq!(ctrl_vocab.values, *ctrl_vocab.values());
        assert_eq!(ctrl_vocab.special_values, *ctrl_vocab.special_values());
    }

    #[test]
    fn test_create_object_from_file() -> Result<(), io::Error> {
//        Given
        let mut vocab_file = tempfile::NamedTempFile::new()?;
        write!(vocab_file, "{{\"hello\": 1,\n \"world\": 0,\n \"<|endoftext|>\": 2,\n \"!\": 3\n}}")?;
        let path = vocab_file.into_temp_path();
        let target_values: HashMap<String, i64> = [
            ("hello".to_owned(), 1),
            ("world".to_owned(), 0),
            ("<|endoftext|>".to_owned(), 2),
            ("!".to_owned(), 3),
        ].iter().cloned().collect();

        let special_values: HashMap<String, i64> = [
            ("<|endoftext|>".to_owned(), 2)
        ].iter().cloned().collect();

//        When
        let ctrl_vocab = Gpt2Vocab::from_file(path.to_path_buf().to_str().unwrap());

//        Then
        assert_eq!(ctrl_vocab.unknown_value, "<|endoftext|>");
        assert_eq!(ctrl_vocab.values, target_values);
        assert_eq!(ctrl_vocab.special_values, special_values);
        drop(path);
        Ok(())
    }

    #[test]
    #[should_panic]
    fn test_create_object_from_file_without_unknown_token() {
//        Given
        let mut vocab_file = tempfile::NamedTempFile::new().unwrap();
        write!(vocab_file, "{{\"hello\": 1,\n \"world\": 0,\n \"!\": 3\n}}").unwrap();
        let path = vocab_file.into_temp_path();

//        When & Then
        let _ctrl_vocab = Gpt2Vocab::from_file(path.to_path_buf().to_str().unwrap());
    }

    #[test]
    fn test_encode_tokens() -> Result<(), io::Error> {
//        Given
        let mut vocab_file = tempfile::NamedTempFile::new()?;
        write!(vocab_file, "{{\"hello\": 1,\n \"world\": 0,\n \"<|endoftext|>\": 2,\n \"!\": 3\n}}")?;
        let path = vocab_file.into_temp_path();
        let ctrl_vocab = Gpt2Vocab::from_file(path.to_path_buf().to_str().unwrap());

//        When & Then
        assert_eq!(ctrl_vocab.token_to_id("hello"), 1);
        assert_eq!(ctrl_vocab.token_to_id("world"), 0);
        assert_eq!(ctrl_vocab.token_to_id("!"), 3);
        assert_eq!(ctrl_vocab.token_to_id("<|endoftext|>"), 2);
        assert_eq!(ctrl_vocab.token_to_id("oov_value"), 2);

        drop(path);
        Ok(())
    }
}