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

use std::collections::HashMap;
use crate::preprocessing::vocab::base_vocab::{Vocab, swap_key_values};
use std::process;
use std::fs::File;
use std::io::BufReader;

pub struct OpenAiGptVocab {
    pub values: HashMap<String, i64>,
    pub indices: HashMap<i64, String>,
    pub unknown_value: &'static str,
    pub special_values: HashMap<String, i64>,
    pub special_indices: HashMap<i64, String>,
}

impl Vocab for OpenAiGptVocab {
    fn unknown_value() -> &'static str { "<unk>" }

    fn get_unknown_value(&self) -> &'static str { "<unk>" }

    fn values(&self) -> &HashMap<String, i64> {
        &self.values
    }

    fn indices(&self) -> &HashMap<i64, String> { &self.indices }

    fn special_values(&self) -> &HashMap<String, i64> {
        &self.special_values
    }

    fn special_indices(&self) -> &HashMap<i64, String> { &self.special_indices }

    fn from_file(path: &str) -> OpenAiGptVocab {
        let f = File::open(path).expect("Could not open vocabulary file.");
        let br = BufReader::new(f);
        let values: HashMap<String, i64> = serde_json::from_reader(br).expect("could not parse vocabulary");
        let mut special_values = HashMap::new();
        let unknown_value = OpenAiGptVocab::unknown_value();
        OpenAiGptVocab::_register_as_special_value(unknown_value, &values, &mut special_values);

        let indices = swap_key_values(&values);
        let special_indices = swap_key_values(&special_values);

        OpenAiGptVocab { values, indices, unknown_value, special_values, special_indices }
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

    fn id_to_token(&self, id: &i64) -> String {
        match self._id_to_token(&id, &self.indices, &self.special_indices, &self.unknown_value) {
            Ok(token) => token,
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
        let indices: HashMap<i64, String> = HashMap::new();
        let special_indices: HashMap<i64, String> = HashMap::new();
        let unknown_value = OpenAiGptVocab::unknown_value();

//        When
        let openai_gpt_vocab = OpenAiGptVocab {
            values,
            indices,
            unknown_value,
            special_indices,
            special_values,
        };

//        Then
        assert_eq!(openai_gpt_vocab.unknown_value, "<unk>");
        assert_eq!(openai_gpt_vocab.unknown_value, OpenAiGptVocab::unknown_value());
        assert_eq!(openai_gpt_vocab.values, *openai_gpt_vocab.values());
        assert_eq!(openai_gpt_vocab.special_values, *openai_gpt_vocab.special_values());
    }

    #[test]
    fn test_create_object_from_file() -> Result<(), io::Error> {
//        Given
        let mut vocab_file = tempfile::NamedTempFile::new()?;
        write!(vocab_file, "{{\"hello\": 1,\n \"world\": 0,\n \"<unk>\": 2,\n \"!\": 3\n}}")?;
        let path = vocab_file.into_temp_path();
        let target_values: HashMap<String, i64> = [
            ("hello".to_owned(), 1),
            ("world".to_owned(), 0),
            ("<unk>".to_owned(), 2),
            ("!".to_owned(), 3),
        ].iter().cloned().collect();

        let special_values: HashMap<String, i64> = [
            ("<unk>".to_owned(), 2)
        ].iter().cloned().collect();

//        When
        let openai_gpt_vocab = OpenAiGptVocab::from_file(path.to_path_buf().to_str().unwrap());

//        Then
        assert_eq!(openai_gpt_vocab.unknown_value, "<unk>");
        assert_eq!(openai_gpt_vocab.values, target_values);
        assert_eq!(openai_gpt_vocab.special_values, special_values);
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
        let _ctrl_vocab = OpenAiGptVocab::from_file(path.to_path_buf().to_str().unwrap());
    }

    #[test]
    fn test_encode_tokens() -> Result<(), io::Error> {
//        Given
        let mut vocab_file = tempfile::NamedTempFile::new()?;
        write!(vocab_file, "{{\"hello\": 1,\n \"world\": 0,\n \"<unk>\": 2,\n \"!\": 3\n}}")?;
        let path = vocab_file.into_temp_path();
        let openai_gpt_vocab = OpenAiGptVocab::from_file(path.to_path_buf().to_str().unwrap());

//        When & Then
        assert_eq!(openai_gpt_vocab.token_to_id("hello"), 1);
        assert_eq!(openai_gpt_vocab.token_to_id("world"), 0);
        assert_eq!(openai_gpt_vocab.token_to_id("!"), 3);
        assert_eq!(openai_gpt_vocab.token_to_id("<unk>"), 2);
        assert_eq!(openai_gpt_vocab.token_to_id("oov_value"), 2);

        drop(path);
        Ok(())
    }

    #[test]
    fn test_decode_tokens() -> Result<(), io::Error> {
//        Given
        let mut vocab_file = tempfile::NamedTempFile::new()?;
        write!(vocab_file, "{{\"hello\": 1,\n \"world\": 0,\n \"<unk>\": 2,\n \"!\": 3\n}}")?;
        let path = vocab_file.into_temp_path();
        let openai_gpt_vocab = OpenAiGptVocab::from_file(path.to_path_buf().to_str().unwrap());

//        When & Then
        assert_eq!(openai_gpt_vocab.id_to_token(&(1 as i64)), "hello");
        assert_eq!(openai_gpt_vocab.id_to_token(&(0 as i64)), "world");
        assert_eq!(openai_gpt_vocab.id_to_token(&(3 as i64)), "!");
        assert_eq!(openai_gpt_vocab.id_to_token(&(2 as i64)), "<unk>");

        drop(path);
        Ok(())
    }
}
