// Copyright 2018 The Open AI Team Authors, The Google AI Language Team Authors
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

pub struct BertVocab {
    pub values: HashMap<String, i64>,
    pub indices: HashMap<i64, String>,
    pub unknown_value: &'static str,
    pub special_values: HashMap<String, i64>,
    pub special_indices: HashMap<i64, String>,
}

impl BertVocab {
    pub fn pad_value() -> &'static str { "[PAD]" }
    pub fn sep_value() -> &'static str { "[SEP]" }
    pub fn cls_value() -> &'static str { "[CLS]" }
    pub fn mask_value() -> &'static str { "[MASK]" }
}

impl Vocab for BertVocab {
    fn unknown_value() -> &'static str { "[UNK]" }

    fn values(&self) -> &HashMap<String, i64> {
        &self.values
    }

    fn indices(&self) -> &HashMap<i64, String> {&self.indices}

    fn special_values(&self) -> &HashMap<String, i64> { &self.special_values }

    fn special_indices(&self) -> &HashMap<i64, String> {&self.special_indices}

    fn from_file(path: &str) -> BertVocab {
        let values = BertVocab::read_vocab_file(path);
        let mut special_values = HashMap::new();

        let unknown_value = BertVocab::unknown_value();
        BertVocab::_register_as_special_value(unknown_value, &values, &mut special_values);

        let pad_value = BertVocab::pad_value();
        BertVocab::_register_as_special_value(pad_value, &values, &mut special_values);

        let sep_value = BertVocab::sep_value();
        BertVocab::_register_as_special_value(sep_value, &values, &mut special_values);

        let cls_value = BertVocab::cls_value();
        BertVocab::_register_as_special_value(cls_value, &values, &mut special_values);

        let mask_value = BertVocab::mask_value();
        BertVocab::_register_as_special_value(mask_value, &values, &mut special_values);

        let indices = swap_key_values(&values);
        let special_indices = swap_key_values(&special_values);

        BertVocab { values, indices, unknown_value, special_values, special_indices }
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
    fn test_create_object() {
//        Given
        let values: HashMap<String, i64> = HashMap::new();
        let special_values: HashMap<String, i64> = HashMap::new();
        let indices: HashMap<i64, String> = HashMap::new();
        let special_indices: HashMap<i64, String> = HashMap::new();
        let unknown_value = BertVocab::unknown_value();

//        When
        let base_vocab = BertVocab {
            values,
            indices,
            unknown_value,
            special_values,
            special_indices
        };

//        Then
        assert_eq!(base_vocab.unknown_value, "[UNK]");
        assert_eq!(base_vocab.unknown_value, BertVocab::unknown_value());
        assert_eq!(BertVocab::pad_value(), "[PAD]");
        assert_eq!(BertVocab::sep_value(), "[SEP]");
        assert_eq!(BertVocab::cls_value(), "[CLS]");
        assert_eq!(BertVocab::mask_value(), "[MASK]");
        assert_eq!(base_vocab.values, *base_vocab.values());
        assert_eq!(base_vocab.special_values, *base_vocab.special_values());
    }

    #[test]
    fn test_create_object_from_file() -> Result<(), io::Error> {
//        Given
        let mut vocab_file = tempfile::NamedTempFile::new()?;
        write!(vocab_file, "hello \n world \n [UNK] \n ! \n [CLS] \n [SEP] \n [MASK] \n [PAD]")?;
        let path = vocab_file.into_temp_path();
        let target_values: HashMap<String, i64> = [
            ("hello".to_owned(), 0),
            ("world".to_owned(), 1),
            ("[UNK]".to_owned(), 2),
            ("!".to_owned(), 3),
            ("[CLS]".to_owned(), 4),
            ("[SEP]".to_owned(), 5),
            ("[MASK]".to_owned(), 6),
            ("[PAD]".to_owned(), 7)
        ].iter().cloned().collect();

        let special_values: HashMap<String, i64> = [
            ("[UNK]".to_owned(), 2),
            ("[CLS]".to_owned(), 4),
            ("[SEP]".to_owned(), 5),
            ("[MASK]".to_owned(), 6),
            ("[PAD]".to_owned(), 7)
        ].iter().cloned().collect();

//        When
        let base_vocab = BertVocab::from_file(path.to_path_buf().to_str().unwrap());

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
        write!(vocab_file, "hello \n world \n [UNK] \n ! \n [CLS]").unwrap();
        let path = vocab_file.into_temp_path();

//        When & Then
        let _base_vocab = BertVocab::from_file(path.to_path_buf().to_str().unwrap());
    }

    #[test]
    fn test_encode_tokens() -> Result<(), io::Error> {
//        Given
        let mut vocab_file = tempfile::NamedTempFile::new()?;
        write!(vocab_file, "hello \n world \n [UNK] \n ! \n [CLS] \n [SEP] \n [MASK] \n [PAD]")?;
        let path = vocab_file.into_temp_path();
        let base_vocab = BertVocab::from_file(path.to_path_buf().to_str().unwrap());

//        When & Then
        assert_eq!(base_vocab.token_to_id("hello"), 0);
        assert_eq!(base_vocab.token_to_id("world"), 1);
        assert_eq!(base_vocab.token_to_id("!"), 3);
        assert_eq!(base_vocab.token_to_id("[UNK]"), 2);
        assert_eq!(base_vocab.token_to_id("oov_value"), 2);
        assert_eq!(base_vocab.token_to_id("[PAD]"), 7);
        assert_eq!(base_vocab.token_to_id("[MASK]"), 6);
        assert_eq!(base_vocab.token_to_id("[CLS]"), 4);
        assert_eq!(base_vocab.token_to_id("[SEP]"), 5);

        drop(path);
        Ok(())
    }

    #[test]
    fn test_decode_tokens() -> Result<(), io::Error> {
//        Given
        let mut vocab_file = tempfile::NamedTempFile::new()?;
        write!(vocab_file, "hello \n world \n [UNK] \n ! \n [CLS] \n [SEP] \n [MASK] \n [PAD]")?;
        let path = vocab_file.into_temp_path();
        let bert_vocab = BertVocab::from_file(path.to_path_buf().to_str().unwrap());

//        When & Then
        assert_eq!(bert_vocab.id_to_token(&(0 as i64)), "hello");
        assert_eq!(bert_vocab.id_to_token(&(1 as i64)), "world");
        assert_eq!(bert_vocab.id_to_token(&(3 as i64)), "!");
        assert_eq!(bert_vocab.id_to_token(&(2 as i64)), "[UNK]");
        assert_eq!(bert_vocab.id_to_token(&(7 as i64)), "[PAD]");
        assert_eq!(bert_vocab.id_to_token(&(6 as i64)), "[MASK]");
        assert_eq!(bert_vocab.id_to_token(&(4 as i64)), "[CLS]");
        assert_eq!(bert_vocab.id_to_token(&(5 as i64)), "[SEP]");

        drop(path);
        Ok(())
    }
}