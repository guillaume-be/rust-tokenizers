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
use crate::preprocessing::vocab::base_vocab::{Vocab, swap_key_values};
use std::process;
use std::fs::File;
use std::io::BufReader;

pub struct RobertaVocab {
    ///A mapping of tokens as string to indices (i.e. the encoder base)
    pub values: HashMap<String, i64>,

    ///A mapping of token IDs to strings (i.e. the decoder base)
    pub indices: HashMap<i64, String>,

    ///The string to use for unknown (out of vocabulary) tokens
    pub unknown_value: &'static str,

    ///A mapping of special value tokens as strings to IDs (i.e. the encoder base for special
    ///values), special values typically include things like BOS/EOS markers, class markers, mask
    ///markers and padding markers
    pub special_values: HashMap<String, i64>,

    ///A mapping of special value tokens as IDs to strings (i.e. the decoder base for special values)
    pub special_indices: HashMap<i64, String>,
}

impl RobertaVocab {
    pub fn pad_value() -> &'static str { "<pad>" }
    pub fn bos_value() -> &'static str { "<s>" }
    pub fn eos_value() -> &'static str { "</s>" }
    pub fn sep_value() -> &'static str { "</s>" }
    pub fn cls_value() -> &'static str { "<s>" }
    pub fn mask_value() -> &'static str { "<mask>" }
}

impl Vocab for RobertaVocab {
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

    ///Read a Roberta-style vocab.json file
    fn from_file(path: &str) -> RobertaVocab {
        let f = File::open(path).expect("Could not open vocabulary file.");
        let br = BufReader::new(f);
        let values: HashMap<String, i64> = serde_json::from_reader(br).expect("could not parse vocabulary");
        let mut special_values = HashMap::new();
        let unknown_value = RobertaVocab::unknown_value();
        RobertaVocab::_register_as_special_value(unknown_value, &values, &mut special_values);

        let pad_value = RobertaVocab::pad_value();
        RobertaVocab::_register_as_special_value(pad_value, &values, &mut special_values);

        let sep_value = RobertaVocab::sep_value();
        RobertaVocab::_register_as_special_value(sep_value, &values, &mut special_values);

        let cls_value = RobertaVocab::cls_value();
        RobertaVocab::_register_as_special_value(cls_value, &values, &mut special_values);

        let mask_value = RobertaVocab::mask_value();
        RobertaVocab::_register_as_special_value(mask_value, &values, &mut special_values);

        let bos_value = RobertaVocab::bos_value();
        RobertaVocab::_register_as_special_value(bos_value, &values, &mut special_values);

        let eos_value = RobertaVocab::eos_value();
        RobertaVocab::_register_as_special_value(eos_value, &values, &mut special_values);

        let indices = swap_key_values(&values);
        let special_indices = swap_key_values(&special_values);

        RobertaVocab { values, indices, unknown_value, special_values, special_indices }
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
        let unknown_value = RobertaVocab::unknown_value();

//        When
        let roberta_vocab = RobertaVocab {
            values,
            indices,
            unknown_value,
            special_indices,
            special_values,
        };

//        Then
        assert_eq!(roberta_vocab.unknown_value, "<unk>");
        assert_eq!(RobertaVocab::pad_value(), "<pad>");
        assert_eq!(RobertaVocab::sep_value(), "</s>");
        assert_eq!(RobertaVocab::bos_value(), "<s>");
        assert_eq!(RobertaVocab::eos_value(), "</s>");
        assert_eq!(RobertaVocab::cls_value(), "<s>");
        assert_eq!(RobertaVocab::mask_value(), "<mask>");
        assert_eq!(roberta_vocab.unknown_value, RobertaVocab::unknown_value());
        assert_eq!(roberta_vocab.values, *roberta_vocab.values());
        assert_eq!(roberta_vocab.special_values, *roberta_vocab.special_values());
    }

    #[test]
    fn test_create_object_from_file() -> Result<(), io::Error> {
//        Given
        let mut vocab_file = tempfile::NamedTempFile::new()?;
        write!(vocab_file, "{{\"hello\": 1,\n \"world\": 0,\n \"<unk>\": 2,\n \"!\": 3\n, \"<pad>\": 4\n, \"<s>\": 5\n, \"</s>\": 6\n, \"<mask>\": 7\n}}")?;
        let path = vocab_file.into_temp_path();
        let target_values: HashMap<String, i64> = [
            ("hello".to_owned(), 1),
            ("world".to_owned(), 0),
            ("<unk>".to_owned(), 2),
            ("!".to_owned(), 3),
            ("<pad>".to_owned(), 4),
            ("<s>".to_owned(), 5),
            ("</s>".to_owned(), 6),
            ("<mask>".to_owned(), 7),
        ].iter().cloned().collect();

        let special_values: HashMap<String, i64> = [
            ("<unk>".to_owned(), 2),
            ("<pad>".to_owned(), 4),
            ("<s>".to_owned(), 5),
            ("</s>".to_owned(), 6),
            ("<mask>".to_owned(), 7),
        ].iter().cloned().collect();

//        When
        let roberta_vocab = RobertaVocab::from_file(path.to_path_buf().to_str().unwrap());

//        Then
        assert_eq!(roberta_vocab.unknown_value, "<unk>");
        assert_eq!(roberta_vocab.values, target_values);
        assert_eq!(roberta_vocab.special_values, special_values);
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
        let _roberta_vocab = RobertaVocab::from_file(path.to_path_buf().to_str().unwrap());
    }

    #[test]
    fn test_encode_tokens() -> Result<(), io::Error> {
//        Given
        let mut vocab_file = tempfile::NamedTempFile::new()?;
        write!(vocab_file, "{{\"hello\": 1,\n \"world\": 0,\n \"<unk>\": 2,\n \"!\": 3\n, \"<pad>\": 4\n, \"<s>\": 5\n, \"</s>\": 6\n, \"<mask>\": 7\n}}")?;
        let path = vocab_file.into_temp_path();
        let roberta_vocab = RobertaVocab::from_file(path.to_path_buf().to_str().unwrap());

//        When & Then
        assert_eq!(roberta_vocab.token_to_id("hello"), 1);
        assert_eq!(roberta_vocab.token_to_id("world"), 0);
        assert_eq!(roberta_vocab.token_to_id("!"), 3);
        assert_eq!(roberta_vocab.token_to_id("<unk>"), 2);
        assert_eq!(roberta_vocab.token_to_id("<s>"), 5);
        assert_eq!(roberta_vocab.token_to_id("</s>"), 6);
        assert_eq!(roberta_vocab.token_to_id("<mask>"), 7);
        assert_eq!(roberta_vocab.token_to_id("<pad>"), 4);

        drop(path);
        Ok(())
    }

    #[test]
    fn test_decode_tokens() -> Result<(), io::Error> {
//        Given
        let mut vocab_file = tempfile::NamedTempFile::new()?;
        write!(vocab_file, "{{\"hello\": 1,\n \"world\": 0,\n \"<unk>\": 2,\n \"!\": 3\n, \"<pad>\": 4\n, \"<s>\": 5\n, \"</s>\": 6\n, \"<mask>\": 7\n}}")?;
        let path = vocab_file.into_temp_path();
        let roberta_vocab = RobertaVocab::from_file(path.to_path_buf().to_str().unwrap());

//        When & Then
        assert_eq!(roberta_vocab.id_to_token(&(1 as i64)), "hello");
        assert_eq!(roberta_vocab.id_to_token(&(0 as i64)), "world");
        assert_eq!(roberta_vocab.id_to_token(&(3 as i64)), "!");
        assert_eq!(roberta_vocab.id_to_token(&(2 as i64)), "<unk>");
        assert_eq!(roberta_vocab.id_to_token(&(5 as i64)), "<s>");
        assert_eq!(roberta_vocab.id_to_token(&(6 as i64)), "</s>");
        assert_eq!(roberta_vocab.id_to_token(&(7 as i64)), "<mask>");
        assert_eq!(roberta_vocab.id_to_token(&(4 as i64)), "<pad>");
        drop(path);
        Ok(())
    }
}
