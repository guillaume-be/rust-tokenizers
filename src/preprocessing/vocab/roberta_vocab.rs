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

pub struct RobertaVocab {
    pub values: HashMap<String, i64>,
    pub unknown_value: &'static str,
    pub special_values: HashMap<String, i64>,
}

impl RobertaVocab {
    pub(crate) fn pad_value() -> &'static str { "<pad>" }
    pub(crate) fn bos_value() -> &'static str { "<s>" }
    pub(crate) fn eos_value() -> &'static str { "</s>" }
    pub(crate) fn sep_value() -> &'static str { "</s>" }
    pub(crate) fn cls_value() -> &'static str { "<s>" }
    pub(crate) fn mask_value() -> &'static str { "<mask>" }
}

impl Vocab for RobertaVocab {
    fn unknown_value() -> &'static str { "<unk>" }

    fn values(&self) -> &HashMap<String, i64> {
        &self.values
    }

    fn special_values(&self) -> &HashMap<String, i64> {
        &self.special_values
    }

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

        RobertaVocab { values, unknown_value, special_values }
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
        let unknown_value = RobertaVocab::unknown_value();

//        When
        let roberta_vocab = RobertaVocab {
            values,
            unknown_value,
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
}