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

use crate::error::TokenizerError;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::mem::ManuallyDrop;
use std::ptr;

#[derive(Eq, PartialEq, Hash, Clone, Debug)]
pub struct BpePairRef<'a> {
    pub byte_1: &'a String,
    pub byte_2: &'a String,
}

pub struct BpePairVocab {
    pub values: HashMap<(String, String), i64>,
}

impl BpePairVocab {
    pub fn from_file(path: &str) -> Result<BpePairVocab, TokenizerError> {
        let f = File::open(path).map_err(|e| {
            TokenizerError::FileNotFound(format!("{} vocabulary file not found :{}", path, e))
        })?;
        let br = BufReader::new(f);
        let mut data = HashMap::new();
        let mut index = 0;
        for line in br.lines().skip(1) {
            let line = match line {
                Ok(value) => value,
                Err(e) => {
                    return Err(TokenizerError::VocabularyParsingError(e.to_string()));
                }
            };
            let tuple: Vec<String> = line.trim().split(' ').map(|v| v.to_owned()).collect();
            if tuple.len() > 1 {
                data.insert((tuple[0].clone(), tuple[1].clone()), index);
                index += 1;
            }
        }

        Ok(BpePairVocab { values: data })
    }

    pub fn byte_pair_to_id(&self, byte_pair: &BpePairRef) -> Option<&i64> {
        unsafe {
            let byte_1 = byte_pair.byte_1;
            let byte_2 = byte_pair.byte_2;
            let k = (ptr::read(byte_1), ptr::read(byte_2));
            let k = ManuallyDrop::new(k);
            let v = self.values.get(&k);
            v
        }
    }
}

//==============================
// Unit tests
//==============================
#[cfg(test)]
mod tests {
    extern crate anyhow;
    use super::*;
    use std::io::Write;

    #[test]
    fn test_create_pair_vocab() {
        //        Given
        let values: HashMap<(String, String), i64> = HashMap::new();

        //        When
        let pair_vocab = BpePairVocab {
            values: values.clone(),
        };

        //        Then
        assert_eq!(pair_vocab.values, values);
    }

    #[test]
    fn test_create_pair_vocab_from_file() -> anyhow::Result<()> {
        //        Given
        let mut merges_file = tempfile::NamedTempFile::new()?;
        write!(merges_file, "#version: 0.1\n t h\na n\ni n\nth e</w>")?;
        let path = merges_file.into_temp_path();
        let target_values: HashMap<(String, String), i64> = [
            (("t".to_owned(), "h".to_owned()), 0),
            (("a".to_owned(), "n".to_owned()), 1),
            (("i".to_owned(), "n".to_owned()), 2),
            (("th".to_owned(), "e</w>".to_owned()), 3),
        ]
        .iter()
        .cloned()
        .collect();

        //        When
        let pair_vocab = BpePairVocab::from_file(path.to_path_buf().to_str().unwrap())?;

        //        Then
        assert_eq!(pair_vocab.values, target_values);
        drop(path);
        Ok(())
    }

    #[test]
    fn test_encode_byte_pairs() -> anyhow::Result<()> {
        //        Given
        let mut merges_file = tempfile::NamedTempFile::new()?;
        write!(merges_file, "#version: 0.1\n t h\na n\ni n\nth e</w>")?;
        let path = merges_file.into_temp_path();
        let pair_vocab = BpePairVocab::from_file(path.to_path_buf().to_str().unwrap())?;

        //        Given
        let t_token = String::from("t");
        let h_token = String::from("h");
        let a_token = String::from("a");
        let i_token = String::from("i");
        let n_token = String::from("n");
        let th_token = String::from("th");
        let e_eow_token = String::from("e</w>");

        let test_tuples = [
            ((t_token.clone(), h_token.clone()), Some(&(0 as i64))),
            ((a_token.clone(), n_token.clone()), Some(&(1 as i64))),
            ((i_token.clone(), n_token.clone()), Some(&(2 as i64))),
            ((th_token.clone(), e_eow_token.clone()), Some(&(3 as i64))),
            ((a_token.clone(), e_eow_token.clone()), None),
        ];

        //        When & Then
        for (input, expected_output) in &test_tuples {
            assert_eq!(
                pair_vocab.byte_pair_to_id(&BpePairRef {
                    byte_1: &input.0,
                    byte_2: &input.1
                }),
                *expected_output
            );
        }

        drop(path);
        Ok(())
    }
}
