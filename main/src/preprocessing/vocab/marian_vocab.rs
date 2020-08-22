// Copyright 2018-2020 The HuggingFace Inc. team.
// Copyright 2020 Marian Team Authors
// Copyright 2019 Google LLC. All Rights Reserved.
// Copyright 2019-2020 Guillaume Becquin
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
use crate::preprocessing::vocab::base_vocab::swap_key_values;
use crate::Vocab;
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;

pub struct MarianVocab {
    pub values: HashMap<String, i64>,
    pub indices: HashMap<i64, String>,
    pub unknown_value: &'static str,
    pub special_values: HashMap<String, i64>,
    pub special_indices: HashMap<i64, String>,
}

impl MarianVocab {
    pub fn pad_value() -> &'static str {
        "<pad>"
    }
    pub fn eos_value() -> &'static str {
        "</s>"
    }
}

impl Vocab for MarianVocab {
    fn unknown_value() -> &'static str {
        "<unk>"
    }

    fn get_unknown_value(&self) -> &'static str {
        "<unk>"
    }

    fn values(&self) -> &HashMap<String, i64> {
        &self.values
    }

    fn indices(&self) -> &HashMap<i64, String> {
        &self.indices
    }

    fn special_values(&self) -> &HashMap<String, i64> {
        &self.special_values
    }

    fn special_indices(&self) -> &HashMap<i64, String> {
        &self.special_indices
    }

    fn from_file(path: &str) -> Result<MarianVocab, TokenizerError> {
        let f = File::open(path).map_err(|e| {
            TokenizerError::FileNotFound(format!("{} vocabulary file not found :{}", path, e))
        })?;
        let br = BufReader::new(f);
        let values: HashMap<String, i64> = match serde_json::from_reader(br) {
            Ok(value) => value,
            Err(e) => {
                return Err(TokenizerError::VocabularyParsingError(e.to_string()));
            }
        };

        let mut special_values = HashMap::new();
        let unknown_value = MarianVocab::unknown_value();
        MarianVocab::_register_as_special_value(unknown_value, &values, &mut special_values)?;

        let pad_value = MarianVocab::pad_value();
        MarianVocab::_register_as_special_value(pad_value, &values, &mut special_values)?;

        let eos_value = MarianVocab::eos_value();
        MarianVocab::_register_as_special_value(eos_value, &values, &mut special_values)?;

        let indices = swap_key_values(&values);
        let special_indices = swap_key_values(&special_values);

        Ok(MarianVocab {
            values,
            indices,
            unknown_value,
            special_values,
            special_indices,
        })
    }

    fn token_to_id(&self, token: &str) -> i64 {
        self._token_to_id(
            token,
            &self.values,
            &self.special_values,
            &self.unknown_value,
        )
    }

    fn id_to_token(&self, id: &i64) -> String {
        self._id_to_token(
            &id,
            &self.indices,
            &self.special_indices,
            &self.unknown_value,
        )
    }
}
