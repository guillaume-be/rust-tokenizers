// Copyright 2018 The Open AI Team Authors
// Copyright 2020 Microsoft and the HuggingFace Inc. team.
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
use crate::vocab::base_vocab::{swap_key_values, Vocab};
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

/// # DeBERTa Vocab
/// Vocabulary for DeBERTa tokenizer. Contains the following special values:
/// - BOS token
/// - EOS token
/// - CLS token
/// - SEP token
/// - UNK token
/// - PAD token
/// - MASK token
///
/// Expects a JSON-format vocabulary when created from file.
#[derive(Debug, Clone)]
pub struct DeBERTaVocab {
    /// A mapping of tokens as string to indices (i.e. the encoder base)
    pub values: HashMap<String, i64>,

    /// A mapping of token ids to strings (i.e. the decoder base)
    pub indices: HashMap<i64, String>,

    /// The string to use for unknown (out of vocabulary) tokens
    pub unknown_value: &'static str,

    /// A mapping of special value tokens as strings to IDs (i.e. the encoder base for special
    /// values), special values typically include things like BOS/EOS markers, class markers, mask
    /// markers and padding markers
    pub special_values: HashMap<String, i64>,

    /// A mapping of special value tokens as IDs to strings (i.e. the decoder base for special values)
    pub special_indices: HashMap<i64, String>,
}

impl DeBERTaVocab {
    /// Returns the BOS token for DeBERTa (`[CLS]`)
    pub fn bos_value() -> &'static str {
        "[CLS]"
    }

    /// Returns the EOS token for DeBERTa (`[SEP]`)
    pub fn eos_value() -> &'static str {
        "[SEP]"
    }

    /// Returns the SEP token for DeBERTa (`[SEP]`)
    pub fn sep_value() -> &'static str {
        "[SEP]"
    }

    /// Returns the CLS token for DeBERTa (`[CLS]`)
    pub fn cls_value() -> &'static str {
        "[CLS]"
    }

    /// Returns the MASK token for DeBERTa (`[MASK]`)
    pub fn mask_value() -> &'static str {
        "[MASK]"
    }

    /// Returns the PAD token for DeBERTa (`[PAD]`)
    pub fn pad_value() -> &'static str {
        "[PAD]"
    }
}

impl Vocab for DeBERTaVocab {
    fn unknown_value() -> &'static str {
        "[UNK]"
    }

    fn get_unknown_value(&self) -> &'static str {
        "[UNK]"
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

    fn from_file<V: AsRef<Path>, S: AsRef<Path>>(
        vocab: V,
        _special: Option<S>,
    ) -> Result<DeBERTaVocab, TokenizerError> {
        let f = File::open(&vocab).map_err(|e| {
            TokenizerError::FileNotFound(format!(
                "{} vocabulary file not found :{}",
                vocab.as_ref().display(),
                e
            ))
        })?;
        let br = BufReader::new(f);
        let values: HashMap<String, i64> = match serde_json::from_reader(br) {
            Ok(value) => value,
            Err(e) => {
                return Err(TokenizerError::VocabularyParsingError(e.to_string()));
            }
        };
        let mut special_values = HashMap::new();
        let unknown_value = DeBERTaVocab::unknown_value();
        DeBERTaVocab::_register_as_special_value(unknown_value, &values, &mut special_values)?;

        let bos_value = DeBERTaVocab::bos_value();
        DeBERTaVocab::_register_as_special_value(bos_value, &values, &mut special_values)?;

        let eos_value = DeBERTaVocab::eos_value();
        DeBERTaVocab::_register_as_special_value(eos_value, &values, &mut special_values)?;

        let cls_value = DeBERTaVocab::cls_value();
        DeBERTaVocab::_register_as_special_value(cls_value, &values, &mut special_values)?;

        let mask_value = DeBERTaVocab::mask_value();
        DeBERTaVocab::_register_as_special_value(mask_value, &values, &mut special_values)?;

        let pad_value = DeBERTaVocab::pad_value();
        DeBERTaVocab::_register_as_special_value(pad_value, &values, &mut special_values)?;

        let sep_value = DeBERTaVocab::sep_value();
        DeBERTaVocab::_register_as_special_value(sep_value, &values, &mut special_values)?;

        let indices = swap_key_values(&values);
        let special_indices = swap_key_values(&special_values);

        Ok(DeBERTaVocab {
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
            self.unknown_value,
        )
    }

    fn id_to_token(&self, id: &i64) -> String {
        self._id_to_token(id, &self.indices, &self.special_indices, self.unknown_value)
    }
}
