// Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
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
use crate::vocab::base_vocab::{
    read_protobuf_file, read_special_token_mapping_file, swap_key_values, SpecialTokenMap,
};
use crate::vocab::Vocab;
use std::collections::{HashMap, HashSet};

/// # XLNet Vocab
/// Vocabulary for XLNet tokenizer. Contains the following special values:
/// - PAD token
/// - BOS token
/// - EOS token
/// - SEP token
/// - MASK token
/// - CLS token
/// - EOP token
/// - EOD token
///
/// Expects a SentencePiece protobuf file when created from file.
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone)]
pub struct XLNetVocab {
    /// A mapping of tokens as string to indices (i.e. the encoder base)
    pub values: HashMap<String, i64>,

    /// A mapping of token IDs to strings (i.e. the decoder base)
    pub indices: HashMap<i64, String>,

    /// Special tokens used by the vocabulary
    pub special_token_map: SpecialTokenMap,

    /// A mapping of special value tokens as strings to IDs (i.e. the encoder base for special
    /// values), special values typically include things like BOS/EOS markers, class markers, mask
    /// markers and padding markers
    pub special_values: HashMap<String, i64>,

    /// A mapping of special value tokens as IDs to strings (i.e. the decoder base for special values)
    pub special_indices: HashMap<i64, String>,
}

impl XLNetVocab {
    /// Returns the EOP token for XLNet (`<eop>`)
    pub fn eop_value() -> &'static str {
        "<eop>"
    }

    /// Returns the EOD token for XLNet (`<eod>`)
    pub fn eod_value() -> &'static str {
        "<eod>"
    }
}

impl Vocab for XLNetVocab {
    fn get_unknown_value(&self) -> &str {
        &self.special_token_map.unk_token
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

    fn from_file(path: &str) -> Result<XLNetVocab, TokenizerError> {
        let values = read_protobuf_file(path)?;

        let special_token_map = SpecialTokenMap {
            unk_token: "<unk>".to_string(),
            pad_token: Some("<pad>".to_string()),
            bos_token: Some("<s>".to_string()),
            sep_token: Some("<sep>".to_string()),
            cls_token: Some("<cls>".to_string()),
            eos_token: Some("</s>".to_string()),
            mask_token: Some("<mask>".to_string()),
            additional_special_tokens: Some(HashSet::from([
                "<eop>".to_string(),
                "<eod>".to_string(),
            ])),
        };
        Self::from_values_and_special_token_map(values, special_token_map)
    }

    fn from_file_with_special_token_mapping(
        path: &str,
        special_token_mapping_path: &str,
    ) -> Result<Self, TokenizerError> {
        let values = read_protobuf_file(path)?;
        let special_token_map = read_special_token_mapping_file(special_token_mapping_path)?;
        Self::from_values_and_special_token_map(values, special_token_map)
    }

    fn from_values_and_special_token_map(
        values: HashMap<String, i64>,
        special_token_map: SpecialTokenMap,
    ) -> Result<Self, TokenizerError>
    where
        Self: std::marker::Sized,
    {
        let mut special_values = HashMap::new();
        special_token_map.register_special_values(&values, &mut special_values)?;

        let indices = swap_key_values(&values);
        let special_indices = swap_key_values(&special_values);
        Ok(Self {
            values,
            indices,
            special_token_map,
            special_values,
            special_indices,
        })
    }

    fn token_to_id(&self, token: &str) -> i64 {
        self._token_to_id(
            token,
            &self.values,
            &self.special_values,
            self.get_unknown_value(),
        )
    }

    fn id_to_token(&self, id: &i64) -> String {
        self._id_to_token(
            id,
            &self.indices,
            &self.special_indices,
            self.get_unknown_value(),
        )
    }
}
