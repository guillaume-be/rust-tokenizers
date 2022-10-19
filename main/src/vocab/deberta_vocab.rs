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
use crate::vocab::base_vocab::{
    read_json_file, read_special_token_mapping_file, swap_key_values, SpecialTokenMap, Vocab,
};
use std::collections::HashMap;
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

    /// Special tokens used by the vocabulary
    pub special_token_map: SpecialTokenMap,

    /// A mapping of special value tokens as strings to IDs (i.e. the encoder base for special
    /// values), special values typically include things like BOS/EOS markers, class markers, mask
    /// markers and padding markers
    pub special_values: HashMap<String, i64>,

    /// A mapping of special value tokens as IDs to strings (i.e. the decoder base for special values)
    pub special_indices: HashMap<i64, String>,
}

const DEFAULT_UNK_TOKEN: &str = "[UNK]";
const DEFAULT_PAD_TOKEN: &str = "[PAD]";
const DEFAULT_BOS_TOKEN: &str = "[CLS]";
const DEFAULT_SEP_TOKEN: &str = "[SEP]";
const DEFAULT_CLS_TOKEN: &str = "[CLS]";
const DEFAULT_EOS_TOKEN: &str = "[SEP]";
const DEFAULT_MASK_TOKEN: &str = "[MASK]";

impl DeBERTaVocab {
    pub fn get_pad_value(&self) -> &str {
        self.special_token_map
            .pad_token
            .as_deref()
            .unwrap_or(DEFAULT_PAD_TOKEN)
    }

    pub fn get_bos_value(&self) -> &str {
        self.special_token_map
            .bos_token
            .as_deref()
            .unwrap_or(DEFAULT_BOS_TOKEN)
    }

    pub fn get_sep_value(&self) -> &str {
        self.special_token_map
            .sep_token
            .as_deref()
            .unwrap_or(DEFAULT_SEP_TOKEN)
    }

    pub fn get_cls_value(&self) -> &str {
        self.special_token_map
            .cls_token
            .as_deref()
            .unwrap_or(DEFAULT_CLS_TOKEN)
    }

    pub fn get_eos_value(&self) -> &str {
        self.special_token_map
            .eos_token
            .as_deref()
            .unwrap_or(DEFAULT_EOS_TOKEN)
    }

    pub fn get_mask_value(&self) -> &str {
        self.special_token_map
            .mask_token
            .as_deref()
            .unwrap_or(DEFAULT_MASK_TOKEN)
    }
}

impl Vocab for DeBERTaVocab {
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

    fn from_file<P: AsRef<Path>>(path: P) -> Result<DeBERTaVocab, TokenizerError> {
        let values = read_json_file(path)?;

        let special_token_map = SpecialTokenMap {
            unk_token: DEFAULT_UNK_TOKEN.to_string(),
            pad_token: Some(DEFAULT_PAD_TOKEN.to_string()),
            bos_token: Some(DEFAULT_BOS_TOKEN.to_string()),
            sep_token: Some(DEFAULT_SEP_TOKEN.to_string()),
            cls_token: Some(DEFAULT_CLS_TOKEN.to_string()),
            eos_token: Some(DEFAULT_EOS_TOKEN.to_string()),
            mask_token: Some(DEFAULT_MASK_TOKEN.to_string()),
            additional_special_tokens: None,
        };
        Self::from_values_and_special_token_map(values, special_token_map)
    }

    fn from_file_with_special_token_mapping<P: AsRef<Path>, S: AsRef<Path>>(
        path: P,
        special_token_mapping_path: S,
    ) -> Result<Self, TokenizerError> {
        let values = read_json_file(path)?;
        let special_token_map = read_special_token_mapping_file(special_token_mapping_path)?;
        Self::from_values_and_special_token_map(values, special_token_map)
    }
    fn from_values_and_special_token_map(
        values: HashMap<String, i64>,
        special_token_map: SpecialTokenMap,
    ) -> Result<Self, TokenizerError>
    where
        Self: Sized,
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
