// Copyright 2020 Microsoft and the HuggingFace Inc. team.
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
use crate::vocab::sentencepiece_proto::sentencepiece_model::ModelProto;
use crate::vocab::Vocab;
use protobuf::Message;
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;

/// # DeBERTaV2Vocab
/// Vocabulary for DeBERTa (v2) tokenizer. Contains the following special values:
/// - BOS token
/// - EOS token
/// - CLS token
/// - SEP token
/// - UNK token
/// - PAD token
/// - MASK token
///
/// Expects a SentencePiece protobuf file when created from file.
#[derive(Debug, Clone)]
pub struct DeBERTaV2Vocab {
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

impl Default for DeBERTaV2SpecialTokensMap {
    fn default() -> Self {
        Self {
            bos_token: "[CLS]".into(),
            eos_token: "[SEP]".into(),
            unk_token: "[UNK]".into(),
            sep_token: "[SEP]".into(),
            pad_token: "[PAD]".into(),
            cls_token: "[CLS]".into(),
            mask_token: "[MASK]".into(),
        }
    }
}

impl Vocab for DeBERTaV2Vocab {
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

    fn from_file(path: &str) -> Result<DeBERTaV2Vocab, TokenizerError> {
        let mut values = read_protobuf_file(path)?;

        let special_token_map = SpecialTokenMap {
            unk_token: "[UNK]".to_string(),
            pad_token: Some("[PAD]".to_string()),
            bos_token: Some("[CLS]".to_string()),
            sep_token: Some("[SEP]".to_string()),
            cls_token: Some("[CLS]".to_string()),
            eos_token: Some("[SEP]".to_string()),
            mask_token: Some("[MASK]".to_string()),
            additional_special_tokens: None,
        };

        if !values.contains_key(special_token_map.mask_token.as_ref().unwrap()) {
            values.insert(
                special_token_map.mask_token.as_ref().unwrap().clone(),
                values.len() as i64,
            );
        }

        Self::from_values_and_special_token_map(values, special_token_map)
    }

    fn from_file_with_special_token_mapping(
        path: &str,
        special_token_mapping_path: &str,
    ) -> Result<Self, TokenizerError> {
        let mut values = read_protobuf_file(path)?;
        let special_token_map = read_special_token_mapping_file(special_token_mapping_path)?;

        if let Some(mask_token) = &special_token_map.mask_token {
            values.insert(mask_token.clone(), values.len() as i64);
        }

        Self::from_values_and_special_token_map(values, special_token_map)
    }

    fn token_to_id(&self, token: &str) -> i64 {
        self._token_to_id(
            token,
            &self.values,
            &self.special_values,
            &self.special_tokens_map.unk_token,
        )
    }

    fn id_to_token(&self, id: &i64) -> String {
        self._id_to_token(
            id,
            &self.indices,
            &self.special_indices,
            &self.special_tokens_map.unk_token,
        )
    }
}
