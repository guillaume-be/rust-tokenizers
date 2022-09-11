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
use crate::vocab::base_vocab::{
    read_protobuf_file, read_special_token_mapping_file, swap_key_values, SpecialTokenMap,
};
use crate::vocab::sentencepiece_proto::sentencepiece_model::ModelProto;
use crate::vocab::Vocab;
use protobuf::Message;
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;

/// # SentencePieceVocab
/// Vocabulary for SentencePiece model/tokenizer. Contains the following special values:
/// - BOS token
/// - EOS token
/// - CLS token
/// - SEP token
/// - PAD token
/// - MASK token
///
/// Expects a SentencePiece protobuf file when created from file.
#[derive(Debug, Clone)]
pub struct SentencePieceVocab {
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

impl SentencePieceVocab {
    /// Returns the PAD token for SentencePiece (`<pad>`)
    pub fn pad_value() -> &'static str {
        "<pad>"
    }

    /// Returns the SEP token for SentencePiece (`<sep>`)
    pub fn sep_value() -> &'static str {
        "<sep>"
    }

    /// Returns the CLS token for SentencePiece (`<cls>`)
    pub fn cls_value() -> &'static str {
        "<cls>"
    }

    /// Returns the MASK token for SentencePiece (`<mask>`)
    pub fn mask_value() -> &'static str {
        "<mask>"
    }

    /// Returns the BOS token for SentencePiece (`<s>`)
    pub fn bos_value() -> &'static str {
        "<s>"
    }

    /// Returns the EOS token for SentencePiece (`</s>`)
    pub fn eos_value() -> &'static str {
        "</s>"
    }
}

impl Vocab for SentencePieceVocab {
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

    fn from_file(path: &str) -> Result<SentencePieceVocab, TokenizerError> {
        let mut values = read_protobuf_file(path)?;

        let special_token_map = SpecialTokenMap {
            unk_token: "<unk>".to_string(),
            pad_token: None,
            bos_token: None,
            sep_token: None,
            cls_token: None,
            eos_token: None,
            mask_token: None,
            additional_special_tokens: None,
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
