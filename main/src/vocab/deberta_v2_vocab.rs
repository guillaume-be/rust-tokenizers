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
use crate::vocab::base_vocab::swap_key_values;
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

    /// A mapping of special value tokens as strings to IDs (i.e. the encoder base for special
    /// values), special values typically include things like BOS/EOS markers, class markers, mask
    /// markers and padding markers
    pub special_values: HashMap<String, i64>,

    /// A mapping of special value tokens as IDs to strings (i.e. the decoder base for special values)
    pub special_indices: HashMap<i64, String>,

    /// A mapping of special value tokens
    pub special_tokens_map: DeBERTaV2SpecialTokensMap,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct DeBERTaV2SpecialTokensMap {
    pub bos_token: String,
    pub eos_token: String,
    pub unk_token: String,
    pub sep_token: String,
    pub pad_token: String,
    pub cls_token: String,
    pub mask_token: String,
}

impl DeBERTaV2SpecialTokensMap {
    pub fn from_file(path: &str) -> Result<Self, TokenizerError> {
        let f = File::open(path).map_err(|e| {
            TokenizerError::FileNotFound(format!(
                "{} special token map file not found :{}",
                path, e
            ))
        })?;

        serde_json::from_reader(f)
            .map_err(|e| TokenizerError::VocabularyParsingError(e.to_string()))
    }
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

impl DeBERTaV2Vocab {
    pub fn from_file_with_secial_tokens_map(
        path: &str,
        special_tokens_map: DeBERTaV2SpecialTokensMap,
    ) -> Result<DeBERTaV2Vocab, TokenizerError> {
        let mut f = File::open(path).map_err(|e| {
            TokenizerError::FileNotFound(format!("{} vocabulary file not found :{}", path, e))
        })?;
        let mut contents = Vec::new();
        let proto = match f.read_to_end(&mut contents) {
            Ok(_) => match ModelProto::parse_from_bytes(contents.as_slice()) {
                Ok(proto_value) => proto_value,
                Err(e) => {
                    return Err(TokenizerError::VocabularyParsingError(e.to_string()));
                }
            },
            Err(e) => {
                return Err(TokenizerError::VocabularyParsingError(e.to_string()));
            }
        };

        let mut values = HashMap::new();
        for (idx, piece) in proto.get_pieces().iter().enumerate() {
            values.insert(piece.get_piece().to_owned(), idx as i64);
        }
        if !values.contains_key(&special_tokens_map.mask_token) {
            values.insert(special_tokens_map.mask_token.clone(), values.len() as i64);
        }

        let mut special_values = HashMap::new();
        DeBERTaV2Vocab::_register_as_special_value(
            &special_tokens_map.unk_token,
            &values,
            &mut special_values,
        )?;

        DeBERTaV2Vocab::_register_as_special_value(
            &special_tokens_map.bos_token,
            &values,
            &mut special_values,
        )?;

        DeBERTaV2Vocab::_register_as_special_value(
            &special_tokens_map.eos_token,
            &values,
            &mut special_values,
        )?;

        DeBERTaV2Vocab::_register_as_special_value(
            &special_tokens_map.cls_token,
            &values,
            &mut special_values,
        )?;

        DeBERTaV2Vocab::_register_as_special_value(
            &special_tokens_map.mask_token,
            &values,
            &mut special_values,
        )?;

        DeBERTaV2Vocab::_register_as_special_value(
            &special_tokens_map.pad_token,
            &values,
            &mut special_values,
        )?;

        DeBERTaV2Vocab::_register_as_special_value(
            &special_tokens_map.sep_token,
            &values,
            &mut special_values,
        )?;

        let indices = swap_key_values(&values);
        let special_indices = swap_key_values(&special_values);

        Ok(DeBERTaV2Vocab {
            values,
            indices,
            special_values,
            special_indices,
            special_tokens_map,
        })
    }
}

impl Vocab for DeBERTaV2Vocab {
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

    fn from_file(path: &str) -> Result<DeBERTaV2Vocab, TokenizerError> {
        Self::from_file_with_secial_tokens_map(path, Default::default())
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
