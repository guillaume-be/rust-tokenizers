// Copyright 2020 Google and The HuggingFace Inc. team.
// Copyright 2021 Guillaume Becquin
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
    open_protobuf_file, read_special_token_mapping_file, register_as_special_value,
    swap_key_values, SpecialTokenMap,
};
use crate::vocab::Vocab;
use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::path::Path;

/// # Pegasus Vocab
/// Vocabulary for Pegasus tokenizer. Contains the following special values:
/// - PAD token
/// - EOS token
/// - MASK token
/// - MASK_SENT token
///
/// Expects a SentencePiece protobuf file when created from file.
#[derive(Debug, Clone)]
pub struct PegasusVocab {
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

const DEFAULT_UNK_TOKEN: &str = "<unk>";
const DEFAULT_PAD_TOKEN: &str = "<pad>";
const DEFAULT_EOS_TOKEN: &str = "</s>";
const DEFAULT_MASK_TOKEN: &str = "<mask_2>";
const DEFAULT_SENTENCE_MASK_TOKEN: &str = "<mask_1>";

impl PegasusVocab {
    pub fn get_pad_value(&self) -> &str {
        self.special_token_map
            .pad_token
            .as_deref()
            .unwrap_or(DEFAULT_PAD_TOKEN)
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

    fn _add_and_register_special_value(
        values: &mut HashMap<String, i64>,
        special_values: &mut HashMap<String, i64>,
        value: &str,
        offset: i64,
    ) -> Result<i64, TokenizerError> {
        values.insert(value.to_string(), offset as i64);
        register_as_special_value(value, values, special_values)?;
        Ok(offset + 1)
    }
}

impl Vocab for PegasusVocab {
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

    fn from_file(path: &Path) -> Result<PegasusVocab, TokenizerError> {
        let proto = open_protobuf_file(path)?;

        let mut values = HashMap::new();
        let mut special_values = HashMap::new();

        let mut additional_special_tokens = HashSet::from([DEFAULT_SENTENCE_MASK_TOKEN.into()]);
        for idx in 2..103 {
            let _ = additional_special_tokens.insert(format!("<unk_{}>", idx));
        }

        let special_token_map = SpecialTokenMap {
            unk_token: DEFAULT_UNK_TOKEN.to_string(),
            pad_token: Some(DEFAULT_PAD_TOKEN.to_string()),
            bos_token: None,
            sep_token: None,
            cls_token: None,
            eos_token: Some(DEFAULT_EOS_TOKEN.to_string()),
            mask_token: Some(DEFAULT_MASK_TOKEN.to_string()),
            additional_special_tokens: Some(additional_special_tokens),
        };

        // Insert special tokens (not contained in SentencePiece proto)
        let mut offset = 0_i64;

        // pad value
        let pad_value = special_token_map.pad_token.as_ref().unwrap();
        offset = PegasusVocab::_add_and_register_special_value(
            &mut values,
            &mut special_values,
            pad_value,
            offset,
        )?;

        // EOS value
        let eos_value = special_token_map.eos_token.as_ref().unwrap();
        offset = PegasusVocab::_add_and_register_special_value(
            &mut values,
            &mut special_values,
            eos_value,
            offset,
        )?;

        // Mask value
        let mask_value = special_token_map.mask_token.as_ref().unwrap();
        offset = PegasusVocab::_add_and_register_special_value(
            &mut values,
            &mut special_values,
            mask_value,
            offset,
        )?;

        // Sentence mask value & additional tokens
        for additional_token in special_token_map
            .additional_special_tokens
            .as_ref()
            .unwrap()
        {
            offset = PegasusVocab::_add_and_register_special_value(
                &mut values,
                &mut special_values,
                additional_token,
                offset,
            )?;
        }

        let mut current_piece: String;
        let mut idx = 0;
        for piece in proto.get_pieces().iter() {
            current_piece = piece.get_piece().to_owned();
            match values.entry(current_piece) {
                Entry::Vacant(v) => {
                    v.insert(idx as i64 + offset);
                    idx += 1;
                }
                Entry::Occupied(_) => {}
            };
        }

        register_as_special_value(&special_token_map.unk_token, &values, &mut special_values)?;

        let indices = swap_key_values(&values);
        let special_indices = swap_key_values(&special_values);

        Ok(PegasusVocab {
            values,
            indices,
            special_token_map,
            special_values,
            special_indices,
        })
    }

    fn from_file_with_special_token_mapping(
        path: &Path,
        special_token_mapping_path: &Path,
    ) -> Result<Self, TokenizerError> {
        let proto = open_protobuf_file(path)?;

        let mut values = HashMap::new();
        let mut special_values = HashMap::new();

        let mut additional_special_tokens = HashSet::from(["<mask_1>".into()]);
        for idx in 2..103 {
            let _ = additional_special_tokens.insert(format!("<unk_{}>", idx));
        }

        let special_token_map = read_special_token_mapping_file(special_token_mapping_path)?;

        // Insert special tokens (not contained in SentencePiece proto)
        let mut offset = 0_i64;

        // pad value
        if let Some(pad_value) = &special_token_map.pad_token {
            offset = PegasusVocab::_add_and_register_special_value(
                &mut values,
                &mut special_values,
                pad_value,
                offset,
            )?;
        }

        // EOS value
        if let Some(eos_value) = &special_token_map.eos_token {
            offset = PegasusVocab::_add_and_register_special_value(
                &mut values,
                &mut special_values,
                eos_value,
                offset,
            )?;
        }

        // Mask value
        if let Some(mask_value) = &special_token_map.mask_token {
            offset = PegasusVocab::_add_and_register_special_value(
                &mut values,
                &mut special_values,
                mask_value,
                offset,
            )?;
        }

        // Sentence mask value & additional tokens
        if let Some(additional_tokens) = &special_token_map.additional_special_tokens {
            for additional_token in additional_tokens {
                offset = PegasusVocab::_add_and_register_special_value(
                    &mut values,
                    &mut special_values,
                    additional_token,
                    offset,
                )?;
            }
        }

        let mut current_piece: String;
        let mut idx = 0;
        for piece in proto.get_pieces().iter() {
            current_piece = piece.get_piece().to_owned();
            match values.entry(current_piece) {
                Entry::Vacant(v) => {
                    v.insert(idx as i64 + offset);
                    idx += 1;
                }
                Entry::Occupied(_) => {}
            };
        }

        register_as_special_value(&special_token_map.unk_token, &values, &mut special_values)?;

        let indices = swap_key_values(&values);
        let special_indices = swap_key_values(&special_values);

        Ok(PegasusVocab {
            values,
            indices,
            special_token_map,
            special_values,
            special_indices,
        })
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
