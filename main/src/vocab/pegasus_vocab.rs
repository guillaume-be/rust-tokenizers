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
use crate::vocab::base_vocab::swap_key_values;
use crate::vocab::sentencepiece_proto::sentencepiece_model::ModelProto;
use crate::vocab::Vocab;
use protobuf::Message;
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;

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

    /// The string to use for unknown (out of vocabulary) tokens
    pub unknown_value: &'static str,

    /// A mapping of special value tokens as strings to IDs (i.e. the encoder base for special
    /// values), special values typically include things like BOS/EOS markers, class markers, mask
    /// markers and padding markers
    pub special_values: HashMap<String, i64>,

    /// A mapping of special value tokens as IDs to strings (i.e. the decoder base for special values)
    pub special_indices: HashMap<i64, String>,
}

impl PegasusVocab {
    /// Returns the EOS token for Pegasus (`</s>`)
    pub fn eos_value() -> &'static str {
        "</s>"
    }

    /// Returns the MASK token for Pegasus (`<mask_2>`)
    pub fn mask_value() -> &'static str {
        "<mask_2>"
    }

    /// Returns the MASK token for Pegasus (`<mask_1>`)
    pub fn mask_sent_value() -> &'static str {
        "<mask_1>"
    }

    /// Returns the PAD token for Pegasus (`<pad>`)
    pub fn pad_value() -> &'static str {
        "<pad>"
    }
}

impl PegasusVocab {
    fn _add_and_register_special_value(
        values: &mut HashMap<String, i64>,
        special_values: &mut HashMap<String, i64>,
        value: &str,
        offset: i64,
    ) -> Result<i64, TokenizerError> {
        values.insert(value.to_string(), offset as i64);
        PegasusVocab::_register_as_special_value(value, values, special_values)?;
        Ok(offset + 1)
    }
}

impl Vocab for PegasusVocab {
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

    fn from_file(path: &str) -> Result<PegasusVocab, TokenizerError> {
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
        let mut special_values = HashMap::new();

        // Insert special tokens (not contained in SentencePiece proto)
        let mut offset = 0_i64;

        // pad value
        let pad_value = PegasusVocab::pad_value();
        offset = PegasusVocab::_add_and_register_special_value(
            &mut values,
            &mut special_values,
            pad_value,
            offset,
        )?;

        // EOS value
        let eos_value = PegasusVocab::eos_value();
        offset = PegasusVocab::_add_and_register_special_value(
            &mut values,
            &mut special_values,
            eos_value,
            offset,
        )?;

        // Mask value
        let mask_value = PegasusVocab::mask_value();
        offset = PegasusVocab::_add_and_register_special_value(
            &mut values,
            &mut special_values,
            mask_value,
            offset,
        )?;

        // Sentence mask value
        let mask_sent_value = PegasusVocab::mask_sent_value();
        offset = PegasusVocab::_add_and_register_special_value(
            &mut values,
            &mut special_values,
            mask_sent_value,
            offset,
        )?;

        // Reserved additional special tokens
        for idx in 2..103 {
            let add_unk_token = format!("<unk_{}>", idx);
            offset = PegasusVocab::_add_and_register_special_value(
                &mut values,
                &mut special_values,
                add_unk_token.as_str(),
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

        let unknown_value = PegasusVocab::unknown_value();
        PegasusVocab::_register_as_special_value(unknown_value, &values, &mut special_values)?;

        let indices = swap_key_values(&values);
        let special_indices = swap_key_values(&special_values);

        Ok(PegasusVocab {
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
