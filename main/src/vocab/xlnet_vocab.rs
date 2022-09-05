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
use crate::vocab::base_vocab::swap_key_values;
use crate::vocab::sentencepiece_proto::sentencepiece_model::ModelProto;
use crate::vocab::Vocab;
use protobuf::Message;
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

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

    /// The string to use for unknown (out of vocabulary) tokens
    pub unknown_value: &'static str,

    /// A mapping of special value tokens as strings to IDs (i.e. the encoder base for special
    /// values), special values typically include things like BOS/EOS markers, class markers, mask
    /// markers and padding markers
    pub special_values: HashMap<String, i64>,

    /// A mapping of special value tokens as IDs to strings (i.e. the decoder base for special values)
    pub special_indices: HashMap<i64, String>,
}

impl XLNetVocab {
    /// Returns the BOS token for XLNet (`<s>`)
    pub fn bos_value() -> &'static str {
        "<s>"
    }

    /// Returns the EOS token for XLNet (`</s>`)
    pub fn eos_value() -> &'static str {
        "</s>"
    }

    /// Returns the SEP token for XLNet (`<sep>`)
    pub fn sep_value() -> &'static str {
        "<sep>"
    }

    /// Returns the CLS token for XLNet (`<cls>`)
    pub fn cls_value() -> &'static str {
        "<cls>"
    }

    /// Returns the MASK token for XLNet (`<mask>`)
    pub fn mask_value() -> &'static str {
        "<mask>"
    }

    /// Returns the PAD token for XLNet (`<pad>`)
    pub fn pad_value() -> &'static str {
        "<pad>"
    }

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

    fn from_file<V: AsRef<Path>, S: AsRef<Path>>(
        vocab: V,
        _special: Option<S>,
    ) -> Result<XLNetVocab, TokenizerError> {
        let mut f = File::open(&vocab).map_err(|e| {
            TokenizerError::FileNotFound(format!(
                "{} vocabulary file not found :{}",
                vocab.as_ref().display(),
                e
            ))
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

        let mut special_values = HashMap::new();
        let unknown_value = XLNetVocab::unknown_value();
        XLNetVocab::_register_as_special_value(unknown_value, &values, &mut special_values)?;

        let bos_value = XLNetVocab::bos_value();
        XLNetVocab::_register_as_special_value(bos_value, &values, &mut special_values)?;

        let eos_value = XLNetVocab::eos_value();
        XLNetVocab::_register_as_special_value(eos_value, &values, &mut special_values)?;

        let cls_value = XLNetVocab::cls_value();
        XLNetVocab::_register_as_special_value(cls_value, &values, &mut special_values)?;

        let mask_value = XLNetVocab::mask_value();
        XLNetVocab::_register_as_special_value(mask_value, &values, &mut special_values)?;

        let pad_value = XLNetVocab::pad_value();
        XLNetVocab::_register_as_special_value(pad_value, &values, &mut special_values)?;

        let sep_value = XLNetVocab::sep_value();
        XLNetVocab::_register_as_special_value(sep_value, &values, &mut special_values)?;

        let eop_value = XLNetVocab::eop_value();
        XLNetVocab::_register_as_special_value(eop_value, &values, &mut special_values)?;

        let eod_value = XLNetVocab::eod_value();
        XLNetVocab::_register_as_special_value(eod_value, &values, &mut special_values)?;

        let indices = swap_key_values(&values);
        let special_indices = swap_key_values(&special_values);

        Ok(XLNetVocab {
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
