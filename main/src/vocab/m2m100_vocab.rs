// Copyright 2021 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
// Copyright 2019-2021 Guillaume Becquin
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
    read_json_file, read_special_token_mapping_file, register_as_special_value, swap_key_values,
    SpecialTokenMap,
};
use crate::vocab::Vocab;
use std::collections::{HashMap, HashSet};
use std::path::Path;

pub static FAIRSEQ_LANGUAGE_CODES: [&str; 100] = [
    "af", "am", "ar", "ast", "az", "ba", "be", "bg", "bn", "br", "bs", "ca", "ceb", "cs", "cy",
    "da", "de", "el", "en", "es", "et", "fa", "ff", "fi", "fr", "fy", "ga", "gd", "gl", "gu", "ha",
    "he", "hi", "hr", "ht", "hu", "hy", "id", "ig", "ilo", "is", "it", "ja", "jv", "ka", "kk",
    "km", "kn", "ko", "lb", "lg", "ln", "lo", "lt", "lv", "mg", "mk", "ml", "mn", "mr", "ms", "my",
    "ne", "nl", "no", "ns", "oc", "or", "pa", "pl", "ps", "pt", "ro", "ru", "sd", "si", "sk", "sl",
    "so", "sq", "sr", "ss", "su", "sv", "sw", "ta", "th", "tl", "tn", "tr", "uk", "ur", "uz", "vi",
    "wo", "xh", "yi", "yo", "zh", "zu",
];

/// # M2M100 Vocab
/// Vocabulary for M2M100 tokenizer. Contains the following special values:
/// - PAD token
/// - BOS token
/// - EOS token
/// - SEP token
///
/// Expects a JSON-format vocabulary when created from file.
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone)]
pub struct M2M100Vocab {
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

    /// Language code stored as bytes for extraction of the prefix in input sequences
    pub language_codes_bytes: HashSet<Vec<u8>>,
}

const DEFAULT_UNK_TOKEN: &str = "<unk>";
const DEFAULT_PAD_TOKEN: &str = "<pad>";
const DEFAULT_BOS_TOKEN: &str = "<s>";
const DEFAULT_SEP_TOKEN: &str = "</s>";
const DEFAULT_EOS_TOKEN: &str = "</s>";

impl M2M100Vocab {
    pub fn get_pad_value(&self) -> &str {
        self.special_token_map
            .cls_token
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

    pub fn get_eos_value(&self) -> &str {
        self.special_token_map
            .eos_token
            .as_deref()
            .unwrap_or(DEFAULT_EOS_TOKEN)
    }
}

impl Vocab for M2M100Vocab {
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

    fn from_file<P: AsRef<Path>>(path: P) -> Result<M2M100Vocab, TokenizerError> {
        let values = read_json_file(path)?;

        let special_token_map = SpecialTokenMap {
            unk_token: DEFAULT_UNK_TOKEN.to_string(),
            pad_token: Some(DEFAULT_PAD_TOKEN.to_string()),
            bos_token: Some(DEFAULT_BOS_TOKEN.to_string()),
            sep_token: Some(DEFAULT_SEP_TOKEN.to_string()),
            cls_token: None,
            eos_token: Some(DEFAULT_EOS_TOKEN.to_string()),
            mask_token: None,
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
        mut values: HashMap<String, i64>,
        special_token_map: SpecialTokenMap,
    ) -> Result<Self, TokenizerError> {
        let mut special_values = HashMap::new();
        for language_code in FAIRSEQ_LANGUAGE_CODES.iter() {
            let language_code = if language_code.len() == 2 {
                format!(">>{language_code}.<<")
            } else if language_code.len() == 3 {
                format!(">>{language_code}<<")
            } else {
                return Err(TokenizerError::VocabularyParsingError(
                    "M2M100 Vocab only supports language code of length 2 or 3".to_string(),
                ));
            };
            values.insert(language_code.clone(), values.len() as i64);
            register_as_special_value(language_code.as_str(), &values, &mut special_values)?;
        }

        let language_codes_bytes = FAIRSEQ_LANGUAGE_CODES
            .iter()
            .map(|f| {
                if f.len() == 2 {
                    format!(">>{f}.<<")
                } else {
                    format!(">>{f}<<")
                }
                .as_bytes()
                .to_vec()
            })
            .collect::<HashSet<Vec<u8>>>();

        special_token_map.register_special_values(&values, &mut special_values)?;

        let indices = swap_key_values(&values);
        let special_indices = swap_key_values(&special_values);

        Ok(Self {
            values,
            indices,
            special_token_map,
            special_values,
            special_indices,
            language_codes_bytes,
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
