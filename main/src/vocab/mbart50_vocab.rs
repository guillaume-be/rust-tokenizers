// Copyright 2021 The Facebook AI Research Team Authors and The HuggingFace Inc. team.
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
    open_protobuf_file, read_special_token_mapping_file, register_as_special_value,
    swap_key_values, SpecialTokenMap,
};
use crate::vocab::Vocab;
use std::collections::{HashMap, HashSet};

pub static FAIRSEQ_LANGUAGE_CODES: [&str; 52] = [
    ">>ar<<", ">>cs<<", ">>de<<", ">>en<<", ">>es<<", ">>et<<", ">>fi<<", ">>fr<<", ">>gu<<",
    ">>hi<<", ">>it<<", ">>ja<<", ">>kk<<", ">>ko<<", ">>lt<<", ">>lv<<", ">>my<<", ">>ne<<",
    ">>nl<<", ">>ro<<", ">>ru<<", ">>si<<", ">>tr<<", ">>vi<<", ">>zh<<", ">>af<<", ">>az<<",
    ">>bn<<", ">>fa<<", ">>he<<", ">>hr<<", ">>id<<", ">>ka<<", ">>km<<", ">>mk<<", ">>ml<<",
    ">>mn<<", ">>mr<<", ">>pl<<", ">>ps<<", ">>pt<<", ">>sv<<", ">>sw<<", ">>ta<<", ">>te<<",
    ">>th<<", ">>tl<<", ">>uk<<", ">>ur<<", ">>xh<<", ">>gl<<", ">>sl<<",
];

/// # MBart50 Vocab
/// Vocabulary for MBart50 tokenizer. Contains the following special values:
/// - PAD token
/// - EOS token
/// - SEP token
/// - MASK token
/// - CLS token
///
/// Expects a SentencePiece protobuf file when created from file.
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone)]
pub struct MBart50Vocab {
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

impl Vocab for MBart50Vocab {
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

    fn from_file(path: &str) -> Result<MBart50Vocab, TokenizerError> {
        let mut values = HashMap::new();
        let mut special_values = HashMap::new();

        let special_token_map = SpecialTokenMap {
            unk_token: "<unk>".to_string(),
            pad_token: Some("<pad>".to_string()),
            bos_token: None,
            sep_token: Some("</s>".to_string()),
            cls_token: Some("<s>".to_string()),
            eos_token: Some("</s>".to_string()),
            mask_token: Some("<mask>".to_string()),
            additional_special_tokens: None,
        };

        values.insert(
            special_token_map.cls_token.as_ref().unwrap().clone(),
            values.len() as i64,
        );
        values.insert(
            special_token_map.pad_token.as_ref().unwrap().clone(),
            values.len() as i64,
        );
        values.insert(
            special_token_map.eos_token.as_ref().unwrap().clone(),
            values.len() as i64,
        );
        values.insert(special_token_map.unk_token.clone(), values.len() as i64);

        let proto = open_protobuf_file(path)?;
        for piece in proto.get_pieces().iter().skip(3) {
            values.insert(piece.get_piece().to_owned(), values.len() as i64);
        }

        for language_code in FAIRSEQ_LANGUAGE_CODES.iter() {
            values.insert(language_code.to_string(), values.len() as i64);
            register_as_special_value(language_code, &values, &mut special_values)?;
        }

        values.insert(
            special_token_map.mask_token.as_ref().unwrap().to_owned(),
            values.len() as i64,
        );

        special_token_map.register_special_values(&values, &mut special_values);

        let indices = swap_key_values(&values);
        let special_indices = swap_key_values(&special_values);
        let language_codes_bytes = FAIRSEQ_LANGUAGE_CODES
            .iter()
            .map(|f| f.as_bytes().to_vec())
            .collect::<HashSet<Vec<u8>>>();

        Ok(MBart50Vocab {
            values,
            indices,
            special_token_map,
            special_values,
            special_indices,
            language_codes_bytes,
        })
    }

    fn from_file_with_special_token_mapping(
        path: &str,
        special_token_mapping_path: &str,
    ) -> Result<Self, TokenizerError> {
        let mut values = HashMap::new();
        let mut special_values = HashMap::new();

        let special_token_map = read_special_token_mapping_file(special_token_mapping_path)?;

        if let Some(cls_token) = special_token_map.cls_token.as_ref() {
            values.insert(cls_token.clone(), values.len() as i64);
        }

        if let Some(pad_token) = special_token_map.pad_token.as_ref() {
            values.insert(pad_token.clone(), values.len() as i64);
        }

        if let Some(eos_token) = special_token_map.eos_token.as_ref() {
            values.insert(eos_token.clone(), values.len() as i64);
        }

        values.insert(special_token_map.unk_token.clone(), values.len() as i64);

        let proto = open_protobuf_file(path)?;
        for piece in proto.get_pieces().iter().skip(3) {
            values.insert(piece.get_piece().to_owned(), values.len() as i64);
        }

        for language_code in FAIRSEQ_LANGUAGE_CODES.iter() {
            values.insert(language_code.to_string(), values.len() as i64);
            register_as_special_value(language_code, &values, &mut special_values)?;
        }

        values.insert(
            special_token_map.mask_token.as_ref().unwrap().to_owned(),
            values.len() as i64,
        );

        special_token_map.register_special_values(&values, &mut special_values);

        let indices = swap_key_values(&values);
        let special_indices = swap_key_values(&special_values);
        let language_codes_bytes = FAIRSEQ_LANGUAGE_CODES
            .iter()
            .map(|f| f.as_bytes().to_vec())
            .collect::<HashSet<Vec<u8>>>();

        Ok(MBart50Vocab {
            values,
            indices,
            special_token_map,
            special_values,
            special_indices,
            language_codes_bytes,
        })
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

        let language_codes_bytes = FAIRSEQ_LANGUAGE_CODES
            .iter()
            .map(|f| f.as_bytes().to_vec())
            .collect::<HashSet<Vec<u8>>>();

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
