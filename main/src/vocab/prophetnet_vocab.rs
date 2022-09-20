// Copyright 2020 The Microsoft Authors and The HuggingFace Inc. team.
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
    read_flat_file, read_special_token_mapping_file, swap_key_values, SpecialTokenMap, Vocab,
};
use std::collections::{HashMap, HashSet};

/// # ProphetNet Vocab
/// Vocabulary for ProphetNet tokenizer. Contains the following special values:
/// - SEP token
/// - CLS token
/// - X_SEP token
/// - PAD token
/// - MASK token
///
/// Expects a flat text vocabulary when created from file.
#[derive(Debug, Clone)]
pub struct ProphetNetVocab {
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
const DEFAULT_SEP_TOKEN: &str = "[SEP]";
const DEFAULT_X_SEP_TOKEN: &str = "[X_SEP]";
const DEFAULT_CLS_TOKEN: &str = "[CLS]";
const DEFAULT_MASK_TOKEN: &str = "[MASK]";

impl ProphetNetVocab {
    pub fn get_pad_value(&self) -> &str {
        self.special_token_map
            .pad_token
            .as_deref()
            .unwrap_or(DEFAULT_PAD_TOKEN)
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

    pub fn get_mask_value(&self) -> &str {
        self.special_token_map
            .mask_token
            .as_deref()
            .unwrap_or(DEFAULT_MASK_TOKEN)
    }
}

impl Vocab for ProphetNetVocab {
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

    fn from_file(path: &str) -> Result<ProphetNetVocab, TokenizerError> {
        let values = read_flat_file(path)?;

        let special_token_map = SpecialTokenMap {
            unk_token: DEFAULT_UNK_TOKEN.to_string(),
            pad_token: Some(DEFAULT_PAD_TOKEN.to_string()),
            bos_token: None,
            sep_token: Some(DEFAULT_SEP_TOKEN.to_string()),
            cls_token: Some(DEFAULT_CLS_TOKEN.to_string()),
            eos_token: None,
            mask_token: Some(DEFAULT_MASK_TOKEN.to_string()),
            additional_special_tokens: Some(HashSet::from([DEFAULT_X_SEP_TOKEN.into()])),
        };

        Self::from_values_and_special_token_map(values, special_token_map)
    }

    fn from_file_with_special_token_mapping(
        path: &str,
        special_token_mapping_path: &str,
    ) -> Result<Self, TokenizerError> {
        let values = read_flat_file(path)?;
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

//==============================
// Unit tests
//==============================
#[cfg(test)]
mod tests {
    use super::*;
    extern crate anyhow;
    use std::io::Write;

    #[test]
    fn test_create_object() {
        //        Given
        let values: HashMap<String, i64> = HashMap::new();
        let special_values: HashMap<String, i64> = HashMap::new();
        let indices: HashMap<i64, String> = HashMap::new();
        let special_indices: HashMap<i64, String> = HashMap::new();
        let special_token_map = SpecialTokenMap {
            unk_token: "[UNK]".to_string(),
            pad_token: Some("[PAD]".to_string()),
            bos_token: None,
            sep_token: Some("[SEP]".to_string()),
            cls_token: Some("[CLS]".to_string()),
            eos_token: None,
            mask_token: Some("[MASK]".to_string()),
            additional_special_tokens: Some(HashSet::from(["[X_SEP".into()])),
        };

        //        When
        let base_vocab = ProphetNetVocab {
            values,
            indices,
            special_token_map,
            special_values,
            special_indices,
        };

        //        Then
        assert_eq!(base_vocab.get_unknown_value(), "[UNK]");
        assert_eq!(base_vocab.values, *base_vocab.values());
        assert_eq!(base_vocab.special_values, *base_vocab.special_values());
    }

    #[test]
    fn test_create_object_from_file() -> anyhow::Result<()> {
        //        Given
        let mut vocab_file = tempfile::NamedTempFile::new()?;
        write!(
            vocab_file,
            "hello \n world \n [UNK] \n ! \n [X_SEP] \n [SEP] \n [MASK] \n [PAD] \n [CLS]"
        )?;
        let path = vocab_file.into_temp_path();
        let target_values: HashMap<String, i64> = [
            ("hello".to_owned(), 0),
            ("world".to_owned(), 1),
            ("[UNK]".to_owned(), 2),
            ("!".to_owned(), 3),
            ("[X_SEP]".to_owned(), 4),
            ("[SEP]".to_owned(), 5),
            ("[MASK]".to_owned(), 6),
            ("[PAD]".to_owned(), 7),
            ("[CLS]".to_owned(), 8),
        ]
        .iter()
        .cloned()
        .collect();

        let special_values: HashMap<String, i64> = [
            ("[UNK]".to_owned(), 2),
            ("[X_SEP]".to_owned(), 4),
            ("[SEP]".to_owned(), 5),
            ("[MASK]".to_owned(), 6),
            ("[PAD]".to_owned(), 7),
            ("[CLS]".to_owned(), 8),
        ]
        .iter()
        .cloned()
        .collect();

        //        When
        let base_vocab = ProphetNetVocab::from_file(path.to_path_buf().to_str().unwrap())?;

        //        Then
        assert_eq!(base_vocab.get_unknown_value(), "[UNK]");
        assert_eq!(base_vocab.values, target_values);
        assert_eq!(base_vocab.special_values, special_values);
        drop(path);
        Ok(())
    }

    #[test]
    #[should_panic]
    fn test_create_object_from_file_without_unknown_token() {
        //        Given
        let mut vocab_file = tempfile::NamedTempFile::new().unwrap();
        write!(vocab_file, "hello \n world \n [X_SEP] \n ! \n [SEP]").unwrap();
        let path = vocab_file.into_temp_path();

        //        When & Then
        let _base_vocab = ProphetNetVocab::from_file(path.to_path_buf().to_str().unwrap()).unwrap();
    }

    #[test]
    fn test_encode_tokens() -> anyhow::Result<()> {
        //        Given
        let mut vocab_file = tempfile::NamedTempFile::new()?;
        write!(
            vocab_file,
            "hello \n world \n [UNK] \n ! \n [X_SEP] \n [SEP] \n [MASK] \n [PAD] \n [CLS]"
        )?;
        let path = vocab_file.into_temp_path();
        let base_vocab = ProphetNetVocab::from_file(path.to_path_buf().to_str().unwrap())?;

        //        When & Then
        assert_eq!(base_vocab.token_to_id("hello"), 0);
        assert_eq!(base_vocab.token_to_id("world"), 1);
        assert_eq!(base_vocab.token_to_id("!"), 3);
        assert_eq!(base_vocab.token_to_id("[UNK]"), 2);
        assert_eq!(base_vocab.token_to_id("oov_value"), 2);
        assert_eq!(base_vocab.token_to_id("[PAD]"), 7);
        assert_eq!(base_vocab.token_to_id("[MASK]"), 6);
        assert_eq!(base_vocab.token_to_id("[X_SEP]"), 4);
        assert_eq!(base_vocab.token_to_id("[SEP]"), 5);
        assert_eq!(base_vocab.token_to_id("[CLS]"), 8);

        drop(path);
        Ok(())
    }

    #[test]
    fn test_decode_tokens() -> anyhow::Result<()> {
        //        Given
        let mut vocab_file = tempfile::NamedTempFile::new()?;
        write!(
            vocab_file,
            "hello \n world \n [UNK] \n ! \n [X_SEP] \n [SEP] \n [MASK] \n [PAD] \n [CLS]"
        )?;
        let path = vocab_file.into_temp_path();
        let bert_vocab = ProphetNetVocab::from_file(path.to_path_buf().to_str().unwrap())?;

        //        When & Then
        assert_eq!(bert_vocab.id_to_token(&(0_i64)), "hello");
        assert_eq!(bert_vocab.id_to_token(&(1_i64)), "world");
        assert_eq!(bert_vocab.id_to_token(&(3_i64)), "!");
        assert_eq!(bert_vocab.id_to_token(&(2_i64)), "[UNK]");
        assert_eq!(bert_vocab.id_to_token(&(7_i64)), "[PAD]");
        assert_eq!(bert_vocab.id_to_token(&(6_i64)), "[MASK]");
        assert_eq!(bert_vocab.id_to_token(&(4_i64)), "[X_SEP]");
        assert_eq!(bert_vocab.id_to_token(&(5_i64)), "[SEP]");
        assert_eq!(bert_vocab.id_to_token(&(8_i64)), "[CLS]");

        drop(path);
        Ok(())
    }
}
