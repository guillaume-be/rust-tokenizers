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
use crate::vocab::base_vocab::{swap_key_values, Vocab};
use std::collections::HashMap;

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

    /// The string to use for unknown (out of vocabulary) tokens
    pub unknown_value: &'static str,

    /// A mapping of special value tokens as strings to IDs (i.e. the encoder base for special
    /// values), special values typically include things like BOS/EOS markers, class markers, mask
    /// markers and padding markers
    pub special_values: HashMap<String, i64>,

    /// A mapping of special value tokens as IDs to strings (i.e. the decoder base for special values)
    pub special_indices: HashMap<i64, String>,
}

impl ProphetNetVocab {
    /// Returns the PAD token for ProphetNet (`[PAD]`)
    pub fn pad_value() -> &'static str {
        "[PAD]"
    }

    /// Returns the CLS token for ProphetNet (`[CLS]`)
    pub fn cls_value() -> &'static str {
        "[CLS]"
    }

    /// Returns the SEP token for ProphetNet (`[SEP]`)
    pub fn sep_value() -> &'static str {
        "[SEP]"
    }

    /// Returns the X_SEP token for ProphetNet (`[X_SEP]`)
    pub fn x_sep_value() -> &'static str {
        "[X_SEP]"
    }

    /// Returns the MASK token for ProphetNet (`[MASK]`)
    pub fn mask_value() -> &'static str {
        "[MASK]"
    }
}

impl Vocab for ProphetNetVocab {
    fn unknown_value() -> &'static str {
        "[UNK]"
    }

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

    fn from_file(path: &str) -> Result<ProphetNetVocab, TokenizerError> {
        let values = ProphetNetVocab::read_vocab_file(path)?;
        let mut special_values = HashMap::new();

        let unknown_value = ProphetNetVocab::unknown_value();
        ProphetNetVocab::_register_as_special_value(unknown_value, &values, &mut special_values)?;

        let pad_value = ProphetNetVocab::pad_value();
        ProphetNetVocab::_register_as_special_value(pad_value, &values, &mut special_values)?;

        let cls_value = ProphetNetVocab::cls_value();
        ProphetNetVocab::_register_as_special_value(cls_value, &values, &mut special_values)?;

        let sep_value = ProphetNetVocab::sep_value();
        ProphetNetVocab::_register_as_special_value(sep_value, &values, &mut special_values)?;

        let mask_value = ProphetNetVocab::mask_value();
        ProphetNetVocab::_register_as_special_value(mask_value, &values, &mut special_values)?;

        let indices = swap_key_values(&values);
        let special_indices = swap_key_values(&special_values);

        Ok(ProphetNetVocab {
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
            &self.unknown_value,
        )
    }

    fn id_to_token(&self, id: &i64) -> String {
        self._id_to_token(
            &id,
            &self.indices,
            &self.special_indices,
            &self.unknown_value,
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
        let unknown_value = ProphetNetVocab::unknown_value();

        //        When
        let base_vocab = ProphetNetVocab {
            values,
            indices,
            unknown_value,
            special_values,
            special_indices,
        };

        //        Then
        assert_eq!(base_vocab.unknown_value, "[UNK]");
        assert_eq!(base_vocab.unknown_value, ProphetNetVocab::unknown_value());
        assert_eq!(ProphetNetVocab::pad_value(), "[PAD]");
        assert_eq!(ProphetNetVocab::sep_value(), "[SEP]");
        assert_eq!(ProphetNetVocab::x_sep_value(), "[X_SEP]");
        assert_eq!(ProphetNetVocab::mask_value(), "[MASK]");
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
        assert_eq!(base_vocab.unknown_value, "[UNK]");
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
        assert_eq!(bert_vocab.id_to_token(&(0 as i64)), "hello");
        assert_eq!(bert_vocab.id_to_token(&(1 as i64)), "world");
        assert_eq!(bert_vocab.id_to_token(&(3 as i64)), "!");
        assert_eq!(bert_vocab.id_to_token(&(2 as i64)), "[UNK]");
        assert_eq!(bert_vocab.id_to_token(&(7 as i64)), "[PAD]");
        assert_eq!(bert_vocab.id_to_token(&(6 as i64)), "[MASK]");
        assert_eq!(bert_vocab.id_to_token(&(4 as i64)), "[X_SEP]");
        assert_eq!(bert_vocab.id_to_token(&(5 as i64)), "[SEP]");
        assert_eq!(bert_vocab.id_to_token(&(8 as i64)), "[CLS]");

        drop(path);
        Ok(())
    }
}
