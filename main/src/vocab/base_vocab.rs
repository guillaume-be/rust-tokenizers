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
use std::collections::HashMap;
use std::fs::File;
use std::hash::Hash;
use std::io::{BufRead, BufReader};

pub(crate) fn swap_key_values<T: Clone, U: Hash + Eq + Copy>(
    input_hashmap: &HashMap<T, U>,
) -> HashMap<U, T> {
    input_hashmap
        .iter()
        .map(|(key, &value)| (value, key.clone()))
        .collect()
}

/// # Base Vocab trait
/// Defines a common interface to the vocabularies for use in the tokenizers.
pub trait Vocab {
    /// Associative function returning the unknown value for the vocabulary
    fn unknown_value() -> &'static str;

    /// Returns the unknown value on an instance
    fn get_unknown_value(&self) -> &'static str;

    /// Return the map of token strings to IDs
    fn values(&self) -> &HashMap<String, i64>;

    ///Return the map of token IDs to strings
    fn indices(&self) -> &HashMap<i64, String>;

    ///Return the map of token strings to IDs
    fn special_values(&self) -> &HashMap<String, i64>;

    ///Return the map of token IDs to strings for special values
    fn special_indices(&self) -> &HashMap<i64, String>;

    ///Read a vocabulary from file
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_tokenizers::vocab::{BertVocab, Vocab};
    /// let path = "path/to/file";
    ///
    /// let base_vocab = BertVocab::from_file(path);
    /// ```
    fn from_file(path: &str) -> Result<Self, TokenizerError>
    where
        Self: std::marker::Sized;

    /// Read a Bert-style vocab.txt file (single column, one token per line)
    /// The `from_file` method should be preferred, and needs to be implemented by the specific vocabularies
    fn read_vocab_file(path: &str) -> Result<HashMap<String, i64>, TokenizerError> {
        let f = File::open(path).map_err(|e| {
            TokenizerError::FileNotFound(format!("{} vocabulary file not found :{}", path, e))
        })?;
        let br = BufReader::new(f);
        let mut data = HashMap::new();

        for (index, line) in br.lines().enumerate() {
            let line = match line {
                Ok(value) => value,
                Err(e) => {
                    return Err(TokenizerError::VocabularyParsingError(e.to_string()));
                }
            };
            data.insert(line.trim().to_owned(), index as i64);
        }
        Ok(data)
    }

    /// Converts a token to an id, provided a `HashMap` of values, a `HashMap` of special values and
    /// the unknown value token string representation. This is not meant to be directly used, the method
    /// `token_to_id` offers a more convenient interface for most vocabularies, but needs to be implemented
    /// by the specific vocabulary.
    ///
    /// # Parameters
    /// - token (`&str`): token to convert
    /// - values (`&HashMap<String, i64>`): mapping from tokens to ids
    /// - special_values (`&HashMap<String, i64>`): mapping from special tokens to ids
    /// - unknown_value (`&str`): unknown token value
    ///
    /// # Returns
    /// - `i64`: index value for the provided token
    fn _token_to_id(
        &self,
        token: &str,
        values: &HashMap<String, i64>,
        special_values: &HashMap<String, i64>,
        unknown_value: &str,
    ) -> i64 {
        match special_values.get(token) {
            Some(index) => *index,
            None => match values.get(token) {
                Some(index) => *index,
                None => *values.get(unknown_value).unwrap(),
            },
        }
    }

    /// Converts an id to a token, provided a `HashMap` of values, a `HashMap` of special values and
    /// the unknown value token string representation. This is not meant to be directly used, the method
    /// `id_to_token` offers a more convenient interface for most vocabularies, but needs to be implemented
    /// by the specific vocabulary.
    ///
    /// # Parameters
    /// - id (`&i64`): token id to convert
    /// - indices (`&HashMap<i64, String>`): mapping from tokens to ids
    /// - special_indices (`&HashMap<i64, String>`): mapping from special tokens to ids
    /// - unknown_value (`&str`): unknown token value
    ///
    /// # Returns
    /// - `String`: token value for the index provided. If not found in the indices, returns the unknown token value
    fn _id_to_token(
        &self,
        id: &i64,
        indices: &HashMap<i64, String>,
        special_indices: &HashMap<i64, String>,
        unknown_value: &str,
    ) -> String {
        match special_indices.get(id) {
            Some(token) => token.clone(),
            None => match indices.get(id) {
                Some(token) => token.clone(),
                None => unknown_value.to_owned(),
            },
        }
    }

    /// Register a token as a special value
    ///
    /// # Parameters
    /// - token (`&str`): token to register as a special value
    /// - values (`&HashMap<String, i64>`): mapping from tokens to ids. This should contain the token to add and will be used to read the id for registration in `special_values`
    /// - special_values (`&HashMap<String, i64>`): mapping from special tokens to ids
    fn _register_as_special_value(
        token: &str,
        values: &HashMap<String, i64>,
        special_values: &mut HashMap<String, i64>,
    ) -> Result<(), TokenizerError> {
        let token_id = match values.get(token) {
            Some(index) => *index,
            None => {
                return Err(TokenizerError::TokenNotFound(format!(
                    "The special value {} could not be found in the vocabulary",
                    token
                )));
            }
        };
        special_values.insert(String::from(token), token_id);
        Ok(())
    }

    /// Converts a token to an id.
    ///
    /// # Parameters
    /// - token (`&str`): token to convert
    ///
    /// # Returns
    /// - `i64`: token index for the value provided. If not found in the indices, returns the unknown token index
    fn token_to_id(&self, token: &str) -> i64;

    /// Converts an id to a token.
    ///
    /// # Parameters
    /// - id (`&i64`): token id to convert
    ///
    /// # Returns
    /// - `String`: token value for the index provided. If not found in the indices, returns the unknown token value
    fn id_to_token(&self, id: &i64) -> String;

    /// Converts a list of tokens to a list of indices.
    ///
    /// # Parameters
    /// - tokens (`&[&str]`): list of tokens to convert
    ///
    /// # Returns
    /// - `Vec<i64>`: Vector containing the indices for the tokens provided
    fn convert_tokens_to_ids(&self, tokens: &[&str]) -> Vec<i64> {
        tokens.iter().map(|v| self.token_to_id(v)).collect()
    }
}

/// # BaseVocab
/// Base vocabulary with [UNK] unknown token used as a pre-tokenization step for BERT-class tokenizers.
/// Expects a flat text vocabulary when created from file.
pub struct BaseVocab {
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

impl Vocab for BaseVocab {
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

    fn from_file(path: &str) -> Result<BaseVocab, TokenizerError> {
        let values = BaseVocab::read_vocab_file(path)?;
        let mut special_values = HashMap::new();
        let unknown_value = BaseVocab::unknown_value();
        BaseVocab::_register_as_special_value(unknown_value, &values, &mut special_values)?;

        let indices = swap_key_values(&values);
        let special_indices = swap_key_values(&special_values);

        Ok(BaseVocab {
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
    extern crate anyhow;

    use super::*;
    use std::io::Write;

    #[test]
    fn test_create_object() {
        //        Given
        let values: HashMap<String, i64> = HashMap::new();
        let special_values: HashMap<String, i64> = HashMap::new();
        let indices: HashMap<i64, String> = HashMap::new();
        let special_indices: HashMap<i64, String> = HashMap::new();
        let unknown_value = BaseVocab::unknown_value();

        //        When
        let base_vocab = BaseVocab {
            values,
            indices,
            unknown_value,
            special_values,
            special_indices,
        };

        //        Then
        assert_eq!(base_vocab.unknown_value, "[UNK]");
        assert_eq!(base_vocab.unknown_value, BaseVocab::unknown_value());
        assert_eq!(base_vocab.values, *base_vocab.values());
        assert_eq!(base_vocab.special_values, *base_vocab.special_values());
    }

    #[test]
    fn test_create_object_from_file() -> anyhow::Result<()> {
        //        Given
        let mut vocab_file = tempfile::NamedTempFile::new()?;
        write!(vocab_file, "hello \n world \n [UNK] \n !")?;
        let path = vocab_file.into_temp_path();
        let target_values: HashMap<String, i64> = [
            ("hello".to_owned(), 0),
            ("world".to_owned(), 1),
            ("[UNK]".to_owned(), 2),
            ("!".to_owned(), 3),
        ]
        .iter()
        .cloned()
        .collect();

        let special_values: HashMap<String, i64> =
            [("[UNK]".to_owned(), 2)].iter().cloned().collect();

        //        When
        let base_vocab = BaseVocab::from_file(path.to_path_buf().to_str().unwrap())?;

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
        write!(vocab_file, "hello \n world \n !").unwrap();
        let path = vocab_file.into_temp_path();

        //        When & Then
        let _base_vocab = BaseVocab::from_file(path.to_path_buf().to_str().unwrap()).unwrap();
    }

    #[test]
    fn test_encode_tokens() -> anyhow::Result<()> {
        //        Given
        let mut vocab_file = tempfile::NamedTempFile::new()?;
        write!(vocab_file, "hello \n world \n [UNK] \n !")?;
        let path = vocab_file.into_temp_path();
        let base_vocab = BaseVocab::from_file(path.to_path_buf().to_str().unwrap())?;

        //        When & Then
        assert_eq!(base_vocab.token_to_id("hello"), 0);
        assert_eq!(base_vocab.token_to_id("world"), 1);
        assert_eq!(base_vocab.token_to_id("!"), 3);
        assert_eq!(base_vocab.token_to_id("[UNK]"), 2);
        assert_eq!(base_vocab.token_to_id("oov_value"), 2);

        drop(path);
        Ok(())
    }

    #[test]
    fn test_decode_tokens() -> anyhow::Result<()> {
        //        Given
        let mut vocab_file = tempfile::NamedTempFile::new()?;
        write!(vocab_file, "hello \n world \n [UNK] \n !")?;
        let path = vocab_file.into_temp_path();
        let base_vocab = BaseVocab::from_file(path.to_path_buf().to_str().unwrap())?;

        //        When & Then
        assert_eq!(base_vocab.id_to_token(&(0 as i64)), "hello");
        assert_eq!(base_vocab.id_to_token(&(1 as i64)), "world");
        assert_eq!(base_vocab.id_to_token(&(3 as i64)), "!");
        assert_eq!(base_vocab.id_to_token(&(2 as i64)), "[UNK]");

        drop(path);
        Ok(())
    }
}
