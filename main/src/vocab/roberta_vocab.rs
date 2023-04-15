// Copyright 2018 The Open AI Team Authors
// Copyright 2018 The HuggingFace Inc. team.
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
    read_json_file, read_special_token_mapping_file, swap_key_values, SpecialTokenMap, Vocab,
};
use std::collections::HashMap;
use std::path::Path;

/// # RoBERTa Vocab
/// Vocabulary for RoBERTa tokenizer. Contains the following special values:
/// - PAD token
/// - BOS token
/// - EOS token
/// - SEP token
/// - MASK token
/// - CLS token
///
/// Expects a JSON-format vocabulary when created from file.
#[derive(Debug, Clone)]
pub struct RobertaVocab {
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
const DEFAULT_BOS_TOKEN: &str = "<s>";
const DEFAULT_SEP_TOKEN: &str = "</s>";
const DEFAULT_CLS_TOKEN: &str = "<s>";
const DEFAULT_EOS_TOKEN: &str = "</s>";
const DEFAULT_MASK_TOKEN: &str = "<mask>";

impl RobertaVocab {
    pub fn get_pad_value(&self) -> &str {
        self.special_token_map
            .pad_token
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

    pub fn get_cls_value(&self) -> &str {
        self.special_token_map
            .cls_token
            .as_deref()
            .unwrap_or(DEFAULT_CLS_TOKEN)
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
}

impl Vocab for RobertaVocab {
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

    fn values_mut(&mut self) -> &mut HashMap<String, i64> {
        &mut self.values
    }

    fn indices_mut(&mut self) -> &mut HashMap<i64, String> {
        &mut self.indices
    }

    fn special_values_mut(&mut self) -> &mut HashMap<String, i64> {
        &mut self.special_values
    }

    fn special_indices_mut(&mut self) -> &mut HashMap<i64, String> {
        &mut self.special_indices
    }

    ///Read a Roberta-style vocab.json file
    fn from_file<P: AsRef<Path>>(path: P) -> Result<RobertaVocab, TokenizerError> {
        let values = read_json_file(path)?;

        let special_token_map = SpecialTokenMap {
            unk_token: DEFAULT_UNK_TOKEN.to_string(),
            pad_token: Some(DEFAULT_PAD_TOKEN.to_string()),
            bos_token: Some(DEFAULT_BOS_TOKEN.to_string()),
            sep_token: Some(DEFAULT_SEP_TOKEN.to_string()),
            cls_token: Some(DEFAULT_CLS_TOKEN.to_string()),
            eos_token: Some(DEFAULT_EOS_TOKEN.to_string()),
            mask_token: Some(DEFAULT_MASK_TOKEN.to_string()),
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

//==============================
// Unit tests
//==============================
#[cfg(test)]
mod tests {
    extern crate anyhow;

    use super::*;
    use std::io::Write;

    #[test]
    fn test_create_vocab() {
        //        Given
        let values: HashMap<String, i64> = HashMap::new();
        let special_values: HashMap<String, i64> = HashMap::new();
        let indices: HashMap<i64, String> = HashMap::new();
        let special_indices: HashMap<i64, String> = HashMap::new();
        let special_token_map = SpecialTokenMap {
            unk_token: "<unk>".to_string(),
            pad_token: Some("<pad>".to_string()),
            bos_token: Some("<s>".to_string()),
            sep_token: Some("</s>".to_string()),
            cls_token: Some("<s>".to_string()),
            eos_token: Some("</s>".to_string()),
            mask_token: Some("<mask>".to_string()),
            additional_special_tokens: None,
        };

        //        When
        let roberta_vocab = RobertaVocab {
            values,
            indices,
            special_token_map,
            special_values,
            special_indices,
        };

        //        Then
        assert_eq!(roberta_vocab.get_unknown_value(), "<unk>");
        assert_eq!(roberta_vocab.values, *roberta_vocab.values());
        assert_eq!(
            roberta_vocab.special_values,
            *roberta_vocab.special_values()
        );
    }

    #[test]
    fn test_create_object_from_file() -> anyhow::Result<()> {
        //        Given
        let mut vocab_file = tempfile::NamedTempFile::new()?;
        write!(vocab_file, "{{\"hello\": 1,\n \"world\": 0,\n \"<unk>\": 2,\n \"!\": 3\n, \"<pad>\": 4\n, \"<s>\": 5\n, \"</s>\": 6\n, \"<mask>\": 7\n}}")?;
        let path = vocab_file.into_temp_path();
        let target_values: HashMap<String, i64> = [
            ("hello".to_owned(), 1),
            ("world".to_owned(), 0),
            ("<unk>".to_owned(), 2),
            ("!".to_owned(), 3),
            ("<pad>".to_owned(), 4),
            ("<s>".to_owned(), 5),
            ("</s>".to_owned(), 6),
            ("<mask>".to_owned(), 7),
        ]
        .iter()
        .cloned()
        .collect();

        let special_values: HashMap<String, i64> = [
            ("<unk>".to_owned(), 2),
            ("<pad>".to_owned(), 4),
            ("<s>".to_owned(), 5),
            ("</s>".to_owned(), 6),
            ("<mask>".to_owned(), 7),
        ]
        .iter()
        .cloned()
        .collect();

        //        When
        let roberta_vocab = RobertaVocab::from_file(&path)?;

        //        Then
        assert_eq!(roberta_vocab.get_unknown_value(), "<unk>");
        assert_eq!(roberta_vocab.values, target_values);
        assert_eq!(roberta_vocab.special_values, special_values);
        drop(path);
        Ok(())
    }

    #[test]
    #[should_panic]
    fn test_create_object_from_file_without_unknown_token() {
        //        Given
        let mut vocab_file = tempfile::NamedTempFile::new().unwrap();
        write!(vocab_file, "{{\"hello\": 1,\n \"world\": 0,\n \"!\": 3\n}}").unwrap();
        let path = vocab_file.into_temp_path();

        //        When & Then
        let _roberta_vocab = RobertaVocab::from_file(&path).unwrap();
    }

    #[test]
    fn test_encode_tokens() -> anyhow::Result<()> {
        //        Given
        let mut vocab_file = tempfile::NamedTempFile::new()?;
        write!(vocab_file, "{{\"hello\": 1,\n \"world\": 0,\n \"<unk>\": 2,\n \"!\": 3\n, \"<pad>\": 4\n, \"<s>\": 5\n, \"</s>\": 6\n, \"<mask>\": 7\n}}")?;
        let path = vocab_file.into_temp_path();
        let roberta_vocab = RobertaVocab::from_file(&path)?;

        //        When & Then
        assert_eq!(roberta_vocab.token_to_id("hello"), 1);
        assert_eq!(roberta_vocab.token_to_id("world"), 0);
        assert_eq!(roberta_vocab.token_to_id("!"), 3);
        assert_eq!(roberta_vocab.token_to_id("<unk>"), 2);
        assert_eq!(roberta_vocab.token_to_id("<s>"), 5);
        assert_eq!(roberta_vocab.token_to_id("</s>"), 6);
        assert_eq!(roberta_vocab.token_to_id("<mask>"), 7);
        assert_eq!(roberta_vocab.token_to_id("<pad>"), 4);

        drop(path);
        Ok(())
    }

    #[test]
    fn test_decode_tokens() -> anyhow::Result<()> {
        //        Given
        let mut vocab_file = tempfile::NamedTempFile::new()?;
        write!(vocab_file, "{{\"hello\": 1,\n \"world\": 0,\n \"<unk>\": 2,\n \"!\": 3\n, \"<pad>\": 4\n, \"<s>\": 5\n, \"</s>\": 6\n, \"<mask>\": 7\n}}")?;
        let path = vocab_file.into_temp_path();
        let roberta_vocab = RobertaVocab::from_file(&path)?;

        //        When & Then
        assert_eq!(roberta_vocab.id_to_token(&(1_i64)), "hello");
        assert_eq!(roberta_vocab.id_to_token(&(0_i64)), "world");
        assert_eq!(roberta_vocab.id_to_token(&(3_i64)), "!");
        assert_eq!(roberta_vocab.id_to_token(&(2_i64)), "<unk>");
        assert_eq!(roberta_vocab.id_to_token(&(5_i64)), "<s>");
        assert_eq!(roberta_vocab.id_to_token(&(6_i64)), "</s>");
        assert_eq!(roberta_vocab.id_to_token(&(7_i64)), "<mask>");
        assert_eq!(roberta_vocab.id_to_token(&(4_i64)), "<pad>");
        drop(path);
        Ok(())
    }
}
