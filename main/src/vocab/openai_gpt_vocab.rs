// Copyright 2018 Salesforce
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
use crate::vocab::base_vocab::{swap_key_values, Vocab};
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;

/// # GPT Vocab
/// Vocabulary for GPT tokenizer. Only contains the unknown token as a special value.
/// Expects a JSON-format vocabulary when created from file.
#[derive(Debug, Clone)]
pub struct OpenAiGptVocab {
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

impl Vocab for OpenAiGptVocab {
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

    fn from_file(path: &str) -> Result<OpenAiGptVocab, TokenizerError> {
        let f = File::open(path).map_err(|e| {
            TokenizerError::FileNotFound(format!("{} vocabulary file not found :{}", path, e))
        })?;
        let br = BufReader::new(f);
        let values: HashMap<String, i64> = match serde_json::from_reader(br) {
            Ok(value) => value,
            Err(e) => {
                return Err(TokenizerError::VocabularyParsingError(e.to_string()));
            }
        };
        let mut special_values = HashMap::new();
        let unknown_value = OpenAiGptVocab::unknown_value();
        OpenAiGptVocab::_register_as_special_value(unknown_value, &values, &mut special_values)?;

        let indices = swap_key_values(&values);
        let special_indices = swap_key_values(&special_values);

        Ok(OpenAiGptVocab {
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
    fn test_create_vocab() {
        //        Given
        let values: HashMap<String, i64> = HashMap::new();
        let special_values: HashMap<String, i64> = HashMap::new();
        let indices: HashMap<i64, String> = HashMap::new();
        let special_indices: HashMap<i64, String> = HashMap::new();
        let unknown_value = OpenAiGptVocab::unknown_value();

        //        When
        let openai_gpt_vocab = OpenAiGptVocab {
            values,
            indices,
            unknown_value,
            special_values,
            special_indices,
        };

        //        Then
        assert_eq!(openai_gpt_vocab.unknown_value, "<unk>");
        assert_eq!(
            openai_gpt_vocab.unknown_value,
            OpenAiGptVocab::unknown_value()
        );
        assert_eq!(openai_gpt_vocab.values, *openai_gpt_vocab.values());
        assert_eq!(
            openai_gpt_vocab.special_values,
            *openai_gpt_vocab.special_values()
        );
    }

    #[test]
    fn test_create_object_from_file() -> anyhow::Result<()> {
        //        Given
        let mut vocab_file = tempfile::NamedTempFile::new()?;
        write!(
            vocab_file,
            "{{\"hello\": 1,\n \"world\": 0,\n \"<unk>\": 2,\n \"!\": 3\n}}"
        )?;
        let path = vocab_file.into_temp_path();
        let target_values: HashMap<String, i64> = [
            ("hello".to_owned(), 1),
            ("world".to_owned(), 0),
            ("<unk>".to_owned(), 2),
            ("!".to_owned(), 3),
        ]
        .iter()
        .cloned()
        .collect();

        let special_values: HashMap<String, i64> =
            [("<unk>".to_owned(), 2)].iter().cloned().collect();

        //        When
        let openai_gpt_vocab = OpenAiGptVocab::from_file(path.to_path_buf().to_str().unwrap())?;

        //        Then
        assert_eq!(openai_gpt_vocab.unknown_value, "<unk>");
        assert_eq!(openai_gpt_vocab.values, target_values);
        assert_eq!(openai_gpt_vocab.special_values, special_values);
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
        let _ctrl_vocab = OpenAiGptVocab::from_file(path.to_path_buf().to_str().unwrap()).unwrap();
    }

    #[test]
    fn test_encode_tokens() -> anyhow::Result<()> {
        //        Given
        let mut vocab_file = tempfile::NamedTempFile::new()?;
        write!(
            vocab_file,
            "{{\"hello\": 1,\n \"world\": 0,\n \"<unk>\": 2,\n \"!\": 3\n}}"
        )?;
        let path = vocab_file.into_temp_path();
        let openai_gpt_vocab = OpenAiGptVocab::from_file(path.to_path_buf().to_str().unwrap())?;

        //        When & Then
        assert_eq!(openai_gpt_vocab.token_to_id("hello"), 1);
        assert_eq!(openai_gpt_vocab.token_to_id("world"), 0);
        assert_eq!(openai_gpt_vocab.token_to_id("!"), 3);
        assert_eq!(openai_gpt_vocab.token_to_id("<unk>"), 2);
        assert_eq!(openai_gpt_vocab.token_to_id("oov_value"), 2);

        drop(path);
        Ok(())
    }

    #[test]
    fn test_decode_tokens() -> anyhow::Result<()> {
        //        Given
        let mut vocab_file = tempfile::NamedTempFile::new()?;
        write!(
            vocab_file,
            "{{\"hello\": 1,\n \"world\": 0,\n \"<unk>\": 2,\n \"!\": 3\n}}"
        )?;
        let path = vocab_file.into_temp_path();
        let openai_gpt_vocab = OpenAiGptVocab::from_file(path.to_path_buf().to_str().unwrap())?;

        //        When & Then
        assert_eq!(openai_gpt_vocab.id_to_token(&(1_i64)), "hello");
        assert_eq!(openai_gpt_vocab.id_to_token(&(0_i64)), "world");
        assert_eq!(openai_gpt_vocab.id_to_token(&(3_i64)), "!");
        assert_eq!(openai_gpt_vocab.id_to_token(&(2_i64)), "<unk>");

        drop(path);
        Ok(())
    }
}
