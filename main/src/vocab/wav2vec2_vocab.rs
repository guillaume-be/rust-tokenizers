// Copyright 2018 The Open AI Team Authors, The Google AI Language Team Authors
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

/// # BERT Vocab
/// Vocabulary for BERT tokenizer. Contains the following special values:
/// - CLS token
/// - SEP token
/// - PAD token
/// - MASK token
///
/// Expects a flat text vocabulary when created from file.
#[derive(Debug, Clone)]
pub struct Wav2Vec2Vocab {
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

const DEFAULT_UNK_TOKEN: &str = "<unk>";
const DEFAULT_BOS_TOKEN: &str = "<s>";
const DEFAULT_EOS_TOKEN: &str = "</s>";
const DEFAULT_PAD_TOKEN: &str = "<pad>";
const DEFAULT_SEP_TOKEN: &str = "|";

impl Wav2Vec2Vocab {
    pub fn get_bos_value(&self) -> &str {
        self.special_token_map
            .bos_token
            .as_deref()
            .unwrap_or(DEFAULT_BOS_TOKEN)
    }

    pub fn get_eos_value(&self) -> &str {
        self.special_token_map
            .eos_token
            .as_deref()
            .unwrap_or(DEFAULT_EOS_TOKEN)
    }

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
}

impl Vocab for Wav2Vec2Vocab {
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

    fn from_file<P: AsRef<Path>>(path: P) -> Result<Wav2Vec2Vocab, TokenizerError> {
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
    use super::*;
    extern crate anyhow;
    use std::io::Write;

    const VOCAB_STR: &str = "
    {
        \"<pad>\":0,
        \"<s>\":1,
        \"</s>\":2,
        \"<unk>\":3,
        \"|\":4,
        \"E\":5,
        \"T\":6,
        \"A\":7,
        \"O\":8,
        \"N\":9,
        \"I\":10,
        \"H\":11,
        \"S\":12,
        \"R\":13,
        \"D\":14,
        \"L\":15,
        \"U\":16,
        \"M\":17,
        \"W\":18,
        \"C\":19,
        \"F\":20,
        \"G\":21,
        \"Y\":22,
        \"P\":23,
        \"B\":24,
        \"V\":25,
        \"K\":26,
        \"'\":27,
        \"X\":28,
        \"J\":29,
        \"Q\":30,
        \"Z\":31
     }
    ";

    #[test]
    fn test_create_object() {
        //        Given
        let values: HashMap<String, i64> = HashMap::new();
        let special_values: HashMap<String, i64> = HashMap::new();
        let indices: HashMap<i64, String> = HashMap::new();
        let special_indices: HashMap<i64, String> = HashMap::new();
        let special_token_map = SpecialTokenMap {
            unk_token: "<unk>".to_string(),
            pad_token: Some("<pad>".to_string()),
            bos_token: Some("<s>".to_string()),
            sep_token: Some("|".to_string()),
            cls_token: None,
            eos_token: Some("</s>".to_string()),
            mask_token: None,
            additional_special_tokens: None,
        };

        //        When
        let wav_vocab = Wav2Vec2Vocab {
            values,
            indices,
            special_token_map,
            special_values,
            special_indices,
        };

        //        Then
        assert_eq!(wav_vocab.get_unknown_value(), "<unk>");
        assert_eq!(wav_vocab.values, *wav_vocab.values());
        assert_eq!(wav_vocab.special_values, *wav_vocab.special_values());
    }

    #[test]
    fn test_create_object_from_file() -> anyhow::Result<()> {
        //        Given
        let mut vocab_file = tempfile::NamedTempFile::new()?;
        write!(
            vocab_file,
            "{}", VOCAB_STR
        )?;
        let path = vocab_file.into_temp_path();
        let target_values: HashMap<String, i64> = [
            ("<pad>".to_owned(), 0),
            ("<s>".to_owned(), 1),
            ("</s>".to_owned(), 2),
            ("<unk>".to_owned(), 3),
            ("|".to_owned(), 4),
            ("E".to_owned(), 5),
            ("T".to_owned(), 6),
            ("A".to_owned(), 7),
            ("O".to_owned(), 8),
            ("N".to_owned(), 9),
            ("I".to_owned(), 10),
            ("H".to_owned(), 11),
            ("S".to_owned(), 12),
            ("R".to_owned(), 13),
            ("D".to_owned(), 14),
            ("L".to_owned(), 15),
            ("U".to_owned(), 16),
            ("M".to_owned(), 17),
            ("W".to_owned(), 18),
            ("C".to_owned(), 19),
            ("F".to_owned(), 20),
            ("G".to_owned(), 21),
            ("Y".to_owned(), 22),
            ("P".to_owned(), 23),
            ("B".to_owned(), 24),
            ("V".to_owned(), 25),
            ("K".to_owned(), 26),
            ("'".to_owned(), 27),
            ("X".to_owned(), 28),
            ("J".to_owned(), 29),
            ("Q".to_owned(), 30),
            ("Z".to_owned(), 31)
        ]
        .iter()
        .cloned()
        .collect();

        let special_values: HashMap<String, i64> = [
            ("<unk>".to_owned(), 3),
            ("|".to_owned(), 4),
            ("<pad>".to_owned(), 0),
            ("<s>".to_owned(), 1),
            ("</s>".to_owned(), 2),
        ]
        .iter()
        .cloned()
        .collect();

        //        When
        let wav_vocab = Wav2Vec2Vocab::from_file(&path)?;

        //        Then
        assert_eq!(wav_vocab.get_unknown_value(), "<unk>");
        assert_eq!(wav_vocab.values, target_values);
        assert_eq!(wav_vocab.special_values, special_values);
        drop(path);
        Ok(())
    }

    #[test]
    #[should_panic]
    fn test_create_object_from_file_without_unknown_token() {
        //        Given
        let mut vocab_file = tempfile::NamedTempFile::new().unwrap();
        write!(vocab_file, "{}", "{}").unwrap();
        let path = vocab_file.into_temp_path();

        //        When & Then
        let _base_vocab = Wav2Vec2Vocab::from_file(&path).unwrap();
    }

    #[test]
    fn test_encode_tokens() -> anyhow::Result<()> {
        //        Given
        let mut vocab_file = tempfile::NamedTempFile::new()?;
        write!(
            vocab_file,
            "{}", VOCAB_STR
        )?;
        let path = vocab_file.into_temp_path();
        let base_vocab = Wav2Vec2Vocab::from_file(&path)?;

        //        When & Then
        assert_eq!(base_vocab.token_to_id("<pad>"), 0);
        assert_eq!(base_vocab.token_to_id("<s>"), 1);
        assert_eq!(base_vocab.token_to_id("</s>"), 2);
        assert_eq!(base_vocab.token_to_id("<unk>"), 3);
        assert_eq!(base_vocab.token_to_id("oov_value"), 3);
        assert_eq!(base_vocab.token_to_id("|"), 4);
        assert_eq!(base_vocab.token_to_id("E"), 5);
        assert_eq!(base_vocab.token_to_id("T"), 6);
        assert_eq!(base_vocab.token_to_id("A"), 7);
        assert_eq!(base_vocab.token_to_id("O"), 8);
        assert_eq!(base_vocab.token_to_id("N"), 9);
        assert_eq!(base_vocab.token_to_id("I"), 10);
        assert_eq!(base_vocab.token_to_id("H"), 11);
        assert_eq!(base_vocab.token_to_id("S"), 12);
        assert_eq!(base_vocab.token_to_id("R"), 13);
        assert_eq!(base_vocab.token_to_id("D"), 14);
        assert_eq!(base_vocab.token_to_id("L"), 15);
        assert_eq!(base_vocab.token_to_id("U"), 16);
        assert_eq!(base_vocab.token_to_id("M"), 17);
        assert_eq!(base_vocab.token_to_id("W"), 18);
        assert_eq!(base_vocab.token_to_id("C"), 19);
        assert_eq!(base_vocab.token_to_id("F"), 20);
        assert_eq!(base_vocab.token_to_id("G"), 21);
        assert_eq!(base_vocab.token_to_id("Y"), 22);
        assert_eq!(base_vocab.token_to_id("P"), 23);
        assert_eq!(base_vocab.token_to_id("B"), 24);
        assert_eq!(base_vocab.token_to_id("V"), 25);
        assert_eq!(base_vocab.token_to_id("K"), 26);
        assert_eq!(base_vocab.token_to_id("'"), 27);
        assert_eq!(base_vocab.token_to_id("X"), 28);
        assert_eq!(base_vocab.token_to_id("J"), 29);
        assert_eq!(base_vocab.token_to_id("Q"), 30);
        assert_eq!(base_vocab.token_to_id("Z"), 31);

        drop(path);
        Ok(())
    }

    #[test]
    fn test_decode_tokens() -> anyhow::Result<()> {
        //        Given
        let mut vocab_file = tempfile::NamedTempFile::new()?;
        write!(
            vocab_file,
            "{}", VOCAB_STR
        )?;
        let path = vocab_file.into_temp_path();
        let wav_vocab = Wav2Vec2Vocab::from_file(&path)?;

        //        When & Then
        assert_eq!(wav_vocab.id_to_token(&(0_i64)), "<pad>");
        assert_eq!(wav_vocab.id_to_token(&(1_i64)), "<s>");
        assert_eq!(wav_vocab.id_to_token(&(2_i64)), "</s>");
        assert_eq!(wav_vocab.id_to_token(&(3_i64)), "<unk>");
        assert_eq!(wav_vocab.id_to_token(&(4_i64)), "|");
        assert_eq!(wav_vocab.id_to_token(&5_i64), "E");
        assert_eq!(wav_vocab.id_to_token(&6_i64), "T");
        assert_eq!(wav_vocab.id_to_token(&7_i64), "A");
        assert_eq!(wav_vocab.id_to_token(&8_i64), "O");
        assert_eq!(wav_vocab.id_to_token(&9_i64), "N");
        assert_eq!(wav_vocab.id_to_token(&10_i64), "I");
        assert_eq!(wav_vocab.id_to_token(&11_i64), "H");
        assert_eq!(wav_vocab.id_to_token(&12_i64), "S");
        assert_eq!(wav_vocab.id_to_token(&13_i64), "R");
        assert_eq!(wav_vocab.id_to_token(&14_i64), "D");
        assert_eq!(wav_vocab.id_to_token(&15_i64), "L");
        assert_eq!(wav_vocab.id_to_token(&16_i64), "U");
        assert_eq!(wav_vocab.id_to_token(&17_i64), "M");
        assert_eq!(wav_vocab.id_to_token(&18_i64), "W");
        assert_eq!(wav_vocab.id_to_token(&19_i64), "C");
        assert_eq!(wav_vocab.id_to_token(&20_i64), "F");
        assert_eq!(wav_vocab.id_to_token(&21_i64), "G");
        assert_eq!(wav_vocab.id_to_token(&22_i64), "Y");
        assert_eq!(wav_vocab.id_to_token(&23_i64), "P");
        assert_eq!(wav_vocab.id_to_token(&24_i64), "B");
        assert_eq!(wav_vocab.id_to_token(&25_i64), "V");
        assert_eq!(wav_vocab.id_to_token(&26_i64), "K");
        assert_eq!(wav_vocab.id_to_token(&27_i64), "'");
        assert_eq!(wav_vocab.id_to_token(&28_i64), "X");
        assert_eq!(wav_vocab.id_to_token(&29_i64), "J");
        assert_eq!(wav_vocab.id_to_token(&30_i64), "Q");
        assert_eq!(wav_vocab.id_to_token(&31_i64), "Z");

        drop(path);
        Ok(())
    }
}
