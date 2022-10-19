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
use crate::vocab::sentencepiece_proto::sentencepiece_model::ModelProto;
use protobuf::Message;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::mem::ManuallyDrop;
use std::path::Path;
use std::ptr;

/// # Byte pair query
/// Structure holding a pair of bytes for query in the BPE vocabulary
#[derive(Eq, PartialEq, Hash, Clone, Debug)]
pub struct BpePairRef<'a> {
    pub byte_1: &'a String,
    pub byte_2: &'a String,
}

/// # Byte pair Encoding Vocab
/// BPE vocab containing the merges (dictionary of pairs with their priority) used to merge
/// pairs together. This vocabulary element is used on BPE tokenizers such as GPT2 or RoBERTa.
/// This vocabulary is not meant to be used directly, but rather as part of a BPE Tokenizer.
#[derive(Debug, Clone)]
pub struct BpePairVocab {
    pub values: HashMap<(String, String), i64>,
}

impl BpePairVocab {
    /// Create a new `BpePairVocab` from a flat file containing merges in the format `first_element second_element`)
    /// The indices are implied by the lien position of each pair in the merges file. The first line needs to be a
    /// header and is skipped.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_tokenizers::vocab::{BpePairVocab, Vocab};
    /// let path = "path/to/file";
    ///
    /// let bpe_vocab = BpePairVocab::from_file(path);
    /// ```
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<BpePairVocab, TokenizerError> {
        let f = File::open(&path).map_err(|e| {
            TokenizerError::FileNotFound(format!(
                "{} vocabulary file not found :{}",
                path.as_ref().display(),
                e
            ))
        })?;
        let br = BufReader::new(f);
        let mut data = HashMap::new();
        let mut index = 0;
        for line in br.lines().skip(1) {
            let line = match line {
                Ok(value) => value,
                Err(e) => {
                    return Err(TokenizerError::VocabularyParsingError(e.to_string()));
                }
            };
            let tuple: Vec<String> = line.trim().split(' ').map(|v| v.to_owned()).collect();
            if tuple.len() > 1 {
                data.insert((tuple[0].clone(), tuple[1].clone()), index);
                index += 1;
            }
        }

        Ok(BpePairVocab { values: data })
    }

    /// Create a new `BpePairVocab` from a SentencePiece file containing a BPE model.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_tokenizers::vocab::{BpePairVocab, Vocab};
    /// let path = "path/to/spiece.model";
    ///
    /// let bpe_vocab = BpePairVocab::from_sentencepiece_file(path);
    /// ```
    pub fn from_sentencepiece_file<P: AsRef<Path>>(
        path: P,
    ) -> Result<BpePairVocab, TokenizerError> {
        let mut f = File::open(&path).map_err(|e| {
            TokenizerError::FileNotFound(format!(
                "{} vocabulary file not found :{}",
                path.as_ref().display(),
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

        let mut data = HashMap::new();
        for l_piece in proto.get_pieces().iter().map(|v| v.get_piece()) {
            for r_piece in proto.get_pieces().iter().map(|v| v.get_piece()) {
                if let Some(id) = values.get(&[l_piece, r_piece].concat()) {
                    data.insert((l_piece.to_string(), r_piece.to_string()), *id);
                }
            }
        }

        Ok(BpePairVocab { values: data })
    }

    /// Gets the id of a "byte pair" in the merges vocab. Returns an optional index for the pair if
    /// it is found in the vocabulary.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_tokenizers::vocab::{BpePairRef, BpePairVocab, Vocab};
    /// let path = "path/to/file";
    ///
    /// let bpe_vocab = BpePairVocab::from_file(path).unwrap();
    ///
    /// let query = BpePairRef {
    ///     byte_1: &"won".to_string(),
    ///     byte_2: &"derful".to_string(),
    /// };
    /// let id = bpe_vocab.byte_pair_to_id(&query);
    /// ```
    pub fn byte_pair_to_id(&self, byte_pair: &BpePairRef) -> Option<&i64> {
        unsafe {
            let byte_1 = byte_pair.byte_1;
            let byte_2 = byte_pair.byte_2;
            let k = (ptr::read(byte_1), ptr::read(byte_2));
            let k = ManuallyDrop::new(k);
            self.values.get(&k)
        }
    }
}

//==============================
// Unit tests
//==============================
#[cfg(test)]
#[allow(clippy::type_complexity)]
mod tests {
    extern crate anyhow;
    use super::*;
    use std::io::Write;

    #[test]
    fn test_create_pair_vocab() {
        //        Given
        let values: HashMap<(String, String), i64> = HashMap::new();

        //        When
        let pair_vocab = BpePairVocab {
            values: values.clone(),
        };

        //        Then
        assert_eq!(pair_vocab.values, values);
    }

    #[test]
    fn test_create_pair_vocab_from_file() -> anyhow::Result<()> {
        //        Given
        let mut merges_file = tempfile::NamedTempFile::new()?;
        write!(merges_file, "#version: 0.1\n t h\na n\ni n\nth e</w>")?;
        let path = merges_file.into_temp_path();
        let target_values: HashMap<(String, String), i64> = [
            (("t".to_owned(), "h".to_owned()), 0),
            (("a".to_owned(), "n".to_owned()), 1),
            (("i".to_owned(), "n".to_owned()), 2),
            (("th".to_owned(), "e</w>".to_owned()), 3),
        ]
        .iter()
        .cloned()
        .collect();

        //        When
        let pair_vocab = BpePairVocab::from_file(&path)?;

        //        Then
        assert_eq!(pair_vocab.values, target_values);
        drop(path);
        Ok(())
    }

    #[test]
    fn test_encode_byte_pairs() -> anyhow::Result<()> {
        //        Given
        let mut merges_file = tempfile::NamedTempFile::new()?;
        write!(merges_file, "#version: 0.1\n t h\na n\ni n\nth e</w>")?;
        let path = merges_file.into_temp_path();
        let pair_vocab = BpePairVocab::from_file(&path)?;

        //        Given
        let t_token = String::from("t");
        let h_token = String::from("h");
        let a_token = String::from("a");
        let i_token = String::from("i");
        let n_token = String::from("n");
        let th_token = String::from("th");
        let e_eow_token = String::from("e</w>");

        let test_tuples = [
            ((t_token, h_token), Some(&(0_i64))),
            ((a_token.clone(), n_token.clone()), Some(&(1_i64))),
            ((i_token, n_token), Some(&(2_i64))),
            ((th_token, e_eow_token.clone()), Some(&(3_i64))),
            ((a_token, e_eow_token), None),
        ];

        //        When & Then
        for (input, expected_output) in &test_tuples {
            assert_eq!(
                pair_vocab.byte_pair_to_id(&BpePairRef {
                    byte_1: &input.0,
                    byte_2: &input.1
                }),
                *expected_output
            );
        }

        drop(path);
        Ok(())
    }
}
