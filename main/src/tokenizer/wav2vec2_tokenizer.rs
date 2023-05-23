// Copyright 2018 The Google AI Language Team Authors
// Copyright 2018 The HuggingFace Inc. team.
// Copyright 2019-2020 Guillaume Becquin
// Copyright 2020 Maarten van Gompel
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::path::Path;

use itertools::Itertools;

use crate::error::TokenizerError;
use crate::tokenizer::base_tokenizer::{Mask, MultiThreadedTokenizer, Offset, OffsetSize, Token, TokenIdsWithOffsets,
    TokenIdsWithSpecialTokens, TokenRef, Tokenizer,
};

use crate::vocab::{Wav2Vec2Vocab, Vocab};

use super::tokenization_utils::{split_on_special_tokens};

/// # BERT tokenizer
/// BERT tokenizer performing:
/// - BaseTokenizer tokenization (see `BaseTokenizer` for more details)
/// - WordPiece tokenization
pub struct Wav2Vec2Tokenizer {
    vocab: Wav2Vec2Vocab,
}

impl Wav2Vec2Tokenizer {
    /// Create a new instance of a `Wav2Vec2Tokenizer`
    /// Expects a vocabulary flat-file as an input.
    ///
    /// # Parameters
    /// - path (`&str`): path to the vocabulary file
    /// - lower_case (`bool`): flag indicating if the text should be lower-cased as part of the tokenization
    /// - strip_accents (`bool`): flag indicating if accents should be stripped from the text
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_tokenizers::tokenizer::{Wav2Vec2Tokenizer, Tokenizer};
    /// let strip_accents = false;
    /// let lower_case = false;
    /// let tokenizer =
    ///     Wav2Vec2Tokenizer::from_file("path/to/vocab/file", lower_case, strip_accents).unwrap();
    /// ```
    pub fn from_file<P: AsRef<Path>>(
        path: P,
    ) -> Result<Wav2Vec2Tokenizer, TokenizerError> {
        let vocab = Wav2Vec2Vocab::from_file(path)?;
        Ok(Wav2Vec2Tokenizer {
            vocab,
        })
    }

    /// Create a new instance of a `Wav2Vec2Tokenizer`
    /// Expects a vocabulary flat-file and special token mapping file as inputs.
    ///
    /// # Parameters
    /// - path (`&str`): path to the vocabulary file
    /// - lower_case (`bool`): flag indicating if the text should be lower-cased as part of the tokenization
    /// - strip_accents (`bool`): flag indicating if accents should be stripped from the text
    /// - special_token_mapping_path (`&str`): path to a special token mapping file to overwrite default special tokens
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_tokenizers::tokenizer::{Wav2Vec2Tokenizer, Tokenizer};
    /// let strip_accents = false;
    /// let lower_case = false;
    /// let tokenizer = Wav2Vec2Tokenizer::from_file_with_special_token_mapping(
    ///     "path/to/vocab/file",
    ///     lower_case,
    ///     strip_accents,
    ///     "path/to/special/token/mapping/file",
    /// )
    /// .unwrap();
    /// ```
    pub fn from_file_with_special_token_mapping<P: AsRef<Path>, S: AsRef<Path>>(
        path: P,
        special_token_mapping_path: S,
    ) -> Result<Wav2Vec2Tokenizer, TokenizerError> {
        let vocab =
            Wav2Vec2Vocab::from_file_with_special_token_mapping(path, special_token_mapping_path)?;
        Ok(Wav2Vec2Tokenizer {
            vocab,
        })
    }
    /// Create a new instance of a `Wav2Vec2Tokenizer` from an existing vocabulary
    ///
    /// # Parameters
    /// - vocab (`Wav2Vec2Vocab`): Thread-safe reference to a BERT vocabulary
    /// - lower_case (`bool`): flag indicating if the text should be lower-cased as part of the tokenization
    /// - strip_accents (`bool`): flag indicating if accents should be stripped from the text
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_tokenizers::tokenizer::{Wav2Vec2Tokenizer, Tokenizer};
    /// use rust_tokenizers::vocab::{Wav2Vec2Vocab, Vocab};
    /// let strip_accents = false;
    /// let lower_case = false;
    /// let vocab = Wav2Vec2Vocab::from_file("path/to/vocab/file").unwrap();
    ///
    /// let tokenizer = Wav2Vec2Tokenizer::from_existing_vocab(vocab, lower_case, strip_accents);
    /// ```
    pub fn from_existing_vocab(
        vocab: Wav2Vec2Vocab,
    ) -> Wav2Vec2Tokenizer {
        Wav2Vec2Tokenizer {
            vocab,
        }
    }
}

impl Tokenizer<Wav2Vec2Vocab> for Wav2Vec2Tokenizer {
    fn vocab(&self) -> &Wav2Vec2Vocab {
        &self.vocab
    }
    fn vocab_mut(&mut self) -> &mut Wav2Vec2Vocab {
        &mut self.vocab
    }

    fn tokenize_to_tokens(&self, initial_token: TokenRef) -> Vec<Token> {
        //split on whitespace
        let tokens: Vec<Token> = split_on_special_tokens(initial_token, &self.vocab)
            .into_iter()
            .flat_map(|token| {
                if token.mask == Mask::Unknown || token.mask == Mask::Special {
                    return vec![token.to_owned()];
                }
                let begin = token.offset.begin;
                let mut split_tokens = Vec::new();
                for (i, c) in token.text.char_indices() {
                    let owned_char = c.to_string();
                    if !self.vocab.values.contains_key(&owned_char) {
                        split_tokens.push(Token {
                            text: self.vocab.get_unknown_value().to_owned(),
                            offset: Offset { begin: begin + i as u32, end: begin + i as u32 },
                            reference_offsets: (begin..i as u32).into_iter().collect_vec(),
                            mask: Mask::Unknown,
                        })
                    } else {
                        split_tokens.push(Token {
                            text: owned_char,
                            offset: Offset { begin: begin + i as u32, end: begin + i as u32 },
                            reference_offsets: (begin..i as u32).into_iter().collect_vec(),
                            mask: Default::default(),
                        })
                    }
                }
                split_tokens
            })
            .filter(|token| !token.text.is_empty())
            .collect();

        tokens
    }

    fn convert_tokens_to_string(&self, tokens: Vec<String>) -> String {
        let binding = tokens.iter().group_by(|&token| token);
        let grouped_tokens = binding.into_iter().map(|group| group.0);

        let filtered_tokens = grouped_tokens.filter(|&token| token != self.vocab.get_pad_value());

        let joined_tokens: String = filtered_tokens
            .map(|token| if token == self.vocab.get_sep_value() { " " } else { token })
            .collect();

        joined_tokens.trim().to_owned()
    }

    fn build_input_with_special_tokens(
        &self,
        tokens_ids_with_offsets_1: TokenIdsWithOffsets,
        tokens_ids_with_offsets_2: Option<TokenIdsWithOffsets>,
    ) -> TokenIdsWithSpecialTokens {
        let mut output: Vec<i64> = vec![];
        let mut token_segment_ids: Vec<i8> = vec![];
        let mut special_tokens_mask: Vec<i8> = vec![];
        let mut offsets: Vec<Option<Offset>> = vec![];
        let mut original_offsets: Vec<Vec<OffsetSize>> = vec![];
        let mut mask: Vec<Mask> = vec![];
        special_tokens_mask.push(1);
        special_tokens_mask.extend(vec![0; tokens_ids_with_offsets_1.ids.len()]);
        special_tokens_mask.push(1);
        token_segment_ids.extend(vec![0; tokens_ids_with_offsets_1.ids.len() + 2]);
        output.extend(tokens_ids_with_offsets_1.ids);
        output.push(self.vocab.token_to_id(self.vocab.get_sep_value()));
        offsets.push(None);
        offsets.extend(tokens_ids_with_offsets_1.offsets);
        offsets.push(None);
        original_offsets.push(vec![]);
        original_offsets.extend(tokens_ids_with_offsets_1.reference_offsets);
        original_offsets.push(vec![]);
        mask.push(Mask::Special);
        mask.extend(tokens_ids_with_offsets_1.masks);
        mask.push(Mask::Special);
        if let Some(tokens_ids_with_offsets_2_value) = tokens_ids_with_offsets_2 {
            let length = tokens_ids_with_offsets_2_value.ids.len();
            special_tokens_mask.push(1);
            special_tokens_mask.extend(vec![0; length]);
            special_tokens_mask.push(1);
            token_segment_ids.push(0);
            token_segment_ids.extend(vec![1; length + 1]);
            output.push(self.vocab.token_to_id(self.vocab.get_sep_value()));
            output.extend(tokens_ids_with_offsets_2_value.ids);
            output.push(self.vocab.token_to_id(self.vocab.get_sep_value()));
            offsets.push(None);
            offsets.extend(tokens_ids_with_offsets_2_value.offsets);
            original_offsets.extend(tokens_ids_with_offsets_2_value.reference_offsets);
            offsets.push(None);
            original_offsets.push(vec![]);
            mask.extend(tokens_ids_with_offsets_2_value.masks);
            mask.push(Mask::Special);
        }
        TokenIdsWithSpecialTokens {
            token_ids: output,
            segment_ids: token_segment_ids,
            special_tokens_mask,
            token_offsets: offsets,
            reference_offsets: original_offsets,
            mask,
        }
    }
}

impl MultiThreadedTokenizer<Wav2Vec2Vocab> for Wav2Vec2Tokenizer {}

//==============================
// Unit tests
//==============================
#[cfg(test)]
mod tests {
    use super::*;
    
    use crate::vocab::base_vocab::{swap_key_values, SpecialTokenMap};
    use crate::vocab::Wav2Vec2Vocab;
    
    
    use std::collections::HashMap;

    fn generate_test_vocab() -> Wav2Vec2Vocab {
        let values: HashMap<String, i64> = [
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

        let special_values: HashMap<String, i64> = [
            ("<pad>".to_owned(), 0),
            ("<s>".to_owned(), 1),
            ("</s>".to_owned(), 2),
            ("<unk>".to_owned(), 3),
            ("|".to_owned(), 4),
        ]
        .iter()
        .cloned()
        .collect();

        let indices = swap_key_values(&values);
        let special_indices = swap_key_values(&special_values);

        Wav2Vec2Vocab {
            values,
            indices,
            special_token_map,
            special_values,
            special_indices,
        }
    }

    #[test]
    fn test_wav_tokenizer() {
        //        Given
        let vocab = generate_test_vocab();
        let wav_tokenizer: Wav2Vec2Tokenizer = Wav2Vec2Tokenizer::from_existing_vocab(vocab);
        let test_tuples = [
            ("HELLO|WORLD", vec!["H", "E", "L", "L", "O", "|", "W", "O", "R", "L", "D"]),
            (
                "<s><pad>,H</s>",
                vec!["<s>", "<pad>", "<unk>", "H", "</s>"],
            ),
            (
                "<unk>中<pad> ",
                vec![
                    "<unk>", "<unk>", "<pad>", "<unk>",
                ],
            ),
        ];
        let source_texts: Vec<&str> = test_tuples.iter().map(|v| v.0).collect();
        let expected_results: Vec<Vec<&str>> = test_tuples.iter().map(|v| v.1.clone()).collect();

        //        When & Then
        for (source_text, expected_result) in test_tuples.iter() {
            assert_eq!(wav_tokenizer.tokenize(source_text), *expected_result);
        }

        assert_eq!(
            Tokenizer::tokenize_list(&wav_tokenizer, &source_texts),
            expected_results
        );
        assert_eq!(
            MultiThreadedTokenizer::tokenize_list(&wav_tokenizer, &source_texts),
            expected_results
        );
    }
}
