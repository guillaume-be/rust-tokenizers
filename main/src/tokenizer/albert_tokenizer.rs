// Copyright 2018 Google AI, Google Brain and the HuggingFace Inc. team.
// Copyright 2019-2020 Guillaume Becquin
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
use crate::tokenizer::tokenization_utils::{
    clean_text, decompose_nfkc, is_whitespace, lowercase, replace_string, split_on_special_tokens,
    strip_accents,
};
use crate::vocab::{AlbertVocab, SentencePieceModel};

use crate::tokenizer::base_tokenizer::{TokenIdsWithOffsets, TokenIdsWithSpecialTokens};
use crate::tokenizer::MultiThreadedTokenizer;
use crate::tokenizer::Tokenizer;
use crate::vocab::Vocab;
use crate::{Mask, Offset, OffsetSize, Token, TokenRef};

/// # ALBERT tokenizer
/// ALBERT tokenizer performing:
/// - splitting on special characters
/// - text cleaning
/// - NFKC decomposition
/// - (optional) lower casing
/// - (optional) accent stripping
/// - SentencePiece decomposition
pub struct AlbertTokenizer {
    model: SentencePieceModel,
    vocab: AlbertVocab,
    lower_case: bool,
    strip_accents: bool,
}

impl AlbertTokenizer {
    /// Create a new instance of a `AlbertTokenizer`
    /// Expects a SentencePiece protobuf file as an input.
    ///
    /// # Parameters
    /// - path (`&str`): path to the SentencePiece model file
    /// - lower_case (`bool`): flag indicating if the text should be lower-cased as part of the tokenization
    /// - strip_accents (`bool`): flag indicating if accents should be stripped from the text
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_tokenizers::tokenizer::{AlbertTokenizer, Tokenizer};
    /// let strip_accents = false;
    /// let lower_case = false;
    /// let tokenizer =
    ///     AlbertTokenizer::from_file("path/to/vocab/file", lower_case, strip_accents).unwrap();
    /// ```
    pub fn from_file(
        path: &str,
        lower_case: bool,
        strip_accents: bool,
    ) -> Result<AlbertTokenizer, TokenizerError> {
        let model = SentencePieceModel::from_file(path)?;
        let vocab = AlbertVocab::from_file(path)?;
        Ok(AlbertTokenizer {
            model,
            vocab,
            lower_case,
            strip_accents,
        })
    }

    /// Create a new instance of a `AlbertTokenizer`
    /// Expects a SentencePiece protobuf file and special token mapping file as inputs.
    ///
    /// # Parameters
    /// - path (`&str`): path to the SentencePiece model file
    /// - lower_case (`bool`): flag indicating if the text should be lower-cased as part of the tokenization
    /// - strip_accents (`bool`): flag indicating if accents should be stripped from the text
    /// - special_token_mapping_path (`&str`): path to a special token mapping file to overwrite default special tokens
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_tokenizers::tokenizer::{AlbertTokenizer, Tokenizer};
    /// let strip_accents = false;
    /// let lower_case = false;
    /// let tokenizer = AlbertTokenizer::from_file_with_special_token_mapping(
    ///     "path/to/vocab/file",
    ///     lower_case,
    ///     strip_accents,
    ///     "path/to/special/token/mapping/file",
    /// )
    /// .unwrap();
    /// ```
    pub fn from_file_with_special_token_mapping(
        path: &str,
        lower_case: bool,
        strip_accents: bool,
        special_token_mapping_path: &str,
    ) -> Result<AlbertTokenizer, TokenizerError> {
        let model = SentencePieceModel::from_file(path)?;
        let vocab =
            AlbertVocab::from_file_with_special_token_mapping(path, special_token_mapping_path)?;
        Ok(AlbertTokenizer {
            model,
            vocab,
            lower_case,
            strip_accents,
        })
    }

    /// Create a new instance of a `AlbertTokenizer` from an existing vocabulary and model
    ///
    /// # Parameters
    /// - vocab (`AlbertVocab`): vocabulary
    /// - model (`SentencePieceModel`): SentencePiece model
    /// - lower_case (`bool`): flag indicating if the text should be lower-cased as part of the tokenization
    /// - strip_accents (`bool`): flag indicating if accents should be stripped from the text
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_tokenizers::tokenizer::{AlbertTokenizer, Tokenizer};
    /// use rust_tokenizers::vocab::{AlbertVocab, SentencePieceModel, Vocab};
    /// let strip_accents = false;
    /// let lower_case = false;
    /// let vocab = AlbertVocab::from_file("path/to/vocab/file").unwrap();
    /// let model = SentencePieceModel::from_file("path/to/model/file").unwrap();
    ///
    /// let tokenizer =
    ///     AlbertTokenizer::from_existing_vocab_and_model(vocab, model, lower_case, strip_accents);
    /// ```
    pub fn from_existing_vocab_and_model(
        vocab: AlbertVocab,
        model: SentencePieceModel,
        lower_case: bool,
        strip_accents: bool,
    ) -> AlbertTokenizer {
        AlbertTokenizer {
            model,
            vocab,
            lower_case,
            strip_accents,
        }
    }

    fn post_process_pieces<'a>(&self, tokens: &'a mut Vec<Token>) -> &'a Vec<Token> {
        let mut positions_to_update: Vec<(usize, Vec<Token>)> = vec![];
        for (token_idx, token) in tokens.iter().enumerate() {
            let mut token_chars = token.text.chars().rev();
            if token.text.chars().count() > 1
                && (token_chars.next().unwrap() == ',')
                    & token_chars.next().unwrap().is_ascii_digit()
            {
                let mut new_token = token.clone();
                let last_char = new_token.text.pop().unwrap();
                let updated_tokens = self.model.decode_forward_token_ref(new_token.as_ref());
                let updated_tokens = self.model.decode_backward(&updated_tokens);
                let mut updated_tokens = self.model.parse_nodes_to_tokens(updated_tokens);

                if !token.text.starts_with('\u{2581}')
                    & updated_tokens[0].text.starts_with('\u{2581}')
                {
                    if updated_tokens[0].text.chars().count() == 1 {
                        updated_tokens.remove(0);
                    } else {
                        let first_char_length =
                            updated_tokens[0].text.chars().next().unwrap().len_utf8();
                        updated_tokens[0].text = (updated_tokens[0].text[first_char_length..])
                            .parse()
                            .unwrap();
                    }
                }
                updated_tokens.push(Token {
                    text: last_char.to_string(),
                    offset: Offset {
                        begin: token.offset.end,
                        end: token.offset.end,
                    },
                    reference_offsets: vec![*token.reference_offsets.last().unwrap()],
                    mask: token.mask,
                });
                positions_to_update.push((token_idx, updated_tokens.clone()));
            }
        }
        for (pos, new_tokens) in positions_to_update {
            tokens.splice(pos..pos + 1, new_tokens);
        }
        tokens
    }
}

impl Tokenizer<AlbertVocab> for AlbertTokenizer {
    fn vocab(&self) -> &AlbertVocab {
        &self.vocab
    }

    fn tokenize_to_tokens(&self, text: TokenRef) -> Vec<Token> {
        let mut tokens = split_on_special_tokens(text, &self.vocab)
            .into_iter()
            .map(|token| token.to_owned())
            .collect::<Vec<Token>>();

        let mut sub_tokens: Vec<Token> = Vec::new();
        for token in tokens.iter_mut() {
            if token.mask != Mask::Special && token.mask != Mask::Unknown {
                replace_string(token, "``", "\"");
                replace_string(token, "\'\'", "\"");
                clean_text(token, true);
                decompose_nfkc(token);
                if self.lower_case {
                    lowercase(token);
                }
                if self.strip_accents {
                    strip_accents(token);
                }
                token.text = token.text.replace(|c: char| is_whitespace(&c), "\u{2581}");
                if !token.text.starts_with('\u{2581}') {
                    token.text.insert(0, '\u{2581}');
                    token.reference_offsets.insert(0, 0);
                };
                let output = self.model.decode_forward_token_ref(token.as_ref());
                let decoded = self.model.decode_backward(&output);

                let mut output: Vec<Token> = self.model.parse_nodes_to_tokens(decoded);
                self.post_process_pieces(&mut output);
                sub_tokens.extend(output)
            } else {
                sub_tokens.push(token.clone());
            }
        }
        sub_tokens
    }

    fn convert_tokens_to_string(&self, tokens: Vec<String>) -> String {
        tokens
            .into_iter()
            .map(|v| v.replace('\u{2581}', " "))
            .collect::<Vec<String>>()
            .join("")
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
        output.push(self.vocab.token_to_id(self.vocab.get_cls_value()));
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
            special_tokens_mask.extend(vec![0; length]);
            special_tokens_mask.push(1);
            token_segment_ids.extend(vec![1; length + 1]);
            output.extend(tokens_ids_with_offsets_2_value.ids);
            output.push(self.vocab.token_to_id(self.vocab.get_sep_value()));
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

impl MultiThreadedTokenizer<AlbertVocab> for AlbertTokenizer {}
