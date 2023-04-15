// Copyright 2021 The Facebook AI Research Team Authors and The HuggingFace Inc. team.
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

use std::path::Path;

use crate::error::TokenizerError;
use crate::tokenizer::base_tokenizer::{
    Mask, Offset, OffsetSize, Token, TokenIdsWithOffsets, TokenIdsWithSpecialTokens, TokenRef,
};
use crate::tokenizer::tokenization_utils::{
    clean_text, decompose_nfkc, is_whitespace, lowercase, split_on_language_code,
};
use crate::tokenizer::{MultiThreadedTokenizer, Tokenizer};
use crate::vocab::{MBart50Vocab, SentencePieceModel, Vocab};

/// # MBart50 tokenizer
/// MBart50 tokenizer performing:
/// - Splitting on language and special tokens
/// - text cleaning
/// - NFKC decomposition
/// - (optional) lower casing
/// - SentencePiece decomposition
#[allow(clippy::upper_case_acronyms)]
pub struct MBart50Tokenizer {
    model: SentencePieceModel,
    vocab: MBart50Vocab,
    lower_case: bool,
}

impl MBart50Tokenizer {
    /// Create a new instance of a `MBart50Tokenizer`
    /// Expects a SentencePiece protobuf file as an input.
    ///
    /// # Parameters
    /// - path (`&str`): path to the SentencePiece model file
    /// - lower_case (`bool`): flag indicating if the text should be lower-cased as part of the tokenization
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_tokenizers::tokenizer::{MBart50Tokenizer, Tokenizer};
    /// let lower_case = false;
    /// let tokenizer = MBart50Tokenizer::from_file("path/to/vocab/file", lower_case).unwrap();
    /// ```
    pub fn from_file<P: AsRef<Path>>(
        path: P,
        lower_case: bool,
    ) -> Result<MBart50Tokenizer, TokenizerError> {
        let model = SentencePieceModel::from_file(&path)?;
        let vocab = MBart50Vocab::from_file(path)?;
        Ok(MBart50Tokenizer {
            model,
            vocab,
            lower_case,
        })
    }

    /// Create a new instance of a `MBart50Tokenizer`
    /// Expects a SentencePiece protobuf file and special token mapping file as inputs.
    ///
    /// # Parameters
    /// - path (`&str`): path to the SentencePiece model file
    /// - lower_case (`bool`): flag indicating if the text should be lower-cased as part of the tokenization
    /// - special_token_mapping_path (`&str`): path to a special token mapping file to overwrite default special tokens
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_tokenizers::tokenizer::{MBart50Tokenizer, Tokenizer};
    /// let lower_case = false;
    /// let tokenizer = MBart50Tokenizer::from_file_with_special_token_mapping(
    ///     "path/to/vocab/file",
    ///     lower_case,
    ///     "path/to/special/token/mapping/file",
    /// )
    /// .unwrap();
    /// ```
    pub fn from_file_with_special_token_mapping<P: AsRef<Path>, S: AsRef<Path>>(
        path: P,
        lower_case: bool,
        special_token_mapping_path: S,
    ) -> Result<MBart50Tokenizer, TokenizerError> {
        let model = SentencePieceModel::from_file(&path)?;
        let vocab =
            MBart50Vocab::from_file_with_special_token_mapping(path, special_token_mapping_path)?;
        Ok(MBart50Tokenizer {
            model,
            vocab,
            lower_case,
        })
    }

    /// Create a new instance of a `MBart50Tokenizer` from an existing vocabulary and model
    ///
    /// # Parameters
    /// - vocab (`MBart50Vocab`): vocabulary
    /// - model (`SentencePieceModel`): SentencePiece model
    /// - lower_case (`bool`): flag indicating if the text should be lower-cased as part of the tokenization
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_tokenizers::tokenizer::{MBart50Tokenizer, Tokenizer};
    /// use rust_tokenizers::vocab::{MBart50Vocab, SentencePieceModel, Vocab};
    /// let lower_case = false;
    /// let vocab = MBart50Vocab::from_file("path/to/vocab/file").unwrap();
    /// let model = SentencePieceModel::from_file("path/to/model/file").unwrap();
    ///
    /// let tokenizer = MBart50Tokenizer::from_existing_vocab_and_model(vocab, model, lower_case);
    /// ```
    pub fn from_existing_vocab_and_model(
        vocab: MBart50Vocab,
        model: SentencePieceModel,
        lower_case: bool,
    ) -> MBart50Tokenizer {
        MBart50Tokenizer {
            model,
            vocab,
            lower_case,
        }
    }
}

impl Tokenizer<MBart50Vocab> for MBart50Tokenizer {
    fn vocab(&self) -> &MBart50Vocab {
        &self.vocab
    }
    fn vocab_mut(&mut self) -> &mut MBart50Vocab {
        &mut self.vocab
    }

    fn tokenize_to_tokens(&self, text: TokenRef) -> Vec<Token> {
        let tokens = split_on_language_code(text, 6, &self.vocab.language_codes_bytes);
        let (code_token, mut token) = match tokens.len() {
            0 => {
                return vec![];
            }
            1 => (None, tokens[0].to_owned()),
            _ => (Some(tokens[0].to_owned()), tokens[1].to_owned()),
        };

        clean_text(&mut token, true);
        decompose_nfkc(&mut token);
        if self.lower_case {
            lowercase(&mut token);
        }
        token.text = token.text.replace(|c: char| is_whitespace(&c), "\u{2581}");
        if !token.text.starts_with('\u{2581}') {
            token.text.insert(0, '\u{2581}');
            token
                .reference_offsets
                .insert(0, token.reference_offsets[0]);
        };
        let output = self.model.decode_forward_token_ref(token.as_ref());
        let decoded = self.model.decode_backward(&output);

        let mut output: Vec<Token> = Vec::with_capacity(decoded.len() + 1);
        if let Some(code) = code_token {
            output.push(code);
        };
        output.extend(self.model.parse_nodes_to_tokens(decoded));
        output
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
        // MBart50 is a special case where it expects the target language to be provided in the input text
        // This is similar to Marian where the target language may be passed before the sentence to translate
        let mut output: Vec<i64> = vec![];
        let mut token_segment_ids: Vec<i8> = vec![];
        let mut special_tokens_mask: Vec<i8> = vec![];
        let mut offsets: Vec<Option<Offset>> = vec![];
        let mut original_offsets: Vec<Vec<OffsetSize>> = vec![];
        let mut mask: Vec<Mask> = vec![];
        special_tokens_mask.extend(vec![0; tokens_ids_with_offsets_1.ids.len()]);
        if !special_tokens_mask.is_empty() {
            special_tokens_mask[0] = 1;
        }
        special_tokens_mask.push(1);
        token_segment_ids.extend(vec![0; tokens_ids_with_offsets_1.ids.len() + 1]);
        output.extend(tokens_ids_with_offsets_1.ids);
        output.push(self.vocab.token_to_id(self.vocab.get_sep_value()));
        offsets.extend(tokens_ids_with_offsets_1.offsets);
        if !offsets.is_empty() {
            offsets[0] = None;
        }
        offsets.push(None);
        original_offsets.extend(tokens_ids_with_offsets_1.reference_offsets);
        if !original_offsets.is_empty() {
            original_offsets[0] = vec![];
        }
        original_offsets.push(vec![]);
        mask.extend(tokens_ids_with_offsets_1.masks);
        if !mask.is_empty() {
            mask[0] = Mask::Special;
        }
        mask.push(Mask::Special);
        if let Some(tokens_ids_with_offsets_2_value) = tokens_ids_with_offsets_2 {
            let length = tokens_ids_with_offsets_2_value.ids.len();
            special_tokens_mask.extend(vec![0; length]);
            special_tokens_mask.push(1);
            token_segment_ids.extend(vec![1; length + 1]);
            output.extend(tokens_ids_with_offsets_2_value.ids);
            output.push(self.vocab.token_to_id(self.vocab.get_sep_value()));
            offsets.extend(tokens_ids_with_offsets_2_value.offsets);
            offsets.push(None);
            original_offsets.extend(tokens_ids_with_offsets_2_value.reference_offsets);
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

impl MultiThreadedTokenizer<MBart50Vocab> for MBart50Tokenizer {}
