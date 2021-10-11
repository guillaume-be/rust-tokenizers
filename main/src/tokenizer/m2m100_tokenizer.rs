// Copyright 2021 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
// Copyright 2019-2021 Guillaume Becquin
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
use crate::tokenizer::base_tokenizer::{
    Mask, Offset, OffsetSize, Token, TokenIdsWithOffsets, TokenIdsWithSpecialTokens, TokenRef,
};
use crate::tokenizer::tokenization_utils::{
    clean_text, decompose_nfkc, is_whitespace, lowercase, split_on_language_code,
};
use crate::tokenizer::{MultiThreadedTokenizer, Tokenizer};
use crate::vocab::{M2M100Vocab, SentencePieceBpeModel, Vocab};

/// # M2M100 tokenizer
/// M2M100 tokenizer performing:
/// - Splitting on language and special tokens
/// - text cleaning
/// - NFKC decomposition
/// - (optional) lower casing
/// - SentencePiece decomposition
#[allow(clippy::upper_case_acronyms)]
pub struct M2M100Tokenizer {
    model: SentencePieceBpeModel,
    vocab: M2M100Vocab,
    lower_case: bool,
}

impl M2M100Tokenizer {
    /// Create a new instance of a `M2M100Tokenizer`
    /// Expects a json vocab file and a SentencePiece protobuf file as an input.
    ///
    /// # Parameters
    /// - vocab_path (`&str`): path to the JSON vocab file
    /// - model_path (`&str`): path to the SentencePiece model file
    /// - lower_case (`bool`): flag indicating if the text should be lower-cased as part of the tokenization
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_tokenizers::tokenizer::{M2M100Tokenizer, Tokenizer};
    /// let lower_case = false;
    /// let tokenizer = M2M100Tokenizer::from_files(
    ///     "path/to/vocab/file",
    ///     "path/to/spiece/model/file",
    ///     lower_case,
    /// )
    /// .unwrap();
    /// ```
    pub fn from_files(
        vocab_path: &str,
        model_path: &str,
        lower_case: bool,
    ) -> Result<M2M100Tokenizer, TokenizerError> {
        let vocab = M2M100Vocab::from_file(vocab_path)?;
        let model = SentencePieceBpeModel::from_file(model_path)?;

        Ok(M2M100Tokenizer {
            model,
            vocab,
            lower_case,
        })
    }

    /// Create a new instance of a `M2M100Tokenizer` from an existing vocabulary and model
    ///
    /// # Parameters
    /// - vocab (`M2M100Vocab`): vocabulary
    /// - model (`SentencePieceBpeModel`): SentencePiece BPE model
    /// - lower_case (`bool`): flag indicating if the text should be lower-cased as part of the tokenization
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_tokenizers::tokenizer::{M2M100Tokenizer, Tokenizer};
    /// use rust_tokenizers::vocab::{M2M100Vocab, SentencePieceBpeModel, Vocab};
    /// let lower_case = false;
    /// let vocab = M2M100Vocab::from_file("path/to/vocab/file").unwrap();
    /// let model = SentencePieceBpeModel::from_file("path/to/model/file").unwrap();
    ///
    /// let tokenizer = M2M100Tokenizer::from_existing_vocab_and_model(vocab, model, lower_case);
    /// ```
    pub fn from_existing_vocab_and_model(
        vocab: M2M100Vocab,
        model: SentencePieceBpeModel,
        lower_case: bool,
    ) -> M2M100Tokenizer {
        M2M100Tokenizer {
            model,
            vocab,
            lower_case,
        }
    }
}

impl Tokenizer<M2M100Vocab> for M2M100Tokenizer {
    fn vocab(&self) -> &M2M100Vocab {
        &self.vocab
    }

    fn tokenize_to_tokens(&self, text: TokenRef) -> Vec<Token> {
        let tokens = split_on_language_code(text, 7, &self.vocab.language_codes_bytes);
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

        let mut output: Vec<Token> = Vec::new();
        if let Some(code) = code_token {
            output.push(code);
        };
        output.extend(self.model.tokenize_to_tokens(token.as_ref()));

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
        // M2M100 is a special case where it expects the target language code to be provided in the input text
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
        token_segment_ids.extend(vec![0; tokens_ids_with_offsets_1.ids.len()]);
        output.extend(tokens_ids_with_offsets_1.ids);
        offsets.extend(tokens_ids_with_offsets_1.offsets);
        if !offsets.is_empty() {
            offsets[0] = None;
        }
        original_offsets.extend(tokens_ids_with_offsets_1.reference_offsets);
        if !original_offsets.is_empty() {
            original_offsets[0] = vec![];
        }

        mask.extend(tokens_ids_with_offsets_1.masks);
        if !mask.is_empty() {
            mask[0] = Mask::Special;
        }
        if let Some(tokens_ids_with_offsets_2_value) = tokens_ids_with_offsets_2 {
            let length = tokens_ids_with_offsets_2_value.ids.len();
            special_tokens_mask.extend(vec![0; length]);
            token_segment_ids.extend(vec![1; length + 1]);
            output.extend(tokens_ids_with_offsets_2_value.ids);
            offsets.extend(tokens_ids_with_offsets_2_value.offsets);
            original_offsets.extend(tokens_ids_with_offsets_2_value.reference_offsets);
            mask.extend(tokens_ids_with_offsets_2_value.masks);
        } else {
            token_segment_ids.push(0);
        }
        special_tokens_mask.push(1);
        output.push(self.vocab.token_to_id(M2M100Vocab::eos_value()));
        offsets.push(None);
        original_offsets.push(vec![]);
        mask.push(Mask::Special);

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

impl MultiThreadedTokenizer<M2M100Vocab> for M2M100Tokenizer {}
