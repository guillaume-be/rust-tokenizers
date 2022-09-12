// Copyright 2018-2020 The HuggingFace Inc. team.
// Copyright 2020 Marian Team Authors
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
use crate::tokenizer::tokenization_utils::{clean_text, decompose_nfkc, is_whitespace, lowercase};
use crate::tokenizer::{MultiThreadedTokenizer, Tokenizer};
use crate::vocab::{PegasusVocab, SentencePieceModel, Vocab};

/// # Pegasus tokenizer
/// Pegasus tokenizer performing:
/// - text cleaning
/// - NFKC decomposition
/// - (optional) lower casing
/// - SentencePiece decomposition
pub struct PegasusTokenizer {
    model: SentencePieceModel,
    vocab: PegasusVocab,
    lower_case: bool,
}

impl PegasusTokenizer {
    /// Create a new instance of a `PegasusTokenizer`
    /// Expects a SentencePiece protobuf file as an input.
    ///
    /// # Parameters
    /// - path (`&str`): path to the SentencePiece model file
    /// - lower_case (`bool`): flag indicating if the text should be lower-cased as part of the tokenization
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_tokenizers::tokenizer::{PegasusTokenizer, Tokenizer};
    /// let lower_case = false;
    /// let tokenizer = PegasusTokenizer::from_file("path/to/vocab/file", lower_case).unwrap();
    /// ```
    pub fn from_file(
        path: &Path,
        lower_case: bool,
    ) -> Result<PegasusTokenizer, TokenizerError> {
        let vocab = PegasusVocab::from_file(&path)?;
        let model = SentencePieceModel::from_file(path)?;
        Ok(PegasusTokenizer {
            model,
            vocab,
            lower_case,
        })
    }

    /// Create a new instance of a `PegasusTokenizer`
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
    /// use rust_tokenizers::tokenizer::{PegasusTokenizer, Tokenizer};
    /// let lower_case = false;
    /// let tokenizer = PegasusTokenizer::from_file_with_special_token_mapping(
    ///     "path/to/vocab/file",
    ///     lower_case,
    ///     "path/to/special/token/mapping/file",
    /// )
    /// .unwrap();
    /// ```
    pub fn from_file_with_special_token_mapping(
        path: &Path,
        lower_case: bool,
        special_token_mapping_path: &Path,
    ) -> Result<PegasusTokenizer, TokenizerError> {
        let vocab =
            PegasusVocab::from_file_with_special_token_mapping(path, special_token_mapping_path)?;
        let model = SentencePieceModel::from_file(path)?;
        Ok(PegasusTokenizer {
            model,
            vocab,
            lower_case,
        })
    }

    /// Create a new instance of a `PegasusTokenizer` from an existing vocabulary and model
    ///
    /// # Parameters
    /// - vocab (`PegasusVocab`): vocabulary
    /// - model (`SentencePieceModel`): SentencePiece model
    /// - lower_case (`bool`): flag indicating if the text should be lower-cased as part of the tokenization
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_tokenizers::tokenizer::{PegasusTokenizer, Tokenizer};
    /// use rust_tokenizers::vocab::{PegasusVocab, SentencePieceModel, Vocab};
    /// let lower_case = false;
    /// let vocab = PegasusVocab::from_file("path/to/vocab/file").unwrap();
    /// let model = SentencePieceModel::from_file("path/to/model/file").unwrap();
    ///
    /// let tokenizer = PegasusTokenizer::from_existing_vocab_and_model(vocab, model, lower_case);
    /// ```
    pub fn from_existing_vocab_and_model(
        vocab: PegasusVocab,
        model: SentencePieceModel,
        lower_case: bool,
    ) -> PegasusTokenizer {
        PegasusTokenizer {
            model,
            vocab,
            lower_case,
        }
    }
}

impl Tokenizer<PegasusVocab> for PegasusTokenizer {
    fn vocab(&self) -> &PegasusVocab {
        &self.vocab
    }

    fn tokenize_to_tokens(&self, text: TokenRef) -> Vec<Token> {
        let mut token = text.to_owned();
        clean_text(&mut token, true);
        decompose_nfkc(&mut token);
        if self.lower_case {
            lowercase(&mut token);
        }
        token.text = token.text.replace(|c: char| is_whitespace(&c), "\u{2581}");
        if !token.text.starts_with('\u{2581}') {
            token.text.insert(0, '\u{2581}');
            token.reference_offsets.insert(0, 0);
        };
        let output = self.model.decode_forward_token_ref(token.as_ref());
        let decoded = self.model.decode_backward(&output);

        let mut output: Vec<Token> = Vec::with_capacity(decoded.len() + 1);

        let mut is_prev_unknown = false;
        for node in decoded {
            // Group unknown tokens
            if is_prev_unknown & (node.index == 0) {
                let prev_token = output.last().unwrap();
                let mut text = prev_token.text.clone();
                text.push_str(node.text);
                let mut reference_offsets = prev_token.reference_offsets.clone();
                reference_offsets.extend_from_slice(node.reference_offsets);
                let consolidated_unknown = Token {
                    text,
                    offset: Offset { begin: 0, end: 0 },
                    reference_offsets,
                    mask: Default::default(),
                };
                output.pop();
                output.push(consolidated_unknown);
            } else {
                output.push(Token {
                    text: node.text.to_owned(),
                    offset: Offset { begin: 0, end: 0 },
                    reference_offsets: node.reference_offsets.to_vec(),
                    mask: Default::default(),
                });
            }
            is_prev_unknown = node.index == 0;
        }
        self.model.populate_masks(output.as_mut_slice(), '\u{2581}');
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
        let mut output: Vec<i64> = vec![];
        let mut token_segment_ids: Vec<i8> = vec![];
        let mut special_tokens_mask: Vec<i8> = vec![];
        let mut offsets: Vec<Option<Offset>> = vec![];
        let mut original_offsets: Vec<Vec<OffsetSize>> = vec![];
        let mut mask: Vec<Mask> = vec![];
        special_tokens_mask.extend(vec![0; tokens_ids_with_offsets_1.ids.len()]);
        token_segment_ids.extend(vec![0; tokens_ids_with_offsets_1.ids.len()]);
        output.extend(tokens_ids_with_offsets_1.ids);
        offsets.extend(tokens_ids_with_offsets_1.offsets);
        original_offsets.extend(tokens_ids_with_offsets_1.reference_offsets);
        mask.extend(tokens_ids_with_offsets_1.masks);

        if let Some(tokens_ids_with_offsets_2_value) = tokens_ids_with_offsets_2 {
            let length = tokens_ids_with_offsets_2_value.ids.len();
            special_tokens_mask.extend(vec![0; length]);
            token_segment_ids.extend(vec![1; length + 1]);
            output.extend(tokens_ids_with_offsets_2_value.ids);
            offsets.extend(tokens_ids_with_offsets_2_value.offsets);
            original_offsets.extend(tokens_ids_with_offsets_2_value.reference_offsets);
            mask.extend(tokens_ids_with_offsets_2_value.masks);
        }
        special_tokens_mask.push(1);
        token_segment_ids.push(1);
        output.push(self.vocab.token_to_id(self.vocab.get_eos_value()));
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

impl MultiThreadedTokenizer<PegasusVocab> for PegasusTokenizer {}
