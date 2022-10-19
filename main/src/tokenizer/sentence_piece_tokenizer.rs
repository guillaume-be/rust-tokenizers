// Copyright 2019 Google LLC. All Rights Reserved.
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
use crate::tokenizer::base_tokenizer::{Token, TokenRef};
use crate::tokenizer::tokenization_utils::{clean_text, lowercase};
use crate::tokenizer::tokenization_utils::{decompose_nfkc, is_whitespace};
use crate::tokenizer::{MultiThreadedTokenizer, Tokenizer};
use crate::vocab::{SentencePieceModel, SentencePieceVocab, Vocab};

/// # SentencePiece tokenizer
/// SentencePiece tokenizer performing:
/// - text cleaning
/// - NFKC decomposition
/// - (optional) lower casing
/// - SentencePiece decomposition
pub struct SentencePieceTokenizer {
    model: SentencePieceModel,
    vocab: SentencePieceVocab,
    lower_case: bool,
}

impl SentencePieceTokenizer {
    /// Create a new instance of a `SentencePieceTokenizer`
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
    /// use rust_tokenizers::tokenizer::{SentencePieceTokenizer, Tokenizer};
    /// use std::path::Path;
    ///
    /// let lower_case = false;
    /// let tokenizer = SentencePieceTokenizer::from_file(&Path::new("path/to/vocab/file"), lower_case).unwrap();
    /// ```
    pub fn from_file<P: AsRef<Path>>(
        path: P,
        lower_case: bool,
    ) -> Result<SentencePieceTokenizer, TokenizerError> {
        let model = SentencePieceModel::from_file(&path)?;
        let vocab = SentencePieceVocab::from_file(path)?;
        Ok(SentencePieceTokenizer {
            model,
            vocab,
            lower_case,
        })
    }

    /// Create a new instance of a `SentencePieceTokenizer`
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
    /// use rust_tokenizers::tokenizer::{SentencePieceTokenizer, Tokenizer};
    /// use std::path::Path;
    ///
    /// let lower_case = false;
    /// let tokenizer = SentencePieceTokenizer::from_file_with_special_token_mapping(
    ///     &Path::new("path/to/vocab/file"),
    ///     lower_case,
    ///     &Path::new("path/to/special/token/mapping/file"),
    /// )
    /// .unwrap();
    /// ```
    pub fn from_file_with_special_token_mapping<P: AsRef<Path>, S: AsRef<Path>>(
        path: P,
        lower_case: bool,
        special_token_mapping_path: S,
    ) -> Result<SentencePieceTokenizer, TokenizerError> {
        let model = SentencePieceModel::from_file(&path)?;
        let vocab = SentencePieceVocab::from_file_with_special_token_mapping(
            path,
            special_token_mapping_path,
        )?;
        Ok(SentencePieceTokenizer {
            model,
            vocab,
            lower_case,
        })
    }
    /// Create a new instance of a `SentencePieceTokenizer` from an existing vocabulary and model
    ///
    /// # Parameters
    /// - vocab (`SentencePieceVocab`): vocabulary
    /// - model (`SentencePieceModel`): SentencePiece model
    /// - lower_case (`bool`): flag indicating if the text should be lower-cased as part of the tokenization
    /// - strip_accents (`bool`): flag indicating if accents should be stripped from the text
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_tokenizers::tokenizer::{SentencePieceTokenizer, Tokenizer};
    /// use rust_tokenizers::vocab::{SentencePieceModel, SentencePieceVocab, Vocab};
    /// use std::path::Path;
    ///
    /// let lower_case = false;
    /// let vocab = SentencePieceVocab::from_file(&Path::new("path/to/vocab/file")).unwrap();
    /// let model = SentencePieceModel::from_file(&Path::new("path/to/model/file")).unwrap();
    ///
    /// let tokenizer = SentencePieceTokenizer::from_existing_vocab_and_model(vocab, model, lower_case);
    /// ```
    pub fn from_existing_vocab_and_model(
        vocab: SentencePieceVocab,
        model: SentencePieceModel,
        lower_case: bool,
    ) -> SentencePieceTokenizer {
        SentencePieceTokenizer {
            model,
            vocab,
            lower_case,
        }
    }
}

impl Tokenizer<SentencePieceVocab> for SentencePieceTokenizer {
    fn vocab(&self) -> &SentencePieceVocab {
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
        self.model.parse_nodes_to_tokens(decoded)
    }

    fn convert_tokens_to_string(&self, tokens: Vec<String>) -> String {
        tokens
            .into_iter()
            .map(|v| v.replace('\u{2581}', " "))
            .collect::<Vec<String>>()
            .join("")
    }
}

impl MultiThreadedTokenizer<SentencePieceVocab> for SentencePieceTokenizer {}
