// Copyright 2016 Google Inc.
// Adapted from https://github.com/google/sentencepiece/blob/master/src/bpe_model.cc
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

use std::path::Path;

use crate::error::TokenizerError;
use crate::tokenizer::tokenization_utils::{clean_text, decompose_nfkc, is_whitespace, lowercase};
use crate::tokenizer::{MultiThreadedTokenizer, Tokenizer};
use crate::vocab::{SentencePieceBpeModel, SentencePieceVocab, Vocab};
use crate::{Token, TokenRef};

/// # SentencePiece tokenizer
/// SentencePiece BPE tokenizer performing:
/// - text cleaning
/// - NFKC decomposition
/// - (optional) lower casing
/// - SentencePiece decomposition
pub struct SentencePieceBpeTokenizer {
    model: SentencePieceBpeModel,
    vocab: SentencePieceVocab,
    lower_case: bool,
}

impl SentencePieceBpeTokenizer {
    /// Create a new instance of a `SentencePieceBpeTokenizer`
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
    ) -> Result<SentencePieceBpeTokenizer, TokenizerError> {
        let model = SentencePieceBpeModel::from_file(&path)?;
        let vocab = SentencePieceVocab::from_file_with_special_token_mapping(
            path,
            special_token_mapping_path,
        )?;
        Ok(SentencePieceBpeTokenizer {
            model,
            vocab,
            lower_case,
        })
    }

    /// Create a new instance of a `SentencePieceBpeTokenizer`
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
    ) -> Result<SentencePieceBpeTokenizer, TokenizerError> {
        let model = SentencePieceBpeModel::from_file(&path)?;
        let vocab = SentencePieceVocab::from_file(path)?;
        Ok(SentencePieceBpeTokenizer {
            model,
            vocab,
            lower_case,
        })
    }

    /// Create a new instance of a `SentencePieceBpeTokenizer` from an existing vocabulary and model
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
    /// use rust_tokenizers::tokenizer::{SentencePieceBpeTokenizer, Tokenizer};
    /// use rust_tokenizers::vocab::{SentencePieceBpeModel, SentencePieceVocab, Vocab};
    /// use std::path::Path;
    ///
    /// let lower_case = false;
    /// let vocab = SentencePieceVocab::from_file(&Path::new("path/to/vocab/file")).unwrap();
    /// let model = SentencePieceBpeModel::from_file(&Path::new("path/to/model/file")).unwrap();
    ///
    /// let tokenizer =
    ///     SentencePieceBpeTokenizer::from_existing_vocab_and_model(vocab, model, lower_case);
    /// ```
    pub fn from_existing_vocab_and_model(
        vocab: SentencePieceVocab,
        model: SentencePieceBpeModel,
        lower_case: bool,
    ) -> SentencePieceBpeTokenizer {
        SentencePieceBpeTokenizer {
            model,
            vocab,
            lower_case,
        }
    }
}

impl Tokenizer<SentencePieceVocab> for SentencePieceBpeTokenizer {
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
        self.model.tokenize_to_tokens(token.as_ref())
    }

    fn convert_tokens_to_string(&self, tokens: Vec<String>) -> String {
        tokens
            .into_iter()
            .map(|v| v.replace('\u{2581}', " "))
            .collect::<Vec<String>>()
            .join("")
    }
}

impl MultiThreadedTokenizer<SentencePieceVocab> for SentencePieceBpeTokenizer {}
