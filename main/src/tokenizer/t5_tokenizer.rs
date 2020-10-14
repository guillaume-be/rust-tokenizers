// Copyright 2018 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
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
    _clean_text, decompose_nfkc, is_whitespace, lowercase, split_on_special_tokens,
};
use crate::tokenizer::{MultiThreadedTokenizer, Tokenizer};
use crate::vocab::{SentencePieceModel, T5Vocab, Vocab};
use crate::{Mask, Token, TokenRef};

/// # T5 tokenizer
/// T5 tokenizer performing:
/// - Splitting on special tokens
/// - text cleaning
/// - NFKC decomposition
/// - (optional) lower casing
/// - SentencePiece decomposition
#[derive(Debug, Clone)]
pub struct T5Tokenizer {
    model: SentencePieceModel,
    vocab: T5Vocab,
    lower_case: bool,
}

impl T5Tokenizer {
    /// Create a new instance of a `T5Tokenizer`
    /// Expects a SentencePiece protobuf file as an input.
    ///
    /// # Parameters
    /// - path (`&str`): path to the SentencePiece model file
    /// - lower_case (`bool`): flag indicating if the text should be lower-cased as part of the tokenization
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_tokenizers::tokenizer::{T5Tokenizer, Tokenizer};
    /// let lower_case = false;
    /// let tokenizer = T5Tokenizer::from_file("path/to/vocab/file", lower_case).unwrap();
    /// ```
    pub fn from_file(path: &str, lower_case: bool) -> Result<T5Tokenizer, TokenizerError> {
        let model = SentencePieceModel::from_file(path)?;
        let vocab = T5Vocab::from_file(path)?;
        Ok(T5Tokenizer {
            model,
            vocab,
            lower_case,
        })
    }

    /// Create a new instance of a `T5Tokenizer` from an existing vocabulary and model
    ///
    /// # Parameters
    /// - vocab (`T5Vocab`): vocabulary
    /// - model (`SentencePieceModel`): SentencePiece model
    /// - lower_case (`bool`): flag indicating if the text should be lower-cased as part of the tokenization
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_tokenizers::tokenizer::{T5Tokenizer, Tokenizer};
    /// use rust_tokenizers::vocab::{SentencePieceModel, T5Vocab, Vocab};
    /// let lower_case = false;
    /// let vocab = T5Vocab::from_file("path/to/vocab/file").unwrap();
    /// let model = SentencePieceModel::from_file("path/to/model/file").unwrap();
    ///
    /// let tokenizer = T5Tokenizer::from_existing_vocab_and_model(vocab, model, lower_case);
    /// ```
    pub fn from_existing_vocab_and_model(
        vocab: T5Vocab,
        model: SentencePieceModel,
        lower_case: bool,
    ) -> T5Tokenizer {
        T5Tokenizer {
            model,
            vocab,
            lower_case,
        }
    }
}

impl Tokenizer<T5Vocab> for T5Tokenizer {
    fn vocab(&self) -> &T5Vocab {
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
                _clean_text(token, true);
                decompose_nfkc(token);
                if self.lower_case {
                    lowercase(token);
                }
                token.text = token.text.replace(|c: char| is_whitespace(&c), "\u{2581}");
                if !token.text.starts_with('\u{2581}') {
                    token.text.insert(0, '\u{2581}');
                    token.reference_offsets.insert(0, 0);
                };
                let output = self.model.decode_forward_token_ref(token.as_ref());
                let decoded = self.model.decode_backward(&output);

                let output: Vec<Token> = self.model.parse_nodes_to_tokens(decoded);
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
}

impl MultiThreadedTokenizer<T5Vocab> for T5Tokenizer {}
