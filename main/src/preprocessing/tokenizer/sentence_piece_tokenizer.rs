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

use crate::preprocessing::vocab::sentence_piece_vocab::{SentencePieceModel, SentencePieceVocab};
use crate::{Vocab, Tokenizer, MultiThreadedTokenizer};
use crate::preprocessing::tokenizer::base_tokenizer::{TokenRef, Token};
use crate::tokenization_utils::{is_whitespace, decompose_nfkc};
use crate::preprocessing::tokenizer::tokenization_utils::{lowercase, clean_text};
use crate::preprocessing::error::TokenizerError;

pub struct SentencePieceTokenizer {
    model: SentencePieceModel,
    vocab: SentencePieceVocab,
    lower_case: bool,
}

impl SentencePieceTokenizer {
    pub fn from_file(path: &str, lower_case: bool) -> Result<SentencePieceTokenizer, TokenizerError> {
        let model = SentencePieceModel::from_file(path)?;
        let vocab = SentencePieceVocab::from_file(path)?;
        Ok(SentencePieceTokenizer { model, vocab, lower_case })
    }

    pub fn from_existing_vocab_and_model(vocab: SentencePieceVocab, model: SentencePieceModel, lower_case: bool) -> SentencePieceTokenizer {
        SentencePieceTokenizer { model, vocab, lower_case }
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

    fn convert_tokens_to_string(&self, tokens: Vec<String>) -> Result<String, TokenizerError> {
        Ok(tokens.into_iter().map(|v| v.replace('\u{2581}', " ")).collect::<Vec<String>>().join(""))
    }
}

impl MultiThreadedTokenizer<SentencePieceVocab> for SentencePieceTokenizer {}