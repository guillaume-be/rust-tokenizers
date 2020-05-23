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

use crate::preprocessing::vocab::sentence_piece_vocab::SentencePieceVocab;
use crate::{Vocab, Tokenizer, MultiThreadedTokenizer};
use crate::preprocessing::tokenizer::base_tokenizer::{TokenRef, Token, Offset};
use crate::tokenization_utils::{is_whitespace, decompose_nfkc};
use crate::preprocessing::tokenizer::tokenization_utils::lowercase;

pub struct SentencePieceTokenizer {
    vocab: SentencePieceVocab,
    lower_case: bool,
}

impl SentencePieceTokenizer {
    pub fn from_file(path: &str, _lower_case: bool) -> SentencePieceTokenizer {
        let vocab = SentencePieceVocab::from_file(path);
        SentencePieceTokenizer { vocab, lower_case: _lower_case }
    }

    pub fn from_existing_vocab(vocab: SentencePieceVocab, _lower_case: bool) -> SentencePieceTokenizer {
        SentencePieceTokenizer { vocab, lower_case: _lower_case }
    }

    pub fn vocab(&self) -> &SentencePieceVocab {
        &self.vocab
    }

    pub fn tokenize_to_pieces(&self, text: &str) {
        let text = text.replace(' ', "\u{2581}");
        let text = text.as_str();
        let output = self.vocab.decode_forward(text);
        let _decoded = self.vocab.decode_backward(&output);
    }
}

impl Tokenizer<SentencePieceVocab> for SentencePieceTokenizer {
    fn vocab(&self) -> &SentencePieceVocab {
        &self.vocab
    }

    fn tokenize_to_tokens(&self, text: TokenRef) -> Vec<Token> {
        let mut token = text.to_owned();
        decompose_nfkc(&mut token);
        if self.lower_case {
            lowercase(&mut token);
        }
        token.text = token.text.replace(|c: char| is_whitespace(&c), "\u{2581}");
        if !token.text.starts_with('\u{2581}') {
            token.text.insert(0, '\u{2581}');
            token.reference_offsets.insert(0, 0);
        };
        let output = self.vocab.decode_forward_token_ref(token.as_ref());
        let decoded = self.vocab.decode_backward(&output);

        let mut output: Vec<Token> = Vec::with_capacity(decoded.len());
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
        output
    }

    fn convert_tokens_to_string(&self, tokens: Vec<String>) -> String {
        tokens.into_iter().map(|v| v.replace('\u{2581}', " ")).collect::<Vec<String>>().join("")
    }
}

impl MultiThreadedTokenizer<SentencePieceVocab> for SentencePieceTokenizer {}