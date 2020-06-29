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

use crate::preprocessing::vocab::sentence_piece_vocab::{SentencePieceModel, Node};
use crate::{Vocab, Tokenizer, MultiThreadedTokenizer};
use crate::preprocessing::tokenizer::base_tokenizer::{Token, TokenRef, Offset, Mask};
use crate::tokenization_utils::{clean_text, decompose_nfkc, lowercase, is_whitespace, split_on_special_tokens};
use crate::preprocessing::vocab::t5_vocab::T5Vocab;

pub struct T5Tokenizer {
    model: SentencePieceModel,
    vocab: T5Vocab,
    lower_case: bool,
}

impl T5Tokenizer {
    pub fn from_file(path: &str, lower_case: bool) -> T5Tokenizer {
        let model = SentencePieceModel::from_file(path);
        let vocab = T5Vocab::from_file(path);
        T5Tokenizer { model, vocab, lower_case }
    }

    pub fn from_existing_vocab_and_model(vocab: T5Vocab, model: SentencePieceModel, lower_case: bool) -> T5Tokenizer {
        T5Tokenizer { model, vocab, lower_case }
    }

    fn post_process_pieces<'a>(&self, tokens: &'a mut Vec<Token>) -> &'a Vec<Token> {
        let mut positions_to_update: Vec<(usize, Vec<Token>)> = vec!();
        for (token_idx, token) in tokens.iter().enumerate() {
            let mut token_chars = token.text.chars().rev();
            if token.text.chars().count() > 1 {
                if (token_chars.next().unwrap() == ',') & token_chars.next().unwrap().is_ascii_digit() {
                    let mut new_token = token.clone();
                    let last_char = new_token.text.pop().unwrap();
                    new_token.text = new_token.text.replace('\u{2581}', "");
                    let updated_tokens = self.model.decode_forward_token_ref(new_token.as_ref());
                    let updated_tokens = self.model.decode_backward(&updated_tokens);
                    let mut updated_tokens = self.parse_nodes_to_tokens(updated_tokens);

                    if (token.text.chars().next().unwrap() != '\u{2581}') &
                        (updated_tokens[0].text.chars().next().unwrap() == '\u{2581}') {
                        if updated_tokens[0].text.chars().count() == 1 {
                            updated_tokens.remove(0);
                        } else {
                            let first_char_length = updated_tokens[0].text.chars().next().unwrap().len_utf8();
                            updated_tokens[0].text = (&updated_tokens[0].text[first_char_length..]).parse().unwrap();
                        }
                    }
                    updated_tokens.push(Token {
                        text: last_char.to_string(),
                        offset: Offset { begin: token.offset.end, end: token.offset.end - 1 },
                        reference_offsets: vec!(*token.reference_offsets.last().unwrap()),
                        mask: token.mask,
                    });
                    positions_to_update.push((token_idx, updated_tokens.clone()));
                }
            }
        };
        for (pos, new_tokens) in positions_to_update {
            tokens.splice(pos..pos, new_tokens);
        }
        tokens
    }

    fn parse_nodes_to_tokens(&self, nodes: Vec<&Node>) -> Vec<Token> {
        let mut output: Vec<Token> = Vec::with_capacity(nodes.len() + 1);
        let mut is_prev_unknown = false;
        for node in nodes {
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
}

impl Tokenizer<T5Vocab> for T5Tokenizer {
    fn vocab(&self) -> &T5Vocab { &self.vocab }

    fn tokenize_to_tokens(&self, text: TokenRef) -> Vec<Token> {
        let mut tokens = split_on_special_tokens(text, &self.vocab)
            .into_iter()
            .map(|token| token.to_owned())
            .collect::<Vec<Token>>();

        let mut sub_tokens: Vec<Token> = Vec::new();
        for token in tokens.iter_mut() {
            if token.mask != Mask::Special && token.mask != Mask::Unknown {
                clean_text(token, true);
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

                let mut output: Vec<Token> = self.parse_nodes_to_tokens(decoded);
                self.post_process_pieces(&mut output);
                sub_tokens.extend(output)
            } else {
                sub_tokens.push(token.clone());
            }
        }
        sub_tokens
    }


    fn convert_tokens_to_string(&self, tokens: Vec<String>) -> String {
        tokens.into_iter().map(|v| v.replace('\u{2581}', " ")).collect::<Vec<String>>().join("")
    }
}

impl MultiThreadedTokenizer<T5Vocab> for T5Tokenizer {}