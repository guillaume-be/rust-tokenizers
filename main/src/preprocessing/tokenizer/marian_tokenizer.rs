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

use crate::preprocessing::vocab::sentence_piece_vocab::{SentencePieceModel};
use regex::Regex;
use crate::{Vocab, Tokenizer, MultiThreadedTokenizer};
use crate::preprocessing::tokenizer::base_tokenizer::{Token, TokenRef, Offset, OffsetSize, Mask};
use crate::tokenization_utils::{clean_text, decompose_nfkc, lowercase, is_whitespace, split_at_regex};
use crate::preprocessing::vocab::marian_vocab::MarianVocab;

pub struct MarianTokenizer {
    model: SentencePieceModel,
    vocab: MarianVocab,
    pattern_language_code: Regex,
    lower_case: bool,
}

impl MarianTokenizer {
    pub fn from_files(vocab_path: &str, model_path: &str, lower_case: bool) -> MarianTokenizer {
        let vocab = MarianVocab::from_file(vocab_path);
        let model = SentencePieceModel::from_file(model_path);
        let pattern_language_code = Regex::new(r"<<.+>>").unwrap();
        MarianTokenizer { model, vocab, pattern_language_code, lower_case }
    }

    pub fn from_existing_vocab_and_model(vocab: MarianVocab, model: SentencePieceModel, lower_case: bool) -> MarianTokenizer {
        let pattern_language_code = Regex::new(r"<<.+>>").unwrap();
        MarianTokenizer { model, vocab, pattern_language_code, lower_case }
    }
}

impl Tokenizer<MarianVocab> for MarianTokenizer {
    fn vocab(&self) -> &MarianVocab { &self.vocab }

    fn tokenize_to_tokens(&self, text: TokenRef) -> Vec<Token> {
        let tokens = split_at_regex(text, &self.pattern_language_code);
        let (code_token, mut token) = match tokens.len() {
            0 => { return vec!(); }
            1 => (None, tokens[0].to_owned()),
            2 => (Some(tokens[0].to_owned()), tokens[1].to_owned()),
            _ => {
                let mut token = Token::new("".to_string());
                for token_ref in tokens[1..].iter() {
                    token.text.push_str(token_ref.text);
                    token.reference_offsets.extend_from_slice(token_ref.reference_offsets);
                    token.offset.end = token_ref.offset.end;
                }
                (Some(tokens[0].to_owned()), token)
            }
        };

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
        if let Some(code) = code_token {
            output.push(code);
        };
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

    fn build_input_with_special_tokens(&self, tokens_1: Vec<i64>, tokens_2: Option<Vec<i64>>,
                                       offsets_1: Vec<Option<Offset>>, offsets_2: Option<Vec<Option<Offset>>>,
                                       original_offsets_1: Vec<Vec<OffsetSize>>, original_offsets_2: Option<Vec<Vec<OffsetSize>>>,
                                       mask_1: Vec<Mask>, mask_2: Option<Vec<Mask>>) -> (Vec<i64>, Vec<i8>, Vec<i8>, Vec<Option<Offset>>, Vec<Vec<OffsetSize>>, Vec<Mask>) {
        let mut output: Vec<i64> = vec!();
        let mut token_segment_ids: Vec<i8> = vec!();
        let mut special_tokens_mask: Vec<i8> = vec!();
        let mut offsets: Vec<Option<Offset>> = vec!();
        let mut original_offsets: Vec<Vec<OffsetSize>> = vec!();
        let mut mask: Vec<Mask> = vec!();
        special_tokens_mask.extend(vec![0; tokens_1.len()]);
        token_segment_ids.extend(vec![0; tokens_1.len()]);
        output.extend(tokens_1);
        offsets.extend(offsets_1);
        original_offsets.extend(original_offsets_1);
        mask.extend(mask_1);

        if let Some(add_tokens) = tokens_2 {
            let length = add_tokens.len();
            special_tokens_mask.extend(vec![0; length]);
            token_segment_ids.extend(vec![1; length + 1]);
            output.extend(add_tokens);
            if let Some(add_offsets) = offsets_2 {
                offsets.extend(add_offsets);
            } else {
                offsets.extend(vec![None; length]);
            }
            if let Some(add_original_offsets) = original_offsets_2 {
                original_offsets.extend(add_original_offsets);
            }
            if let Some(mask_2) = mask_2 {
                mask.extend(mask_2)
            } else {
                mask.extend(vec![Mask::None; length]);
            }
        }
        special_tokens_mask.push(1);
        token_segment_ids.push(1);
        output.push(self.vocab.token_to_id(MarianVocab::eos_value()));
        offsets.push(None);
        original_offsets.push(vec!());
        mask.push(Mask::Special);

        (output, token_segment_ids, special_tokens_mask, offsets, original_offsets, mask)
    }
}

impl MultiThreadedTokenizer<MarianVocab> for MarianTokenizer {}