// Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
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
use crate::tokenizer::base_tokenizer::{Mask, Offset, OffsetSize, Token, TokenRef};
use crate::tokenizer::tokenization_utils::{
    _clean_text, decompose_nfkc, is_whitespace, lowercase, split_on_special_tokens,
};
use crate::tokenizer::{MultiThreadedTokenizer, Tokenizer};
use crate::vocab::{SentencePieceModel, Vocab, XLMRobertaVocab};

pub struct XLMRobertaTokenizer {
    model: SentencePieceModel,
    vocab: XLMRobertaVocab,
    lower_case: bool,
}

impl XLMRobertaTokenizer {
    pub fn from_file(path: &str, lower_case: bool) -> Result<XLMRobertaTokenizer, TokenizerError> {
        let model = SentencePieceModel::from_file(path)?;
        let vocab = XLMRobertaVocab::from_file(path)?;
        Ok(XLMRobertaTokenizer {
            model,
            vocab,
            lower_case,
        })
    }

    pub fn from_existing_vocab_and_model(
        vocab: XLMRobertaVocab,
        model: SentencePieceModel,
        lower_case: bool,
    ) -> XLMRobertaTokenizer {
        XLMRobertaTokenizer {
            model,
            vocab,
            lower_case,
        }
    }
}

impl Tokenizer<XLMRobertaVocab> for XLMRobertaTokenizer {
    fn vocab(&self) -> &XLMRobertaVocab {
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

    fn build_input_with_special_tokens(
        &self,
        tokens_1: Vec<i64>,
        tokens_2: Option<Vec<i64>>,
        offsets_1: Vec<Option<Offset>>,
        offsets_2: Option<Vec<Option<Offset>>>,
        original_offsets_1: Vec<Vec<OffsetSize>>,
        original_offsets_2: Option<Vec<Vec<OffsetSize>>>,
        mask_1: Vec<Mask>,
        mask_2: Option<Vec<Mask>>,
    ) -> (
        Vec<i64>,
        Vec<i8>,
        Vec<i8>,
        Vec<Option<Offset>>,
        Vec<Vec<OffsetSize>>,
        Vec<Mask>,
    ) {
        let mut output: Vec<i64> = vec![];
        let mut token_segment_ids: Vec<i8> = vec![];
        let mut special_tokens_mask: Vec<i8> = vec![];
        let mut offsets: Vec<Option<Offset>> = vec![];
        let mut original_offsets: Vec<Vec<OffsetSize>> = vec![];
        let mut mask: Vec<Mask> = vec![];
        special_tokens_mask.push(1);
        special_tokens_mask.extend(vec![0; tokens_1.len()]);
        special_tokens_mask.push(1);
        token_segment_ids.extend(vec![0; tokens_1.len() + 2]);
        output.push(self.vocab.token_to_id(XLMRobertaVocab::cls_value()));
        output.extend(tokens_1);
        output.push(self.vocab.token_to_id(XLMRobertaVocab::sep_value()));
        offsets.push(None);
        offsets.extend(offsets_1);
        offsets.push(None);
        original_offsets.push(vec![]);
        original_offsets.extend(original_offsets_1);
        original_offsets.push(vec![]);
        mask.push(Mask::Special);
        mask.extend(mask_1);
        mask.push(Mask::Special);
        if let Some(add_tokens) = tokens_2 {
            let length = add_tokens.len();
            special_tokens_mask.push(1);
            special_tokens_mask.extend(vec![0; length]);
            special_tokens_mask.push(1);
            token_segment_ids.extend(vec![1; length + 2]);
            output.push(self.vocab.token_to_id(XLMRobertaVocab::sep_value()));
            output.extend(add_tokens);
            output.push(self.vocab.token_to_id(XLMRobertaVocab::sep_value()));
            if let Some(add_offsets) = offsets_2 {
                offsets.push(None);
                offsets.extend(add_offsets);
            } else {
                offsets.extend(vec![None; length + 2]);
            }
            if let Some(add_original_offsets) = original_offsets_2 {
                original_offsets.push(vec![]);
                original_offsets.extend(add_original_offsets);
            }
            offsets.push(None);
            original_offsets.push(vec![]);
            mask.push(Mask::Special);
            if let Some(mask_2) = mask_2 {
                mask.extend(mask_2)
            } else {
                mask.extend(vec![Mask::None; length]);
            }
            mask.push(Mask::Special);
        }
        (
            output,
            token_segment_ids,
            special_tokens_mask,
            offsets,
            original_offsets,
            mask,
        )
    }

    fn convert_tokens_to_string(&self, tokens: Vec<String>) -> String {
        tokens
            .into_iter()
            .map(|v| v.replace('\u{2581}', " "))
            .collect::<Vec<String>>()
            .join("")
    }
}

impl MultiThreadedTokenizer<XLMRobertaVocab> for XLMRobertaTokenizer {}
