// Copyright 2018 The Open AI Team Authors
// Copyright 2020 Microsoft and the HuggingFace Inc. team.
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
    clean_text, decompose_nfkc, is_whitespace, split_on_special_tokens, strip_accents,
};
use crate::tokenizer::tokenization_utils::{lowercase, unknown_byte_fallback};
use crate::tokenizer::{MultiThreadedTokenizer, Tokenizer};
use crate::vocab::{DeBERTaV2Vocab, SentencePieceModel, Vocab};
use crate::{
    Mask, Offset, OffsetSize, Token, TokenIdsWithOffsets, TokenIdsWithSpecialTokens, TokenRef,
};
use std::iter::Iterator;
use std::path::Path;

/// # DeBERTaV2 tokenizer
/// DeBERTa (v2) tokenizer (based on SentencePiece) performing:
/// - splitting on special characters
/// - text cleaning
/// - NFKC decomposition
/// - (optional) lower casing
/// - (optional) accent stripping
/// - SentencePiece BPE decomposition
pub struct DeBERTaV2Tokenizer {
    model: SentencePieceModel,
    vocab: DeBERTaV2Vocab,
    lower_case: bool,
    strip_accents: bool,
    add_prefix_space: bool,
}
impl DeBERTaV2Tokenizer {
    /// Create a new instance of a `DeBERTaV2Tokenizer`
    /// Expects a SentencePiece BPE protobuf file as an input.
    ///
    /// # Parameters
    /// - path (`&str`): path to the SentencePiece model file
    /// - lower_case (`bool`): flag indicating if the text should be lower-cased as part of the tokenization
    /// - strip_accents (`bool`): flag indicating if accents should be stripped from the text
    /// - add_prefix_space (`bool`): flag indicating if a space should be preprended to the input text before tokenization
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_tokenizers::tokenizer::{DeBERTaV2Tokenizer, Tokenizer};
    ///
    /// let strip_accents = false;
    /// let lower_case = false;
    /// let add_prefix_space = false;
    /// let path = std::path::Path::new("path/to/vocab/file");
    /// let tokenizer = DeBERTaV2Tokenizer::from_file(
    ///     &path,
    ///     lower_case,
    ///     strip_accents,
    ///     add_prefix_space,
    /// )
    /// .unwrap();
    /// ```
    pub fn from_file<P: AsRef<Path>>(
        path: P,
        lower_case: bool,
        strip_accents: bool,
        add_prefix_space: bool,
    ) -> Result<DeBERTaV2Tokenizer, TokenizerError> {
        let model = SentencePieceModel::from_file(&path)?;
        let vocab = DeBERTaV2Vocab::from_file(path)?;
        Ok(DeBERTaV2Tokenizer {
            model,
            vocab,
            lower_case,
            strip_accents,
            add_prefix_space,
        })
    }

    /// Create a new instance of a `DeBERTaV2Tokenizer`
    /// Expects a SentencePiece BPE protobuf file and special token mapping file as inputs.
    ///
    /// # Parameters
    /// - path (`&str`): path to the SentencePiece model file
    /// - lower_case (`bool`): flag indicating if the text should be lower-cased as part of the tokenization
    /// - strip_accents (`bool`): flag indicating if accents should be stripped from the text
    /// - add_prefix_space (`bool`): flag indicating if a space should be preprended to the input text before tokenization
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_tokenizers::tokenizer::{DeBERTaV2Tokenizer, Tokenizer};
    /// use std::path::Path;
    ///
    /// let strip_accents = false;
    /// let lower_case = false;
    /// let add_prefix_space = false;
    /// let tokenizer = DeBERTaV2Tokenizer::from_file_with_special_token_mapping(
    ///     &Path::new("path/to/vocab/file"),
    ///     lower_case,
    ///     strip_accents,
    ///     add_prefix_space,
    ///     &Path::new("path/to/special/token/mapping/file"),
    /// )
    /// .unwrap();
    /// ```
    pub fn from_file_with_special_token_mapping(
        path: &Path,
        lower_case: bool,
        strip_accents: bool,
        add_prefix_space: bool,
        special_token_mapping_path: &Path,
    ) -> Result<DeBERTaV2Tokenizer, TokenizerError> {
        let model = SentencePieceModel::from_file(path)?;
        let vocab =
            DeBERTaV2Vocab::from_file_with_special_token_mapping(path, special_token_mapping_path)?;
        Ok(DeBERTaV2Tokenizer {
            model,
            vocab,
            lower_case,
            strip_accents,
            add_prefix_space,
        })
    }

    /// Create a new instance of a `DeBERTaV2Tokenizer` from an existing vocabulary and model
    ///
    /// # Parameters
    /// - vocab (`DeBERTaV2Vocab`): vocabulary
    /// - model (`SentencePieceBPEModel`): SentencePiece BPE model
    /// - lower_case (`bool`): flag indicating if the text should be lower-cased as part of the tokenization
    /// - strip_accents (`bool`): flag indicating if accents should be stripped from the text
    /// - add_prefix_space (`bool`): flag indicating if a space should be preprended to the input text before tokenization
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_tokenizers::tokenizer::DeBERTaV2Tokenizer;
    /// use rust_tokenizers::vocab::{DeBERTaV2Vocab, SentencePieceModel, Vocab};
    /// use std::path::Path;
    ///
    /// let strip_accents = false;
    /// let lower_case = false;
    /// let add_prefix_space = false;
    /// let vocab = DeBERTaV2Vocab::from_file(&Path::new("path/to/vocab/file")).unwrap();
    /// let model = SentencePieceModel::from_file(&Path::new("path/to/model/file")).unwrap();
    ///
    /// let tokenizer = DeBERTaV2Tokenizer::from_existing_vocab_and_model(
    ///     vocab,
    ///     model,
    ///     lower_case,
    ///     strip_accents,
    ///     add_prefix_space,
    /// );
    /// ```
    pub fn from_existing_vocab_and_model(
        vocab: DeBERTaV2Vocab,
        model: SentencePieceModel,
        lower_case: bool,
        strip_accents: bool,
        add_prefix_space: bool,
    ) -> DeBERTaV2Tokenizer {
        DeBERTaV2Tokenizer {
            model,
            vocab,
            lower_case,
            strip_accents,
            add_prefix_space,
        }
    }

    fn post_process_pieces<'a>(&self, tokens: &'a mut Vec<Token>) -> &'a Vec<Token> {
        let mut positions_to_update: Vec<(usize, Vec<Token>)> = vec![];
        for (token_idx, token) in tokens.iter().enumerate() {
            let mut token_chars = token.text.chars().rev();
            if token.text.chars().count() > 1
                && (token_chars.next().unwrap() == ',')
                    & token_chars.next().unwrap().is_ascii_digit()
            {
                let mut new_token = token.clone();
                let last_char = new_token.text.pop().unwrap();
                let updated_tokens = self.model.decode_forward_token_ref(new_token.as_ref());
                let updated_tokens = self.model.decode_backward(&updated_tokens);
                let mut updated_tokens = self.model.parse_nodes_to_tokens(updated_tokens);

                if !token.text.starts_with('\u{2581}')
                    & updated_tokens[0].text.starts_with('\u{2581}')
                {
                    if updated_tokens[0].text.chars().count() == 1 {
                        updated_tokens.remove(0);
                    } else {
                        let first_char_length =
                            updated_tokens[0].text.chars().next().unwrap().len_utf8();
                        updated_tokens[0].text = (updated_tokens[0].text[first_char_length..])
                            .parse()
                            .unwrap();
                    }
                }
                updated_tokens.push(Token {
                    text: last_char.to_string(),
                    offset: Offset {
                        begin: token.offset.end,
                        end: token.offset.end,
                    },
                    reference_offsets: vec![*token.reference_offsets.last().unwrap()],
                    mask: token.mask,
                });
                positions_to_update.push((token_idx, updated_tokens));
            }
            if let Some(byte_tokens) = unknown_byte_fallback(token.as_ref(), Tokenizer::vocab(self))
            {
                positions_to_update.push((token_idx, byte_tokens));
            }
        }
        for (pos, new_tokens) in positions_to_update.into_iter().rev() {
            tokens.splice(pos..pos + 1, new_tokens);
        }
        tokens
    }
}

impl Tokenizer<DeBERTaV2Vocab> for DeBERTaV2Tokenizer {
    fn vocab(&self) -> &DeBERTaV2Vocab {
        &self.vocab
    }

    fn tokenize_to_tokens(&self, initial_token: TokenRef) -> Vec<Token> {
        let mut initial_token: Token = initial_token.to_owned();
        if !is_whitespace(&initial_token.text.chars().next().unwrap()) & self.add_prefix_space {
            initial_token.text.insert(0, ' ');
            initial_token.reference_offsets.insert(0, 0);
        };

        let mut tokens = split_on_special_tokens(initial_token.as_ref(), &self.vocab)
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
                if self.strip_accents {
                    strip_accents(token);
                }
                token.text = token.text.replace(|c: char| is_whitespace(&c), "\u{2581}");
                if !token.text.starts_with('\u{2581}') {
                    token.text.insert(0, '\u{2581}');
                    token.reference_offsets.insert(0, 0);
                };
                let output = self.model.decode_forward_token_ref(token.as_ref());
                let decoded = self.model.decode_backward(&output);

                let mut output: Vec<Token> = self.model.parse_nodes_to_tokens(decoded);
                self.post_process_pieces(&mut output);
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
        special_tokens_mask.push(1);
        special_tokens_mask.extend(vec![0; tokens_ids_with_offsets_1.ids.len()]);
        special_tokens_mask.push(1);
        token_segment_ids.extend(vec![0; tokens_ids_with_offsets_1.ids.len() + 2]);
        output.push(self.vocab.token_to_id(self.vocab.get_cls_value()));
        output.extend(tokens_ids_with_offsets_1.ids);
        output.push(self.vocab.token_to_id(self.vocab.get_sep_value()));
        offsets.push(None);
        offsets.extend(tokens_ids_with_offsets_1.offsets);
        offsets.push(None);
        original_offsets.push(vec![]);
        original_offsets.extend(tokens_ids_with_offsets_1.reference_offsets);
        original_offsets.push(vec![]);
        mask.push(Mask::Special);
        mask.extend(tokens_ids_with_offsets_1.masks);
        mask.push(Mask::Special);
        if let Some(tokens_ids_with_offsets_2_value) = tokens_ids_with_offsets_2 {
            let length = tokens_ids_with_offsets_2_value.ids.len();
            special_tokens_mask.extend(vec![0; length]);
            special_tokens_mask.push(1);
            token_segment_ids.extend(vec![1; length + 1]);
            output.extend(tokens_ids_with_offsets_2_value.ids);
            output.push(self.vocab.token_to_id(self.vocab.get_sep_value()));
            offsets.extend(tokens_ids_with_offsets_2_value.offsets);
            original_offsets.extend(tokens_ids_with_offsets_2_value.reference_offsets);
            offsets.push(None);
            original_offsets.push(vec![]);
            mask.extend(tokens_ids_with_offsets_2_value.masks);
            mask.push(Mask::Special);
        }
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

impl MultiThreadedTokenizer<DeBERTaV2Vocab> for DeBERTaV2Tokenizer {}
