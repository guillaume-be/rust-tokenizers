use std::path::Path;

use crate::{
    error::TokenizerError,
    vocab::{NLLBVocab, SentencePieceBpeModel, Vocab},
    Mask, Offset, OffsetSize, Token, TokenIdsWithOffsets, TokenIdsWithSpecialTokens,
};

use super::{
    tokenization_utils::{clean_text, decompose_nfkc, is_whitespace, split_on_language_code},
    MultiThreadedTokenizer, Tokenizer,
};

pub struct NLLBTokenizer {
    model: SentencePieceBpeModel,
    vocab: NLLBVocab,
}

impl NLLBTokenizer {
    pub fn from_files<V: AsRef<Path>, M: AsRef<Path>, S: AsRef<Path>>(
        vocab_path: V,
        model_path: M,
        special_tokens: S,
    ) -> Result<Self, TokenizerError> {
        let model = SentencePieceBpeModel::from_file(model_path)?;
        let vocab = NLLBVocab::from_file_with_special_token_mapping(vocab_path, special_tokens)?;
        Ok(Self { model, vocab })
    }

    pub fn from_file<V: AsRef<Path>, M: AsRef<Path>>(
        vocab_path: V,
        model_path: M,
    ) -> Result<Self, TokenizerError> {
        let model = SentencePieceBpeModel::from_file(model_path)?;
        let vocab = NLLBVocab::from_file(vocab_path)?;
        Ok(Self { model, vocab })
    }
}

impl Tokenizer<NLLBVocab> for NLLBTokenizer {
    fn vocab(&self) -> &NLLBVocab {
        &self.vocab
    }

    fn tokenize_to_tokens(&self, text: crate::TokenRef) -> Vec<crate::Token> {
        let tokens = split_on_language_code(text, 8, &self.vocab.language_codes_bytes);
        let (code_token, mut token) = match tokens.len() {
            0 => {
                return vec![];
            }
            1 => (None, tokens[0].to_owned()),
            _ => (Some(tokens[0].to_owned()), tokens[1].to_owned()),
        };

        clean_text(&mut token, true);
        decompose_nfkc(&mut token);

        token.text = token.text.replace(|c: char| is_whitespace(&c), "\u{2581}");

        if !token.text.starts_with('\u{2581}') {
            token.text.insert(0, '\u{2581}');
            token
                .reference_offsets
                .insert(0, token.reference_offsets[0]);
        };

        let mut output: Vec<Token> = Vec::new();
        if let Some(code) = code_token {
            output.push(code);
        };
        output.extend(self.model.tokenize_to_tokens(token.as_ref()));

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
    ) -> crate::TokenIdsWithSpecialTokens {
        // M2M100 is a special case where it expects the target language code to be provided in the input text
        // This is similar to Marian where the target language may be passed before the sentence to translate
        let mut output: Vec<i64> = vec![];
        let mut token_segment_ids: Vec<i8> = vec![];
        let mut special_tokens_mask: Vec<i8> = vec![];
        let mut offsets: Vec<Option<Offset>> = vec![];
        let mut original_offsets: Vec<Vec<OffsetSize>> = vec![];
        let mut mask: Vec<Mask> = vec![];
        special_tokens_mask.extend(vec![0; tokens_ids_with_offsets_1.ids.len()]);
        if !special_tokens_mask.is_empty() {
            special_tokens_mask[0] = 1;
        }
        token_segment_ids.extend(vec![0; tokens_ids_with_offsets_1.ids.len()]);
        output.extend(tokens_ids_with_offsets_1.ids);
        offsets.extend(tokens_ids_with_offsets_1.offsets);
        if !offsets.is_empty() {
            offsets[0] = None;
        }
        original_offsets.extend(tokens_ids_with_offsets_1.reference_offsets);
        if !original_offsets.is_empty() {
            original_offsets[0] = vec![];
        }

        mask.extend(tokens_ids_with_offsets_1.masks);
        if !mask.is_empty() {
            mask[0] = Mask::Special;
        }
        if let Some(tokens_ids_with_offsets_2_value) = tokens_ids_with_offsets_2 {
            let length = tokens_ids_with_offsets_2_value.ids.len();
            special_tokens_mask.extend(vec![0; length]);
            token_segment_ids.extend(vec![1; length + 1]);
            output.extend(tokens_ids_with_offsets_2_value.ids);
            offsets.extend(tokens_ids_with_offsets_2_value.offsets);
            original_offsets.extend(tokens_ids_with_offsets_2_value.reference_offsets);
            mask.extend(tokens_ids_with_offsets_2_value.masks);
        } else {
            token_segment_ids.push(0);
        }
        special_tokens_mask.push(1);
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

impl MultiThreadedTokenizer<NLLBVocab> for NLLBTokenizer {}
