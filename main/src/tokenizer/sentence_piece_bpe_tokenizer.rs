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

use crate::error::TokenizerError;
use crate::tokenizer::base_tokenizer::{Token, TokenRef};
use crate::tokenizer::tokenization_utils::{
    clean_text, fix_mask, lowercase, split_on_special_tokens,
};
use crate::tokenizer::tokenization_utils::{decompose_nfkc, is_whitespace};
use crate::tokenizer::{MultiThreadedTokenizer, Tokenizer};
use crate::vocab::{SentencePieceModel, SentencePieceVocab, Vocab};
use crate::Mask;
use hashbrown::HashMap;
use std::fs::File;
use std::io::Read;

#[derive(Debug, Clone)]
pub struct BpeMergeVocab {
    pub values: HashMap<String, f64>,
}

pub struct SentencePieceBpeTokenizer {
    bpe_ranks: BpeMergeVocab,
}

impl SentencePieceBpeTokenizer {
    pub fn from_file(path: &str) -> Result<SentencePieceBpeTokenizer, TokenizerError> {
        let mut f = File::open(path).map_err(|e| {
            TokenizerError::FileNotFound(format!("{} vocabulary file not found :{}", path, e))
        })?;
        let mut contents = Vec::new();
        let proto = match f.read_to_end(&mut contents) {
            Ok(_) => match ModelProto::parse_from_bytes(contents.as_slice()) {
                Ok(proto_value) => proto_value,
                Err(e) => {
                    return Err(TokenizerError::VocabularyParsingError(e.to_string()));
                }
            },
            Err(e) => {
                return Err(TokenizerError::VocabularyParsingError(e.to_string()));
            }
        };

        let mut values = HashMap::new();
        for (idx, piece) in proto.get_pieces().iter().enumerate() {
            values.insert(piece.get_piece().to_owned(), idx as f64);
        }
        let bpe_ranks = BpeMergeVocab { values };
        Ok(SentencePieceBpeTokenizer { bpe_ranks })
    }
}

impl Tokenizer<BpeMergeVocab> for SentencePieceBpeTokenizer {
    fn vocab(&self) -> &BpeMergeVocab {
        &self.vocab
    }

    fn tokenize_to_tokens(&self, initial_token: TokenRef) -> Vec<Token> {
        let mut tokens = split_on_special_tokens(initial_token, &self.vocab)
            .into_iter()
            .map(|token| token.to_owned())
            .collect::<Vec<Token>>();

        let mut sub_tokens = Vec::new();
        for token in tokens.iter_mut() {
            if token.mask != Mask::Special && token.mask != Mask::Unknown {
                if self.lower_case {
                    lowercase(token);
                }
                // ToDo: implement the BPE using target scores instead of pairs
                // sub_tokens.extend(split_on_bpe_pairs(
                //     token,
                //     bpe,
                //     &self.bpe_ranks,
                //     &self.cache,
                //     true,
                // ));
            } else {
                sub_tokens.push(token.clone());
            }
        }
        fix_mask(&mut sub_tokens);
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

impl MultiThreadedTokenizer<SentencePieceVocab> for SentencePieceTokenizer {}
