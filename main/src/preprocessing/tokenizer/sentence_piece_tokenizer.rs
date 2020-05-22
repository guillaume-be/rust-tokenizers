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
use crate::Vocab;

pub struct SentencePieceTokenizer {
    vocab: SentencePieceVocab,
    _lower_case: bool,
}

impl SentencePieceTokenizer {
    pub fn from_file(path: &str, _lower_case: bool) -> SentencePieceTokenizer {
        let vocab = SentencePieceVocab::from_file(path);
        SentencePieceTokenizer { vocab, _lower_case }
    }

    pub fn from_existing_vocab(vocab: SentencePieceVocab, _lower_case: bool) -> SentencePieceTokenizer {
        SentencePieceTokenizer { vocab, _lower_case }
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

//impl Tokenizer<SentencePieceVocab> for SentencePieceTokenizer {
//    fn vocab(&self) -> &SentencePieceVocab {
//        &self.vocab
//    }
//
//    fn tokenize_to_tokens(&self, text: TokenRef) -> Vec<Token> {
//        let mut token = text.to_owned();
//        decompose_nfkc(&mut token);
//        token.text = token.text.replace(is_whitespace, "\u{2581}");
//        let output = self.vocab.decode_forward(token.text.as_str());
//        let decoded = self.vocab.decode_backward(&output);
////        for
//        vec!()
//    }
//}