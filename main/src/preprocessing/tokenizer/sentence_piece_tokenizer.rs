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
use std::sync::Arc;

pub struct SentencePieceTokenizer {
    vocab: Arc<SentencePieceVocab>,
    _lower_case: bool,
}

impl SentencePieceTokenizer {
    pub fn from_file(path: &str, _lower_case: bool) -> SentencePieceTokenizer {
        let vocab = Arc::new(SentencePieceVocab::from_file(path));
        SentencePieceTokenizer { vocab, _lower_case }
    }

    pub fn from_existing_vocab(vocab: Arc<SentencePieceVocab>, _lower_case: bool) -> SentencePieceTokenizer {
        SentencePieceTokenizer { vocab, _lower_case }
    }

    pub fn vocab(&self) -> &SentencePieceVocab {
        self.vocab.as_ref()
    }
}