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


use radix_trie::Trie;
use crate::preprocessing::vocab::sentencepiece_proto::sentencepiece_model::ModelProto;
use protobuf::parse_from_bytes;
use std::fs::File;
use std::io::Read;
use itertools::Itertools;

#[derive(Debug, Clone, Copy)]
pub struct Node<'a> {
    text: &'a str,
    score: f32,
    index: i64,
    start: usize,
    end: usize,
}

pub struct SentencePieceVocab {
    trie: Trie<String, (f32, i64)>
}

impl SentencePieceVocab {
    pub fn from_file(path: &str) -> SentencePieceVocab {
        let mut f = File::open(path).unwrap();
        let mut contents = Vec::new();
        f.read_to_end(&mut contents).unwrap();

        let proto = parse_from_bytes::<ModelProto>(contents.as_slice()).unwrap();
        SentencePieceVocab::from_proto(&proto)
    }

    pub fn from_proto(proto: &ModelProto) -> SentencePieceVocab {
        let mut trie = Trie::new();
        for (idx, piece) in proto.get_pieces().iter().enumerate() {
            trie.insert(piece.get_piece().to_owned(), (piece.get_score(), idx as i64));
        }
        SentencePieceVocab { trie }
    }

//    pub fn common_prefix_search<'a>(&'a self, text: &'a str) -> Vec<Node> {
//        let mut results = vec!();
//        let mut char_positions = text.char_indices().map(|(pos, _)| pos).collect_vec();
//        char_positions.push(text.len());
//        for &i in char_positions.iter() {
//            if let Some(sub_trie) = self.trie.get(&text[..i]) {
//                results.push(Node {
//                    text: &text[..i],
//                    score: sub_trie.0,
//                    index: sub_trie.1,
//                });
//            };
//        }
//        results
//    }

    pub fn decode_forward<'a>(&'a self, text: &'a str) -> Vec<Option<Node>> {
        let mut char_positions = text
            .char_indices()
            .map(|(pos, _)| pos)
            .collect_vec();
        char_positions.push(text.len());
        let mut results = vec!(None; char_positions.len());
        let mut scores = vec!(std::f32::MIN; char_positions.len());
        scores[0] = 0f32;

        for char_end in 0..char_positions.len() {
            for char_start in 0..char_end {
                let sub_text = &text[char_positions[char_start]..char_positions[char_end]];
                if let Some(sub_trie) = self.trie.get(sub_text) {
                    let local_score = scores[char_start] + sub_trie.0;
                    if local_score > scores[char_end] {
                        results[char_end] = Some(Node {
                            text: sub_text,
                            score: local_score,
                            index: sub_trie.1,
                            start: char_start,
                            end: char_end,
                        });
                        scores[char_end] = local_score;
                    }
                };
            }
        }
        results
    }

    pub fn decode_backward<'a>(&'a self, nodes: &'a Vec<Option<Node<'a>>>) -> Vec<&'a Node> {
        let mut next_node = nodes.last().unwrap();
        let mut best_sequence = vec!();

        while next_node.is_some() {
            let node_value = next_node.as_ref().unwrap();
            best_sequence.push(node_value);
            next_node = &nodes[node_value.start];
        };
        best_sequence.reverse();
        best_sequence
    }
}