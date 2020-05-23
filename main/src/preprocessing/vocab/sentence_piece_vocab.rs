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


use crate::preprocessing::vocab::sentencepiece_proto::sentencepiece_model::ModelProto;
use protobuf::parse_from_bytes;
use std::fs::File;
use std::io::Read;
use hashbrown::HashMap as BrownHashMap;
use itertools::Itertools;
use crate::Vocab;
use std::collections::HashMap;
use crate::preprocessing::vocab::base_vocab::swap_key_values;
use std::process;
use crate::preprocessing::tokenizer::base_tokenizer::{TokenRef, OffsetSize};


#[derive(Debug, Clone, Copy)]
pub struct Node<'a> {
    pub text: &'a str,
    pub score: f32,
    pub index: i64,
    pub start: usize,
    pub end: usize,
    pub reference_offsets: &'a [OffsetSize]
}

#[derive(Debug, Clone)]
pub struct TrieNode {
    pub text: String,
    pub len: usize,
    pub score: f32,
    pub index: i64,
    pub end: bool,
    pub children: BrownHashMap<char, TrieNode>,
}

impl TrieNode {
    pub fn new(text: String) -> TrieNode {
        let len = text.chars().count();
        TrieNode {
            text,
            len,
            score: 0.0,
            index: 0,
            end: false,
            children: BrownHashMap::new(),
        }
    }
}

pub struct SentencePieceVocab {
    pub root: TrieNode,
    pub values: HashMap<String, i64>,
    pub indices: HashMap<i64, String>,
    pub unknown_value: &'static str,
    pub special_values: HashMap<String, i64>,
    pub special_indices: HashMap<i64, String>,
}

impl SentencePieceVocab {
    pub fn pad_value() -> &'static str { "<pad>" }
    pub fn sep_value() -> &'static str { "<sep>" }
    pub fn cls_value() -> &'static str { "<cls>" }
    pub fn mask_value() -> &'static str { "<mask>" }
    pub fn bos_value() -> &'static str { "<s>" }
    pub fn eos_value() -> &'static str { "</s>" }

    pub fn from_proto(proto: &ModelProto) -> SentencePieceVocab {
        let root = TrieNode::new("".to_string());
        let values = HashMap::new();
        let indices = HashMap::new();
        let unknown_value = SentencePieceVocab::unknown_value();
        let special_values = HashMap::new();
        let special_indices = HashMap::new();

        let mut vocab = SentencePieceVocab { root, values, indices, unknown_value, special_values, special_indices };
        for (idx, piece) in proto.get_pieces().iter().enumerate() {
            vocab.insert(piece.get_piece(), piece.get_score(), idx as i64);
        }
        vocab.indices = swap_key_values(&vocab.values);
        SentencePieceVocab::_register_as_special_value(unknown_value, &vocab.values, &mut vocab.special_values);

        let pad_value = SentencePieceVocab::pad_value();
        SentencePieceVocab::_register_as_special_value(pad_value, &vocab.values, &mut vocab.special_values);

        let sep_value = SentencePieceVocab::sep_value();
        SentencePieceVocab::_register_as_special_value(sep_value, &vocab.values, &mut vocab.special_values);

        let cls_value = SentencePieceVocab::cls_value();
        SentencePieceVocab::_register_as_special_value(cls_value, &vocab.values, &mut vocab.special_values);

        let mask_value = SentencePieceVocab::mask_value();
        SentencePieceVocab::_register_as_special_value(mask_value, &vocab.values, &mut vocab.special_values);

        let bos_value = SentencePieceVocab::bos_value();
        SentencePieceVocab::_register_as_special_value(bos_value, &vocab.values, &mut vocab.special_values);

        let eos_value = SentencePieceVocab::eos_value();
        SentencePieceVocab::_register_as_special_value(eos_value, &vocab.values, &mut vocab.special_values);

        vocab.special_indices = swap_key_values(&vocab.special_values);

        vocab
    }

    fn insert(&mut self, word: &str, score: f32, index: i64) {
        let char_count = word.chars().count();
        let mut node = &mut self.root;
        self.values.insert(word.to_owned(), index);

        for (idx, character) in word.chars().enumerate() {
            if !node.children.contains_key(&character) {
                let mut text = node.text.clone();
                text.push(character);
                let new_node = TrieNode::new(text);
                node.children.insert(character, new_node);
            }
            node = node.children.get_mut(&character).unwrap();
            if idx == char_count - 1 {
                node.end = true;
                node.score = score;
                node.index = index;
            }
        }
    }

    pub fn common_prefix_search<'a>(&'a self, text: &'a str) -> Vec<&TrieNode> {
        let mut results = vec!();
        let mut characters = text.chars();

        let mut node = self.root.children.get(&characters.next().unwrap());
        if node.is_some() {
            if node.unwrap().end {
                results.push(node.unwrap());
            }
        } else {
            return vec!();
        }
        while let Some(character) = characters.next() {
            node = node.unwrap().children.get(&character);
            if node.is_some() {
                if node.unwrap().end {
                    results.push(node.unwrap());
                }
            } else {
                break;
            }
        }

        results
    }

    pub fn decode_forward_token_ref<'a>(&'a self, token: TokenRef<'a>) -> Vec<Option<Node<'a>>> {
        let mut char_positions = token.text
            .char_indices()
            .map(|(pos, _)| pos)
            .collect_vec();
        char_positions.push(token.text.len());
        let mut results = vec!(None; char_positions.len());
        let mut scores = vec!(std::f32::NEG_INFINITY; char_positions.len());
        scores[0] = 0f32;

        for char_start in 0..char_positions.len() - 1 {
            let matches = self.common_prefix_search(&token.text[char_positions[char_start]..]);
            for node in matches {
                let local_score = scores[char_start] + node.score;
                let char_end = char_start + node.len;
                if local_score > scores[char_end] {
                    results[char_end] = Some(Node {
                        text: &token.text[char_positions[char_start]..char_positions[char_end]],
                        score: local_score,
                        index: node.index,
                        start: char_start,
                        end: char_end,
                        reference_offsets: &token.reference_offsets[char_start..char_end]
                    });
                    scores[char_end] = local_score;
                }
            }
            if scores[char_start + 1] <= std::f32::MIN {
                results[char_start + 1] = Some(Node {
                    text: &token.text[char_positions[char_start]..char_positions[char_start + 1]],
                    score: std::f32::MIN,
                    index: 0,
                    start: char_start,
                    end: char_start + 1,
                    reference_offsets: &token.reference_offsets[char_start..char_start + 1]
                });
                scores[char_start + 1] = 0f32;
            }
        }
        results
    }

    pub fn decode_forward<'a>(&'a self, text: &'a str) -> Vec<Option<Node<'a>>> {
        let mut char_positions = text
            .char_indices()
            .map(|(pos, _)| pos)
            .collect_vec();
        char_positions.push(text.len());
        let mut results = vec!(None; char_positions.len());
        let mut scores = vec!(std::f32::NEG_INFINITY; char_positions.len());
        scores[0] = 0f32;

        for char_start in 0..char_positions.len() - 1 {
            let matches = self.common_prefix_search(&text[char_positions[char_start]..]);
            for node in matches {
                let local_score = scores[char_start] + node.score;
                let char_end = char_start + node.len;
                if local_score > scores[char_end] {
                    results[char_end] = Some(Node {
                        text: &text[char_positions[char_start]..char_positions[char_end]],
                        score: local_score,
                        index: node.index,
                        start: char_start,
                        end: char_end,
                        reference_offsets: &[]
                    });
                    scores[char_end] = local_score;
                }
            }
            if scores[char_start + 1] <= std::f32::MIN {
                results[char_start + 1] = Some(Node {
                    text: &text[char_positions[char_start]..char_positions[char_start + 1]],
                    score: std::f32::MIN,
                    index: 0,
                    start: char_start,
                    end: char_start + 1,
                    reference_offsets: &[]
                });
                scores[char_start + 1] = 0f32;
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

impl Vocab for SentencePieceVocab {
    fn unknown_value() -> &'static str { "<unk>" }

    fn get_unknown_value(&self) -> &'static str { "<unk>" }

    fn values(&self) -> &HashMap<String, i64> {
        &self.values
    }

    fn indices(&self) -> &HashMap<i64, String> {&self.indices}

    fn special_values(&self) -> &HashMap<String, i64> { &self.special_values }

    fn special_indices(&self) -> &HashMap<i64, String> {&self.special_indices}

    fn from_file(path: &str) -> SentencePieceVocab {
        let mut f = File::open(path).unwrap();
        let mut contents = Vec::new();
        f.read_to_end(&mut contents).unwrap();

        let proto = parse_from_bytes::<ModelProto>(contents.as_slice()).unwrap();
        SentencePieceVocab::from_proto(&proto)
    }

    fn token_to_id(&self, token: &str) -> i64 {
        match self._token_to_id(token, &self.values, &self.special_values, &self.unknown_value) {
            Ok(index) => index,
            Err(err) => {
                println!("{}", err);
                process::exit(1);
            }
        }
    }

    fn id_to_token(&self, id: &i64) -> String {
        match self._id_to_token(&id, &self.indices, &self.special_indices, &self.unknown_value) {
            Ok(token) => token,
            Err(err) => {
                println!("{}", err);
                process::exit(1);
            }
        }
    }
}