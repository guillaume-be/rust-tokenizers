// Copyright 2016 Google LLC. All Rights Reserved.
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
use crate::tokenizer::tokenization_utils::{is_punctuation, is_whitespace};
use crate::vocab::sentencepiece_proto::sentencepiece_model::ModelProto;
use crate::{Mask, Offset, OffsetSize, Token, TokenRef};
use hashbrown::HashMap as BrownHashMap;
use itertools::Itertools;
use protobuf::Message;
use std::fs::File;
use std::io::Read;

#[derive(Debug, Clone, Copy)]
pub struct Node<'a> {
    pub text: &'a str,
    pub score: f32,
    pub index: i64,
    pub start: usize,
    pub end: usize,
    pub reference_offsets: &'a [OffsetSize],
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

/// # SentencePiece Model
/// Model for SentencePiece tokenizer. Contains the following special values. This model performs
/// the SentencePiece unigram decomposition. As such, it contains a `Trie` data structure for efficient
/// common prefix search.
///
/// Expects a SentencePiece protobuf file when created from file.
#[derive(Debug, Clone)]
pub struct SentencePieceModel {
    /// Trie data structure containing the vocabulary elements and their unigram log-probabilities
    pub root: TrieNode,
}

impl SentencePieceModel {
    /// Creates a SentencePiece Model from a protobuf file.
    ///
    /// # Example
    /// ```no_run
    /// use rust_tokenizers::vocab::SentencePieceModel;
    /// let path = "path/to/spiece.model";
    ///
    /// let sentence_piece_model = SentencePieceModel::from_file(path);
    /// ```
    pub fn from_file(path: &str) -> Result<SentencePieceModel, TokenizerError> {
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
        let root = TrieNode::new("".to_string());
        let mut vocab = SentencePieceModel { root };
        for (idx, piece) in proto.get_pieces().iter().enumerate() {
            vocab.insert(piece.get_piece(), piece.get_score(), idx as i64);
        }
        Ok(vocab)
    }

    fn insert(&mut self, word: &str, score: f32, index: i64) {
        let char_count = word.chars().count();
        let mut node = &mut self.root;

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

    /// Performs a common prefix search for a given query on the model Trie structure
    ///
    /// # Arguments
    /// - text (`&str`): query to find common prefixes from
    ///
    /// # Returns
    /// - `Vec<&TrieNode>` containing references to the Trie nodes with a common (character based) prefix with the query
    ///
    /// # Example
    /// ```no_run
    /// use rust_tokenizers::vocab::SentencePieceModel;
    /// let path = "path/to/spiece.model";
    /// let sentence_piece_model = SentencePieceModel::from_file(path).unwrap();
    ///
    /// let query = "hello";
    /// let common_prefixes = sentence_piece_model.common_prefix_search(query);
    /// ```
    pub fn common_prefix_search<'a>(&'a self, text: &'a str) -> Vec<&'a TrieNode> {
        let mut results = vec![];
        let mut characters = text.chars();
        let mut node = self.root.children.get(match &characters.next() {
            Some(character) => character,
            None => {
                return vec![];
            }
        });
        if let Some(node_value) = node {
            if node_value.end {
                results.push(node_value);
            }
        } else {
            return vec![];
        }
        for character in characters {
            node = node.unwrap().children.get(&character);
            if let Some(node_value) = node {
                if node_value.end {
                    results.push(node_value);
                }
            } else {
                break;
            }
        }

        results
    }

    /// Decodes a `TokenRef` to a lattice of potential subtokens.
    /// This step is usually followed by a backward step to find the most likely sequence.
    ///
    /// # Arguments
    /// - token (`TokenRef<'a>`): token to decompose in sub-tokens
    ///
    /// # Returns
    /// - `Vec<Option<Node<'a>>>` vector of lattice nodes. The string for the nodes references back to the original token.
    ///
    /// # Example
    /// ```no_run
    /// use rust_tokenizers::vocab::SentencePieceModel;
    /// use rust_tokenizers::TokenRef;
    /// let path = "path/to/spiece.model";
    /// let sentence_piece_model = SentencePieceModel::from_file(path).unwrap();
    ///
    /// let token = TokenRef::new("hello", &[0, 1, 2, 3]);
    /// let lattice_nodes = sentence_piece_model.decode_forward_token_ref(token);
    /// ```
    pub fn decode_forward_token_ref<'a>(&'a self, token: TokenRef<'a>) -> Vec<Option<Node<'a>>> {
        let mut char_positions = token.text.char_indices().map(|(pos, _)| pos).collect_vec();
        char_positions.push(token.text.len());
        let mut results = vec![None; char_positions.len()];
        let mut scores = vec![f32::NEG_INFINITY; char_positions.len()];
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
                        reference_offsets: &token.reference_offsets[char_start..char_end],
                    });
                    scores[char_end] = local_score;
                }
            }
            if scores[char_start + 1] <= f32::MIN {
                results[char_start + 1] = Some(Node {
                    text: &token.text[char_positions[char_start]..char_positions[char_start + 1]],
                    score: f32::MIN,
                    index: 0,
                    start: char_start,
                    end: char_start + 1,
                    reference_offsets: &token.reference_offsets[char_start..char_start + 1],
                });
                scores[char_start + 1] = 0f32;
            }
        }
        results
    }

    /// Backward pass through an array of nodes (generated as a result of the forward pass), returning
    /// the most likely sequence of nodes. These are usually converted back to tokens in a last step
    ///
    /// # Arguments
    /// - nodes (`&'a [Option<Node<'a>>]`): possible modes generated from the forward step
    ///
    /// # Returns
    /// - `Vec<&'a Node>` sequence of most likely nodes
    ///
    /// # Example
    /// ```no_run
    /// use rust_tokenizers::vocab::SentencePieceModel;
    /// use rust_tokenizers::TokenRef;
    /// let path = "path/to/spiece.model";
    /// let sentence_piece_model = SentencePieceModel::from_file(path).unwrap();
    ///
    /// let token = TokenRef::new("hello", &[0, 1, 2, 3]);
    /// let lattice_nodes = sentence_piece_model.decode_forward_token_ref(token);
    /// let best_nodes_sequence = sentence_piece_model.decode_backward(&lattice_nodes);
    /// ```
    pub fn decode_backward<'a>(&'a self, nodes: &'a [Option<Node<'a>>]) -> Vec<&'a Node> {
        let mut best_sequence = vec![];
        let mut next_node = match nodes.last() {
            Some(value) => value,
            None => {
                return best_sequence;
            }
        };

        while next_node.is_some() {
            let node_value = next_node.as_ref().unwrap();
            best_sequence.push(node_value);
            next_node = &nodes[node_value.start];
        }
        best_sequence.reverse();
        best_sequence
    }

    /// Convert the most likely node sequences to a vector of tokens that can be further processed
    /// by the tokenizer.
    ///
    /// # Arguments
    /// - nodes (`Vec<&Node>`): sequence of most likely nodes
    ///
    /// # Returns
    /// - `Vec<Token>` sequence of most likely sub-tokens
    ///
    /// # Example
    /// ```no_run
    /// use rust_tokenizers::vocab::SentencePieceModel;
    /// use rust_tokenizers::TokenRef;
    /// let path = "path/to/spiece.model";
    /// let sentence_piece_model = SentencePieceModel::from_file(path).unwrap();
    ///
    /// let token = TokenRef::new("hello", &[0, 1, 2, 3]);
    /// let lattice_nodes = sentence_piece_model.decode_forward_token_ref(token);
    /// let best_nodes_sequence = sentence_piece_model.decode_backward(&lattice_nodes);
    /// let sub_tokens = sentence_piece_model.parse_nodes_to_tokens(best_nodes_sequence);
    /// ```
    pub fn parse_nodes_to_tokens(&self, nodes: Vec<&Node>) -> Vec<Token> {
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
                    mask: Mask::Unknown,
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
        self.populate_masks(output.as_mut_slice(), '\u{2581}');
        output
    }

    /// Populates the `mask` field for a sequence of sub-tokens generated by a SentencePiece model.
    /// These masks are not generated as part of the standard unigram decomposition and must be added
    /// afterwards. Mutates the tokens in-place.
    ///
    /// # Arguments
    /// - tokens (`&mut [Token]`): tokens to get the masks from
    /// - whitespace_char (`char`): whitespace character to identify whether a token is a continuation token or not.
    ///
    /// # Example
    /// ```no_run
    /// use rust_tokenizers::vocab::SentencePieceModel;
    /// use rust_tokenizers::TokenRef;
    /// let path = "path/to/spiece.model";
    /// let sentence_piece_model = SentencePieceModel::from_file(path).unwrap();
    ///
    /// let token = TokenRef::new("hello", &[0, 1, 2, 3]);
    /// let lattice_nodes = sentence_piece_model.decode_forward_token_ref(token);
    /// let best_nodes_sequence = sentence_piece_model.decode_backward(&lattice_nodes);
    /// let mut sub_tokens = sentence_piece_model.parse_nodes_to_tokens(best_nodes_sequence);
    /// let sub_tokens_with_masks = sentence_piece_model.populate_masks(&mut sub_tokens, ' ');
    /// ```
    pub fn populate_masks(&self, tokens: &mut [Token], whitespace_token: char) {
        let mut previous_mask = Mask::None;
        for token in tokens {
            if token.text.chars().count() == 1 {
                let first_char = match token.text.chars().last() {
                    Some(value) => value,
                    None => {
                        token.mask = Mask::Unknown;
                        previous_mask = Mask::Unknown;
                        continue;
                    }
                };
                if is_punctuation(&first_char) {
                    token.mask = Mask::Punctuation;
                    previous_mask = Mask::Punctuation;
                    continue;
                }
                if is_whitespace(&first_char) {
                    token.mask = Mask::Whitespace;
                    previous_mask = Mask::Punctuation;
                    continue;
                }
            }
            if !token.text.starts_with(whitespace_token)
                & !(previous_mask == Mask::Punctuation)
                & !(previous_mask == Mask::Whitespace)
            {
                token.mask = Mask::Continuation;
                previous_mask = Mask::Continuation;
            } else {
                previous_mask = Mask::None;
            }
        }
    }
}
