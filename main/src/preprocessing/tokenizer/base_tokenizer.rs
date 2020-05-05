// Copyright 2018 The Open AI Team Authors, The Google AI Language Team Authors
// Copyright 2018 The HuggingFace Inc. team.
// Copyright 2019 Guillaume Becquin
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use crate::preprocessing::vocab::base_vocab::Vocab;
use crate::preprocessing::tokenizer::tokenization_utils::{tokenize_cjk_chars, whitespace_tokenize, strip_accents, split_on_punct, split_on_special_tokens, clean_text, truncate_sequences};
use std::sync::Arc;
use std::borrow::{ToOwned,Borrow};
use std::convert::AsRef;
use rayon::prelude::*;
use itertools::Itertools;
use unzip_n::unzip_n;

unzip_n!(3);

pub enum TruncationStrategy {
    LongestFirst,
    OnlyFirst,
    OnlySecond,
    DoNotTruncate,
}

pub type OffsetSize = u32;

#[derive(Debug, PartialEq, PartialOrd, Clone)]
///Offset information (in unicode points) to relate a token back to its original input string
pub struct Offset {
    pub begin: OffsetSize,
    pub end: OffsetSize,
}

#[derive(Debug, PartialEq, PartialOrd, Clone, Copy)]
pub enum Mask {
    ///The token has no particular mask. This is the default situation. It may indicate that further processing can be done on a token.
    None,
    ///the token represents a whitespace (in any shape or form)
    Whitespace,
    ///the token represents punctuation (in any shape or form)
    Punctuation,
    ///the token represents a single Chinese/Japanese/Korean character (including kana and hangul)
    CJK,
    ///the token is a special marker (such as a seperator marker, a class marker, etc)
    Special,
    ///the token is the begin in a series of subtokens, the offset refers specifically to the subtoken. Subsequent tokens in this sequence will carry the 'Continuation' mask
    Begin,
    ///the token is the continuation of the previous token, the offset refers specifically to the subtoken. All but the first subtoken in a sequence carry this mask (the first carries 'Begin'). (this is the reverse of Mask::Unfinished)
    Continuation,
    ///the token is the start of a token but not finished yet. All but the last subtoken in the a token sequence carry this mask. This is the reverse of Mask::Continuation.
    Unfinished,
    ///This is a a subtoken that a part of a larger token, the offsets, however, refer to the entire token rather than to the part. All subtokens in the sequence will refer to the same offsets. This is the first token in such a sequence.
    InexactBegin,
    ///This is a a subtoken that a part of a larger token, the offsets, however, refer to the entire token rather than to the part. All subtokens in the sequence will refer to the same offsets. This is a continuation token in such a sequence.
    InexactContinuation,
    ///The token is out of vocabulary, it is unknown by the tokeniser and it will decode to unknown. Tokens that can be decoded properly (but may still be out of vocabulary) should not set this.
    Unknown
}

impl Default for Mask {
    fn default() -> Mask {
        Mask::None
    }
}

#[derive(Debug, PartialEq)]
///A token that references the original text
pub struct TokenRef<'a> {
    pub text: &'a str,
    pub offset: Offset,
    pub mask: Mask,
}

impl<'a> TokenRef<'a> {
    pub fn new(text: &'a str) -> TokenRef<'a> {
        TokenRef {
            text: text,
            offset: Offset { begin: 0, end: text.chars().count() as u32},
            mask: Mask::None,
        }
    }

    pub fn owned_token(self) -> Token { //this is still ugly and should probably be done using Borrow or AsRef in some way
        Token {
            text: self.text.to_owned(),
            offset: self.offset,
            mask: self.mask
        }
    }
}

/*
impl<'a> ToOwned for TokenRef<'a> {
    type Owned = Token;
    fn to_owned(&self) -> Self::Owned {
        Token {
            text: self.text.to_owned(),
            offset: self.offset,
            mask: self.mask,
        }
    }
}

impl<'a> Borrow<TokenRef<'a>> for Token {
    fn borrow(&'a self)  -> &'a TokenRef<'a> {
        TokenRef {
            text: &self.text,
            offset: self.offset,
            mask: self.mask
        }
    }
}
*/

#[derive(Debug, PartialEq, Clone)]
///A token that references the original text
///An owned token
pub struct Token {
    pub text: String,
    pub offset: Offset,
    pub mask: Mask,
}



impl Token {
    pub fn new(text: String) -> Token {
        let textsize: OffsetSize = text.chars().count() as OffsetSize;
        Token {
            text: text,
            offset: Offset { begin: 0, end: textsize },
            mask: Mask::None,
        }
    }

    pub fn token_ref<'a>(&'a self) -> TokenRef<'a> { //this is still ugly and should probably be done using Borrow or AsRef in some way
        TokenRef {
            text: self.text.as_str(),
            offset: self.offset.clone(),
            mask: self.mask.clone()
        }
    }
}

impl Offset {
    pub fn new(begin: OffsetSize,end: OffsetSize) -> Offset {
        Offset { begin, end }
    }

    pub fn to_option(self) -> Option<Offset> {
        if self.end > self.begin {
            Some(self)
        } else {
            None
        }
    }
}

#[derive(Debug, PartialEq, PartialOrd, Clone)]
pub struct TokenizedInput {
    ///Vector of token IDs
    pub token_ids: Vec<i64>,

    ///Vector segments ids, segments are seperated with a [SEP] marker, each increments the segment ID. This vector has the same length as token_ids.
    pub segment_ids: Vec<i8>,

    ///Flags tokens as special tokens (1) or not (0). This vector has the same length as token_ids.
    pub special_tokens_mask: Vec<i8>,

    pub overflowing_tokens: Vec<i64>,
    pub num_truncated_tokens: usize,

    ///Offset information in relation to the original text. Tokens that can not be related to the
    ///original source are registered as None.
    pub token_offsets: Vec<Option<Offset>>,

    ///Masks tokens so you can see what type of token something is. This vector has the same length
    ///as token_ids (and also makes special_tokens_mask redundant).
    pub mask: Vec<Mask>,
}

pub trait Tokenizer<T: Vocab> {
    fn vocab(&self) -> &T;

    ///Tokenize a string, returns a vector of tokens as strings.
    ///Use `tokenize_with_offsets` or `tokenize_to_tokens` if you also want offset information.
    fn tokenize(&self, text: &str) -> Vec<String> {
        self.tokenize_with_offsets(text).0
    }

    ///Tokenize a string, return offset information
    fn tokenize_with_offsets<'a>(&self, text: &'a str) -> (Vec<String>,Vec<Offset>,Vec<Mask>) {
        let initial_token: TokenRef<'a> = TokenRef::new(text);
        self.tokenize_to_tokens(initial_token).into_iter().map(|token| (token.text, token.offset, token.mask)).unzip_n_vec()
    }

    ///Tokenize a text, returns a vector of tokens (contains offset information and more)
    fn tokenize_to_tokens<'a>(&self, text: TokenRef<'a>) -> Vec<Token>;

    ///Tokenize a vector of strings, where each corresponds to for example a sentence, returns a vector of vectors of strings.
    ///Use `tokenize_list_with_offsets` if you also want offset information.
    fn tokenize_list(&self, text_list: Vec<&str>) -> Vec<Vec<String>> {
        text_list.
            into_iter().
            map(|text| self.tokenize(text)).
            collect()
    }

    ///Tokenize a vector of strings, where each corresponds to for example a sentence, returns a vector of pairs consists of a vector of tokens and a list of offset information.
    fn tokenize_list_with_offsets(&self, text_list: Vec<&str>) -> Vec<(Vec<String>,Vec<Offset>,Vec<Mask>)> {
        text_list.
            into_iter().
            map(|text| self.tokenize_with_offsets(text)).
            collect()
    }

    fn convert_tokens_to_ids(&self, tokens: &Vec<String>) -> Vec<i64> {
        tokens.into_iter().map(|v| self.vocab().token_to_id(v)).collect()
    }

    fn encode(&self, text_1: &str, text_2: Option<&str>, max_len: usize, truncation_strategy: &TruncationStrategy, stride: usize) -> TokenizedInput {
        let (token_strings, token_offsets, token_mask) = self.tokenize_with_offsets(text_1);
        let token_ids_1 = self.convert_tokens_to_ids(&token_strings);
        let len_1 = token_ids_1.len();
        let (token_ids_2, token_offsets_2, token_mask_2, len_2, pair) = {
            if let Some(text) = text_2 {
                let (token_strings_2, token_offsets_2, token_mask_2) = self.tokenize_with_offsets(text);
                let token_ids_2: Vec<i64> = self.convert_tokens_to_ids(&token_strings_2);
                let len_2 = token_ids_2.len();
                (Some(token_ids_2), Some(token_offsets_2), Some(token_mask_2), len_2, Some(vec!()))
            } else {
                (None, None, None, 0, None)
            }
        };
        let (additional_tokens, _, _, _additional_offsets, _additional_mask) = self.build_input_with_special_tokens(vec!(), pair, vec!(), Some(vec!()), vec!(), Some(vec!()));
        let total_len = len_1 + len_2 + additional_tokens.len();
        let num_truncated_tokens = if total_len > max_len { total_len - max_len } else { 0 };
        let (token_ids_1,
            token_ids_2,
            token_offsets,
            token_offsets_2,
            token_mask,
            token_mask_2,
            overflowing_tokens, _overflowing_offsets) = truncate_sequences(token_ids_1,
                                                         token_ids_2,
                                                         token_offsets,
                                                         token_offsets_2,
                                                         token_mask,
                                                         token_mask_2,
                                                         num_truncated_tokens,
                                                         truncation_strategy,
                                                         stride).unwrap();

        let (token_ids, segment_ids, special_tokens_mask, token_offsets, token_mask) = self.build_input_with_special_tokens(token_ids_1, token_ids_2, token_offsets, token_offsets_2, token_mask, token_mask_2);

        TokenizedInput { token_ids, segment_ids, special_tokens_mask, overflowing_tokens, num_truncated_tokens, token_offsets, mask: token_mask }
    }

    fn encode_list(&self, text_list: Vec<&str>, max_len: usize, truncation_strategy: &TruncationStrategy, stride: usize) -> Vec<TokenizedInput> {
        text_list
            .into_iter()
            .map(|text| self.encode(text, None, max_len, truncation_strategy, stride))
            .collect()
    }

    fn encode_pair_list(&self, text_list: Vec<(&str, &str)>, max_len: usize, truncation_strategy: &TruncationStrategy, stride: usize) -> Vec<TokenizedInput> {
        text_list
            .into_iter()
            .map(|text| self.encode(text.0, Some(text.1), max_len, truncation_strategy, stride))
            .collect()
    }

    fn decode_to_vec(&self, token_ids: Vec<i64>, skip_special_tokens: bool) -> Vec<String> {
        let tokens: Vec<String> = if skip_special_tokens {
            token_ids
                .iter()
                .filter(|id| !self.vocab().special_indices().contains_key(id))
                .map(|id| { self.vocab().id_to_token(id) })
                .collect_vec()
        } else {
            token_ids
                .iter()
                .map(|id| { self.vocab().id_to_token(id) })
                .collect_vec()
        };
        tokens
    }

    ///Converts a sequence of ids (integer) into  astring, using the tokenizer and vocabulary
    ///  with options to remove special tokens and clean up tokenization spaces.
    ///  Args:
    ///   * token_ids: list of tokenized input ids. Can be obtained using the `encode` or `encode_plus` methods.
    ///   * skip_special_tokens: if set to True, will replace special tokens.
    ///   * clean_up_tokenization_spaces: if set to True, will clean up the tokenization spaces.
    fn decode(&self, token_ids: Vec<i64>, skip_special_tokens: bool, clean_up_tokenization_spaces: bool) -> String {
        let tokens = self.decode_to_vec(token_ids, skip_special_tokens);
        let decoded_string = self.convert_tokens_to_string(tokens);
        if clean_up_tokenization_spaces {
            self.clean_up_tokenization(decoded_string)
        } else {
            decoded_string
        }
    }

    fn convert_tokens_to_string(&self, tokens: Vec<String>) -> String {
        tokens.join(" ")
    }

    fn clean_up_tokenization(&self, input_string: String) -> String {
        input_string
            .replace(" .", ".")
            .replace(" !", "!")
            .replace(" ?", "?")
            .replace(" ,", ",")
            .replace(" ' ", "'")
            .replace(" n't", "n't")
            .replace(" 'm'", "'m")
            .replace(" do not", " don't")
            .replace(" 's", "'s")
            .replace(" 've", "'ve")
            .replace(" 're", "'re")
    }

    fn decode_list(&self, token_ids_list: Vec<Vec<i64>>, skip_special_tokens: bool, clean_up_tokenization_spaces: bool) -> Vec<String> {
        token_ids_list
            .into_iter()
            .map(|token_ids| self.decode(token_ids, skip_special_tokens, clean_up_tokenization_spaces))
            .collect()
    }


    /// Build model inputs from a sequence or a pair of sequence for sequence classification tasks
    /// by concatenating and adding special tokens.
    /// A RoBERTa sequence has the following format:
    /// single sequence: <s> X </s>
    /// pair of sequences: <s> A </s></s> B </s>
    ///
    /// Returns a tuple of:
    ///  * output token IDs
    ///  * token segment IDs
    ///  * special token mask
    ///  * offsets (as a vector of `Option<Offset>` because some added markers may not have associated offsets
    ///  * token mask
    fn build_input_with_special_tokens(&self, mut tokens_1: Vec<i64>, tokens_2: Option<Vec<i64>>, offsets_1: Vec<Offset>, offsets_2: Option<Vec<Offset>>, mut mask: Vec<Mask>, mask_2: Option<Vec<Mask>>) -> (Vec<i64>, Vec<i8>, Vec<i8>, Vec<Option<Offset>>, Vec<Mask>) {
        let mut token_segment_ids: Vec<i8> = vec![0; tokens_1.len()];
        let mut special_tokens_mask: Vec<i8> = vec![0; tokens_1.len()];
        let mut offsets: Vec<Option<Offset>> = offsets_1.into_iter().map(|offset| offset.to_option() ).collect();
        let output = match tokens_2 {
            Some(tokens) => {
                let length = tokens.len();
                token_segment_ids.extend(vec![1; length]);
                special_tokens_mask.extend(vec![0; length]);
                tokens_1.extend(tokens);
                if let Some(offsets_2) = offsets_2 {
                    offsets.extend(offsets_2.into_iter().map(|offset| offset.to_option()).collect::<Vec<Option<Offset>>>());
                } else {
                    offsets.extend(vec![None; length]);
                }
                if let Some(mask_2) = mask_2 {
                    mask.extend(mask_2)
                } else {
                    mask.extend(vec![Mask::None; length]);
                }
                tokens_1
            }
            None => tokens_1
        };
        (output, token_segment_ids, special_tokens_mask, offsets, mask)
    }
}

pub trait MultiThreadedTokenizer<T: Vocab>
    where Self: std::marker::Sync + Send + Tokenizer<T> {
    fn vocab(&self) -> &T
    {
        Tokenizer::<T>::vocab(self)
    }

    fn tokenize_list_with_offsets(&self, text_list: Vec<&str>) -> Vec<(Vec<String>,Vec<Offset>,Vec<Mask>)> {
        text_list.
            par_iter().
            map(|text| self.tokenize_with_offsets(text)).
            collect()
    }

    fn tokenize_list(&self, text_list: Vec<&str>) -> Vec<Vec<String>> {
        text_list.
            par_iter().
            map(|text| self.tokenize(text)).
            collect()
    }

    fn encode_list(&self, text_list: Vec<&str>, max_len: usize, truncation_strategy: &TruncationStrategy, stride: usize) -> Vec<TokenizedInput> {
        text_list
            .par_iter()
            .map(|text| self.encode(text, None, max_len, truncation_strategy, stride))
            .collect()
    }

    fn encode_pair_list(&self, text_list: Vec<(&str, &str)>, max_len: usize, truncation_strategy: &TruncationStrategy, stride: usize) -> Vec<TokenizedInput> {
        text_list
            .par_iter()
            .map(|text| self.encode(text.0, Some(text.1), max_len, truncation_strategy, stride))
            .collect()
    }

    fn decode_list(&self, token_ids_list: Vec<Vec<i64>>, skip_special_tokens: bool, clean_up_tokenization_spaces: bool) -> Vec<String> {
        token_ids_list
            .par_iter()
            .map(|token_ids| self.decode(token_ids.to_vec(), skip_special_tokens, clean_up_tokenization_spaces))
            .collect()
    }
}


pub struct BaseTokenizer<T: Vocab> {
    vocab: Arc<T>,
    lower_case: bool,
    strip_accents: bool,
}

impl<T: Vocab + Sync + Send> BaseTokenizer<T> {
    pub fn from_file(path: &str, lower_case: bool, strip_accents: bool) -> BaseTokenizer<T> {
        let vocab = T::from_file(path);
        BaseTokenizer { vocab: Arc::new(vocab), lower_case, strip_accents }
    }

    pub fn from_existing_vocab(vocab: Arc<T>, lower_case: bool, strip_accents: bool) -> BaseTokenizer<T> {
        BaseTokenizer { vocab, lower_case, strip_accents }
    }
}

impl<T: Vocab + Sync + Send> Tokenizer<T> for BaseTokenizer<T> {
    fn vocab(&self) -> &T {
        &self.vocab
    }

    fn tokenize_to_tokens<'a>(&self, initial_token: TokenRef<'a>) -> Vec<Token> {
        //split on whitespace
        let tokens: Vec<Token> = whitespace_tokenize(initial_token).into_iter()
            .map(|token| {
                //split on special tokens
                split_on_special_tokens(token, self.vocab.as_ref())
            })
            .flatten()
            .map(|token| {
                //split on punctuation (with care for maintaining special values)
                split_on_punct(token)
            })
            .flatten()
            .map(|token| {
                //tokenize CJK characters so each character is one token
                tokenize_cjk_chars(token)
            })
            .flatten()
            .map(|token| {
                // v-- this is where the token gets owned, all steps above handle TokenRefs (dealing with &str)
                let mut token = Token {
                    text: clean_text(token.text, true),
                    offset: token.offset,
                    mask: token.mask
                };
                if token.mask != Mask::Special && token.mask != Mask::Unknown {
                    //apply the necessary transformations to the actual tokens (unless it's a special value)
                    if self.lower_case {
                        token.text = token.text.to_lowercase();
                    }
                    if self.strip_accents {
                        token.text = strip_accents(token.text);
                    }
                }
                token
            })
            .collect();

        tokens
   }

}

impl<T: Vocab + Sync + Send> MultiThreadedTokenizer<T> for BaseTokenizer<T> {}

//==============================
// Unit tests
//==============================
#[cfg(test)]
mod tests {
    use super::*;
    use crate::BertVocab;
    use std::collections::HashMap;
    use crate::preprocessing::vocab::base_vocab::swap_key_values;

    fn generate_test_vocab() -> BertVocab {
        let values: HashMap<String, i64> = [
            ("hello".to_owned(), 0),
            ("world".to_owned(), 1),
            ("[UNK]".to_owned(), 2),
            ("!".to_owned(), 3),
            ("[CLS]".to_owned(), 4),
            ("[SEP]".to_owned(), 5),
            ("[MASK]".to_owned(), 6),
            ("中".to_owned(), 7),
            ("华".to_owned(), 8),
            ("人".to_owned(), 9),
            ("[PAD]".to_owned(), 10),
            ("una".to_owned(), 11),
            ("##ffa".to_owned(), 12),
            ("##ble".to_owned(), 13)
        ].iter().cloned().collect();

        let special_values: HashMap<String, i64> = [
            ("[UNK]".to_owned(), 2),
            ("[CLS]".to_owned(), 4),
            ("[SEP]".to_owned(), 5),
            ("[MASK]".to_owned(), 6),
            ("[PAD]".to_owned(), 10)
        ].iter().cloned().collect();

        let indices = swap_key_values(&values);
        let special_indices = swap_key_values(&special_values);

        BertVocab { values, indices, unknown_value: "[UNK]", special_values, special_indices }
    }

    #[test]
    fn test_base_tokenizer() {
//        Given
        let vocab = Arc::new(generate_test_vocab());
        let base_tokenizer: BaseTokenizer<BertVocab> = BaseTokenizer::from_existing_vocab(vocab, true, true);
        let test_tuples = [
            (
                "Sentence with [MASK] token.",
                (vec!("sentence", "with", "[MASK]", "token", "."),
                 vec!(Offset::new(0,8), Offset::new(9,13), Offset::new(14,20), Offset::new(21,26), Offset::new(26,27)),
                 vec!(Mask::None, Mask::None, Mask::Special, Mask::None, Mask::Punctuation))
            ),
            (
                "[CLS]",
                (vec!("[CLS]"),
                 vec!(Offset::new(0,5)),
                 vec!(Mask::Special))
            ),
            (
                "[CLS] [PAD]",
                (vec!("[CLS]", "[PAD]"),
                 vec!(Offset::new(0,5), Offset::new(6,11)),
                 vec!(Mask::Special, Mask::Special))
            ),
            (
                "[CLS]       [PAD]",
                (vec!("[CLS]", "[PAD]"),
                 vec!(Offset::new(0,5), Offset::new(12,17)),
                 vec!(Mask::Special, Mask::Special))
            ),
            (
                "asdf",
                (vec!("asdf"),
                 vec!(Offset::new(0,4)),
                 vec!(Mask::None))
            ),
            (
                "",
                (vec!(),vec!(),vec!()),
            ),
            (
                "Allons, Flipote, allons; que d'eux je me délivre.",
                (vec!("allons", ",", "flipote", ",", "allons", ";", "que", "d", "\'", "eux", "je", "me", "delivre", "."),
                 vec!(
                     Offset { begin: 0, end: 6 }, Offset { begin: 6, end: 7 }, Offset { begin: 8, end: 15 }, Offset { begin: 15, end: 16 }, Offset { begin: 17, end: 23 }, Offset { begin: 23, end: 24 }, Offset { begin: 25, end: 28 }, Offset { begin: 29, end: 30 }, Offset { begin: 30, end: 31 }, Offset { begin: 31, end: 34 }, Offset { begin: 35, end: 37 }, Offset { begin: 38, end: 40 }, Offset { begin: 41, end: 48 }, Offset { begin: 48, end: 49 }
                     ),
                 vec!(Mask::None, Mask::Punctuation, Mask::None, Mask::Punctuation, Mask::None, Mask::Punctuation, Mask::None, Mask::None, Mask::Punctuation, Mask::None, Mask::None, Mask::None, Mask::None, Mask::Punctuation)  ),
            ),
            (
                "[UNK]中华人民共和国 [PAD] asdf",
                (vec!("[UNK]", "中", "华", "人", "民", "共", "和", "国", "[PAD]", "asdf"),
                 vec!(Offset { begin: 0, end: 5 }, Offset { begin: 5, end: 6 }, Offset { begin: 6, end: 7 }, Offset { begin: 7, end: 8 }, Offset { begin: 8, end: 9 }, Offset { begin: 9, end: 10 }, Offset { begin: 10, end: 11 }, Offset { begin: 11, end: 12 }, Offset { begin: 13, end: 18 }, Offset { begin: 19, end: 23 }),
                 vec!(Mask::Unknown, Mask::CJK, Mask::CJK, Mask::CJK, Mask::CJK, Mask::CJK, Mask::CJK, Mask::CJK, Mask::Special, Mask::None)
                 ),
            )
        ];
        let source_texts: Vec<&str> = test_tuples.iter().map(|v| v.0).collect();

//        When & Then
        for (source_text, expected_result) in test_tuples.iter() {
            let (tokens, offsets, mask) = base_tokenizer.tokenize_with_offsets(*source_text);
            let tokens: Vec<&str> = tokens.iter().map(|t|t.as_str()).collect();
            assert_eq!(tokens, expected_result.0);
            assert_eq!(offsets, expected_result.1);
            assert_eq!(mask, expected_result.2);
        }

        let results = Tokenizer::tokenize_list_with_offsets(&base_tokenizer, source_texts.clone());
        for ((_, expected_result), (tokens, offsets, mask)) in test_tuples.iter().zip(results.iter()) {
            let tokens: Vec<&str> = tokens.iter().map(|t|t.as_str()).collect();
            assert_eq!(tokens, expected_result.0);
            assert_eq!(*offsets, expected_result.1);
            assert_eq!(*mask, expected_result.2);
        }

        let results = MultiThreadedTokenizer::tokenize_list_with_offsets(&base_tokenizer, source_texts.clone());
        for ((_, expected_result), (tokens, offsets, mask)) in test_tuples.iter().zip(results.iter()) {
            let tokens: Vec<&str> = tokens.iter().map(|t|t.as_str()).collect();
            assert_eq!(tokens, expected_result.0);
            assert_eq!(*offsets, expected_result.1);
            assert_eq!(*mask, expected_result.2);
        }
    }

    #[test]
    fn test_no_lower_casing() {
//        Given
        let vocab = Arc::new(generate_test_vocab());
        let base_tokenizer: BaseTokenizer<BertVocab> = BaseTokenizer::from_existing_vocab(vocab, false, true);
        let test_tuples = [
            (
                "Sentence with [MASK] token.",
                (vec!("Sentence", "with", "[MASK]", "token", "."),
                 vec!(Offset::new(0,8), Offset::new(9,13), Offset::new(14,20), Offset::new(21,26), Offset::new(26,27)),
                 vec!(Mask::None, Mask::None, Mask::Special, Mask::None, Mask::Punctuation))
            ),
            (
                "[CLS]",
                (vec!("[CLS]"),
                vec!(Offset::new(0,5)),
                vec!(Mask::Special))
            ),
            (
                "[CLS] [PAD]",
                (vec!("[CLS]", "[PAD]"),
                 vec!(Offset::new(0,5), Offset::new(6,11)),
                 vec!(Mask::Special, Mask::Special))
            ),
            (
                "[CLS]       [PAD]",
                (vec!("[CLS]", "[PAD]"),
                 vec!(Offset::new(0,5), Offset::new(12,17)),
                 vec!(Mask::Special, Mask::Special))
            ),
            (
                "aSdF",
                (vec!("aSdF"),
                 vec!(Offset::new(0,4)),
                 vec!(Mask::None))
            ),
            (
                "",
                (vec!(),vec!(),vec!())
            ),
            (
                "Allons, Flipote, allons; que d'eux je me délivre.",
                (vec!("Allons", ",", "Flipote", ",", "allons", ";", "que", "d", "\'", "eux", "je", "me", "delivre", "."),
                 vec!(
                     Offset { begin: 0, end: 6 }, Offset { begin: 6, end: 7 }, Offset { begin: 8, end: 15 }, Offset { begin: 15, end: 16 }, Offset { begin: 17, end: 23 }, Offset { begin: 23, end: 24 }, Offset { begin: 25, end: 28 }, Offset { begin: 29, end: 30 }, Offset { begin: 30, end: 31 }, Offset { begin: 31, end: 34 }, Offset { begin: 35, end: 37 }, Offset { begin: 38, end: 40 }, Offset { begin: 41, end: 48 }, Offset { begin: 48, end: 49 }
                     ),
                 vec!(Mask::None, Mask::Punctuation, Mask::None, Mask::Punctuation, Mask::None, Mask::Punctuation, Mask::None, Mask::None, Mask::Punctuation, Mask::None, Mask::None, Mask::None, Mask::None, Mask::Punctuation)  ),
            ),
            (
                "[UNK]中华人民共和国 [PAD] asdf",
                (vec!("[UNK]", "中", "华", "人", "民", "共", "和", "国", "[PAD]", "asdf"),
                 vec!(Offset { begin: 0, end: 5 }, Offset { begin: 5, end: 6 }, Offset { begin: 6, end: 7 }, Offset { begin: 7, end: 8 }, Offset { begin: 8, end: 9 }, Offset { begin: 9, end: 10 }, Offset { begin: 10, end: 11 }, Offset { begin: 11, end: 12 }, Offset { begin: 13, end: 18 }, Offset { begin: 19, end: 23 }),
                 vec!(Mask::Unknown, Mask::CJK, Mask::CJK, Mask::CJK, Mask::CJK, Mask::CJK, Mask::CJK, Mask::CJK, Mask::Special, Mask::None))
            )
        ];
        let source_texts: Vec<&str> = test_tuples.iter().map(|v| v.0).collect();

//        When & Then
        for (source_text, expected_result) in test_tuples.iter() {
            let (tokens, offsets, mask) = base_tokenizer.tokenize_with_offsets(*source_text);
            let tokens: Vec<&str> = tokens.iter().map(|t|t.as_str()).collect();
            assert_eq!(tokens, expected_result.0);
            assert_eq!(offsets, expected_result.1);
            assert_eq!(mask, expected_result.2);
        }

        let results = Tokenizer::tokenize_list_with_offsets(&base_tokenizer, source_texts.clone());
        for ((_, expected_result), (tokens, offsets, mask)) in test_tuples.iter().zip(results.iter()) {
            let tokens: Vec<&str> = tokens.iter().map(|t|t.as_str()).collect();
            assert_eq!(tokens, expected_result.0);
            assert_eq!(*offsets, expected_result.1);
            assert_eq!(*mask, expected_result.2);
        }

        let results = MultiThreadedTokenizer::tokenize_list_with_offsets(&base_tokenizer, source_texts.clone());
        for ((_, expected_result), (tokens, offsets, mask)) in test_tuples.iter().zip(results.iter()) {
            let tokens: Vec<&str> = tokens.iter().map(|t|t.as_str()).collect();
            assert_eq!(tokens, expected_result.0);
            assert_eq!(*offsets, expected_result.1);
            assert_eq!(*mask, expected_result.2);
        }
    }

    #[test]
    fn test_convert_tokens_to_ids() {
//        Given
        let vocab = Arc::new(generate_test_vocab());
        let base_tokenizer: BaseTokenizer<BertVocab> = BaseTokenizer::from_existing_vocab(vocab, true, true);
        let test_tuples = [
            (
                vec!("hello", "[MASK]", "world", "!"),
                vec!(0, 6, 1, 3)
            ),
            (
                vec!("hello", ",", "una", "##ffa", "##ble", "world", "!"),
                vec!(0, 2, 11, 12, 13, 1, 3)
            ),
            (
                vec!("[UNK]", "[UNK]", "华", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[PAD]", "[UNK]"),
                vec!(2, 2, 8, 2, 2, 2, 2, 2, 10, 2)
            )
        ];

//        When & Then
        for (source_text, expected_result) in test_tuples.iter() {
            assert_eq!(base_tokenizer.convert_tokens_to_ids(source_text.iter().map(|v| String::from(*v)).collect::<Vec<_>>().as_ref()),
                       *expected_result);
        }
    }

    #[test]
    fn test_encode_single_sentence() {
//        Given
        let vocab = Arc::new(generate_test_vocab());
        let base_tokenizer: BaseTokenizer<BertVocab> = BaseTokenizer::from_existing_vocab(vocab, true, true);
        let truncation_strategy = TruncationStrategy::LongestFirst;
        let test_tuples = [
            (
                "hello world!",
                TokenizedInput { token_ids: vec!(0, 1, 3), segment_ids: vec!(0, 0, 0), special_tokens_mask: vec!(0, 0, 0), overflowing_tokens: vec!(), num_truncated_tokens: 0, token_offsets: vec!(Some(Offset::new(0,5)),Some(Offset::new(6,11)),Some(Offset::new(11,12))), mask: vec!(Mask::None, Mask::None, Mask::Punctuation) }
            ),
            (
                "hello, unaffable world!",
                TokenizedInput { token_ids: vec!(0, 2, 2, 1, 3), segment_ids: vec!(0, 0, 0, 0, 0), special_tokens_mask: vec!(0, 0, 0, 0, 0), overflowing_tokens: vec!(), num_truncated_tokens: 0, token_offsets: vec!(Some(Offset::new(0,5)), Some(Offset::new(5,6)), Some(Offset::new(7,16)), Some(Offset::new(17,22)), Some(Offset::new(22,23))), mask: vec!(Mask::None, Mask::Punctuation, Mask::None, Mask::None,  Mask::Punctuation)}
            ),
            (
                "[UNK]中华人民共和国 [PAD] asdf",
                TokenizedInput { token_ids: vec!(2, 7, 8, 9, 2, 2, 2, 2, 10, 2), segment_ids: vec!(0, 0, 0, 0, 0, 0, 0, 0, 0, 0), special_tokens_mask: vec!(0, 0, 0, 0, 0, 0, 0, 0, 0, 0), overflowing_tokens: vec!(), num_truncated_tokens: 0, token_offsets:
vec!(Some(Offset { begin: 0, end: 5 }), Some(Offset { begin: 5, end: 6 }), Some(Offset { begin: 6, end: 7 }), Some(Offset { begin: 7, end: 8 }), Some(Offset { begin: 8, end: 9 }), Some(Offset { begin: 9, end: 10 }), Some(Offset { begin: 10, end: 11 }), Some(Offset { begin: 11, end: 12 }), Some(Offset { begin: 13, end: 18 }), Some(Offset { begin: 19, end: 23 })), mask:
                vec!(Mask::Unknown, Mask::CJK, Mask::CJK, Mask::CJK, Mask::CJK, Mask::CJK, Mask::CJK, Mask::CJK, Mask::Special, Mask::None),
                }
            ),
            (
                "[UNK] a ! c ! e ! g ! i ! [PAD] a ! c ! e ! g ! i !",
                TokenizedInput { token_ids: vec!(2, 2, 3, 2, 3, 2, 3, 2, 3, 2), segment_ids: vec!(0, 0, 0, 0, 0, 0, 0, 0, 0, 0), special_tokens_mask: vec!(0, 0, 0, 0, 0, 0, 0, 0, 0, 0), overflowing_tokens: vec!(3, 10, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3), num_truncated_tokens: 12, token_offsets: vec!( Some(Offset { begin: 0, end: 5 }), Some(Offset { begin: 6, end: 7 }), Some(Offset { begin: 8, end: 9 }), Some(Offset { begin: 10, end: 11 }), Some(Offset { begin: 12, end: 13 }), Some(Offset { begin: 14, end: 15 }), Some(Offset { begin: 16, end: 17 }), Some(Offset { begin: 18, end: 19 }), Some(Offset { begin: 20, end: 21 }), Some(Offset { begin: 22, end: 23 }) ), mask: vec!(Mask::Unknown, Mask::None, Mask::Punctuation, Mask::None, Mask::Punctuation, Mask::None, Mask::Punctuation, Mask::None, Mask::Punctuation, Mask::None) }
            )
        ];
        let source_texts: Vec<&str> = test_tuples.iter().map(|v| v.0).collect();
        let expected_results: Vec<TokenizedInput> = test_tuples.iter().map(|v| v.1.clone()).collect();

//        When & Then
        for (source_text, expected_result) in test_tuples.iter() {
            let tokenized_input = base_tokenizer.encode(source_text, None, 10, &truncation_strategy, 0);
            assert_eq!(tokenized_input.token_ids.len(), tokenized_input.token_offsets.len(), "Offsets and tokens must have same length");
            assert_eq!(tokenized_input, *expected_result, "Testing results");
        }
        assert_eq!(Tokenizer::encode_list(&base_tokenizer, source_texts.clone(), 10, &truncation_strategy, 0), expected_results);
        assert_eq!(MultiThreadedTokenizer::encode_list(&base_tokenizer, source_texts.clone(), 10, &truncation_strategy, 0), expected_results);
    }

    #[test]
    fn test_encode_sentence_pair() {
//        Given
        let vocab = Arc::new(generate_test_vocab());
        let base_tokenizer: BaseTokenizer<BertVocab> = BaseTokenizer::from_existing_vocab(vocab, true, true);
        let truncation_strategy = TruncationStrategy::LongestFirst;
        let test_tuples = [
//            No truncation required
            (
                ("hello world!", "This is the second sentence"),
                TokenizedInput { token_ids: vec!(0, 1, 3, 2, 2, 2, 2, 2), segment_ids: vec!(0, 0, 0, 1, 1, 1, 1, 1), special_tokens_mask: vec!(0, 0, 0, 0, 0, 0, 0, 0), overflowing_tokens: vec!(), num_truncated_tokens: 0, token_offsets: vec!(Some(Offset::new(0,5)),Some(Offset::new(6,11)),Some(Offset::new(11,12)),Some(Offset::new(0,4)), Some(Offset::new(5,7)), Some(Offset::new(8,11)), Some(Offset::new(12,18)), Some(Offset::new(19,27)) ), mask: vec!(Mask::None, Mask::None, Mask::Punctuation, Mask::None, Mask::None, Mask::None, Mask::None, Mask::None) }
            ),
//            Truncation of sentence 2 (longest)
            (
                ("hello world!", "!This is the second sentence!!!"),
                TokenizedInput { token_ids: vec!(0, 1, 3, 3, 2, 2, 2, 2, 2, 3), segment_ids: vec!(0, 0, 0, 1, 1, 1, 1, 1, 1, 1), special_tokens_mask: vec!(0, 0, 0, 0, 0, 0, 0, 0, 0, 0), overflowing_tokens: vec!(), num_truncated_tokens: 2, token_offsets: vec!(
                 Some(Offset { begin: 0, end: 5 }), Some(Offset { begin: 6, end: 11 }), Some(Offset { begin: 11, end: 12 }), Some(Offset { begin: 0, end: 1 }), Some(Offset { begin: 1, end: 5 }), Some(Offset { begin: 6, end: 8 }), Some(Offset { begin: 9, end: 12 }), Some(Offset { begin: 13, end: 19 }), Some(Offset { begin: 20, end: 28 }), Some(Offset { begin: 28, end: 29 })
                ),
                mask: vec!(Mask::None, Mask::None, Mask::Punctuation, Mask::Punctuation, Mask::None, Mask::None, Mask::None, Mask::None, Mask::None, Mask::Punctuation)
                }
            ),
//            Truncation of sentence 1 (longest)
            (
                ("[UNK] hello  hello  hello  hello  hello  hello  hello  hello  hello  hello  hello", "!!!"),
                TokenizedInput { token_ids: vec!(2, 0, 0, 0, 0, 0, 0, 3, 3, 3), segment_ids: vec!(0, 0, 0, 0, 0, 0, 0, 1, 1, 1), special_tokens_mask: vec!(0, 0, 0, 0, 0, 0, 0, 0, 0, 0), overflowing_tokens: vec!(0, 0, 0, 0, 0), num_truncated_tokens: 5, token_offsets: vec!(
                Some(Offset { begin: 0, end: 5 }), Some(Offset { begin: 6, end: 11 }), Some(Offset { begin: 13, end: 18 }), Some(Offset { begin: 20, end: 25 }), Some(Offset { begin: 27, end: 32 }), Some(Offset { begin: 34, end: 39 }), Some(Offset { begin: 41, end: 46 }), Some(Offset { begin: 0, end: 1 }), Some(Offset { begin: 1, end: 2 }), Some(Offset { begin: 2, end: 3 })
                    ),
            mask: vec!(Mask::Unknown, Mask::None, Mask::None, Mask::None, Mask::None, Mask::None, Mask::None, Mask::Punctuation, Mask::Punctuation, Mask::Punctuation)
                }
            ),
//            Truncation of both sentences (longest)
            (
                ("[UNK] hello  hello  hello  hello  hello", "!!!!!!!!"),
                TokenizedInput { token_ids: vec!(2, 0, 0, 0, 0, 3, 3, 3, 3, 3), segment_ids: vec!(0, 0, 0, 0, 0, 1, 1, 1, 1, 1), special_tokens_mask: vec!(0, 0, 0, 0, 0, 0, 0, 0, 0, 0), overflowing_tokens: vec!(0), num_truncated_tokens: 4, token_offsets: vec!(
                Some(Offset { begin: 0, end: 5 }), Some(Offset { begin: 6, end: 11 }), Some(Offset { begin: 13, end: 18 }), Some(Offset { begin: 20, end: 25 }), Some(Offset { begin: 27, end: 32 }), Some(Offset { begin: 0, end: 1 }), Some(Offset { begin: 1, end: 2 }), Some(Offset { begin: 2, end: 3 }), Some(Offset { begin: 3, end: 4 }), Some(Offset { begin: 4, end: 5 })
                    ),
            mask: vec!(Mask::Unknown, Mask::None, Mask::None, Mask::None, Mask::None, Mask::Punctuation, Mask::Punctuation, Mask::Punctuation, Mask::Punctuation, Mask::Punctuation)
                }
            )
        ];
        let source_texts: Vec<(&str, &str)> = test_tuples.iter().map(|v| v.0).collect();
        let expected_results: Vec<TokenizedInput> = test_tuples.iter().map(|v| v.1.clone()).collect();

//        When & Then
        for (source_text, expected_result) in test_tuples.iter() {
            let tokenized_input = base_tokenizer.encode(source_text.0, Some(source_text.1), 10, &truncation_strategy, 0);
            assert_eq!(tokenized_input.token_ids.len(), tokenized_input.token_offsets.len(), "Offsets and tokens must have same length");
            assert_eq!(tokenized_input , *expected_result, "Testing results");
        }
        assert_eq!(Tokenizer::encode_pair_list(&base_tokenizer, source_texts.clone(), 10, &truncation_strategy, 0), expected_results);
        assert_eq!(MultiThreadedTokenizer::encode_pair_list(&base_tokenizer, source_texts.clone(), 10, &truncation_strategy, 0), expected_results);
    }

    #[test]
    fn test_decode() {
//        Given
        let vocab = Arc::new(generate_test_vocab());
        let base_tokenizer: BaseTokenizer<BertVocab> = BaseTokenizer::from_existing_vocab(vocab, true, true);
        let skip_special_tokens = false;
        let clean_up_tokenization_spaces = false;
        let test_tuples = [
            (
                vec!(0, 1, 3),
                "hello world !",
            ),
            (
                vec!(10, 0, 1, 3),
                "[PAD] hello world !",
            ),
            (
                vec!(10, 0, 1, 2, 3),
                "[PAD] hello world [UNK] !",
            )
        ];
        let source_ids: Vec<Vec<i64>> = test_tuples.iter().map(|v| v.0.clone()).collect_vec();
        let expected_results: Vec<&str> = test_tuples.iter().map(|v| v.1.clone()).collect_vec();

//        When & Then
        for (source_ids, expected_result) in test_tuples.iter() {
            assert_eq!(base_tokenizer.decode(source_ids.clone(), skip_special_tokens, clean_up_tokenization_spaces),
                       *expected_result);
        }
        assert_eq!(Tokenizer::decode_list(&base_tokenizer, source_ids.clone(), skip_special_tokens, clean_up_tokenization_spaces), expected_results);
        assert_eq!(MultiThreadedTokenizer::decode_list(&base_tokenizer, source_ids.clone(), skip_special_tokens, clean_up_tokenization_spaces), expected_results);
    }

    #[test]
    fn test_decode_skip_special_tokens() {
//        Given
        let vocab = Arc::new(generate_test_vocab());
        let base_tokenizer: BaseTokenizer<BertVocab> = BaseTokenizer::from_existing_vocab(vocab, true, true);
        let skip_special_tokens = true;
        let clean_up_tokenization_spaces = false;
        let test_tuples = [
            (
                vec!(0, 1, 3),
                "hello world !",
            ),
            (
                vec!(10, 0, 1, 3),
                "hello world !",
            ),
            (
                vec!(10, 0, 1, 2, 3),
                "hello world !",
            )
        ];
        let source_ids: Vec<Vec<i64>> = test_tuples.iter().map(|v| v.0.clone()).collect_vec();
        let expected_results: Vec<&str> = test_tuples.iter().map(|v| v.1.clone()).collect_vec();

//        When & Then
        for (source_ids, expected_result) in test_tuples.iter() {
            assert_eq!(base_tokenizer.decode(source_ids.clone(), skip_special_tokens, clean_up_tokenization_spaces),
                       *expected_result);
        }
        assert_eq!(Tokenizer::decode_list(&base_tokenizer, source_ids.clone(), skip_special_tokens, clean_up_tokenization_spaces), expected_results);
        assert_eq!(MultiThreadedTokenizer::decode_list(&base_tokenizer, source_ids.clone(), skip_special_tokens, clean_up_tokenization_spaces), expected_results);
    }

    #[test]
    fn test_decode_clean_up_tokenization_spaces() {
//        Given
        let vocab = Arc::new(generate_test_vocab());
        let base_tokenizer: BaseTokenizer<BertVocab> = BaseTokenizer::from_existing_vocab(vocab, true, true);
        let skip_special_tokens = true;
        let clean_up_tokenization_spaces = true;
        let test_tuples = [
            (
                vec!(0, 1, 3),
                "hello world!",
            ),
            (
                vec!(10, 0, 1, 3),
                "hello world!",
            ),
            (
                vec!(10, 0, 1, 2, 3),
                "hello world!",
            )
        ];
        let source_ids: Vec<Vec<i64>> = test_tuples.iter().map(|v| v.0.clone()).collect_vec();
        let expected_results: Vec<&str> = test_tuples.iter().map(|v| v.1.clone()).collect_vec();

//        When & Then
        for (source_ids, expected_result) in test_tuples.iter() {
            assert_eq!(base_tokenizer.decode(source_ids.clone(), skip_special_tokens, clean_up_tokenization_spaces),
                       *expected_result);
        }
        assert_eq!(Tokenizer::decode_list(&base_tokenizer, source_ids.clone(), skip_special_tokens, clean_up_tokenization_spaces), expected_results);
        assert_eq!(MultiThreadedTokenizer::decode_list(&base_tokenizer, source_ids.clone(), skip_special_tokens, clean_up_tokenization_spaces), expected_results);
    }
}
