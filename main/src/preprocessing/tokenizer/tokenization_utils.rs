// Copyright 2018 The Open AI Team Authors, The Google AI Language Team Authors
// Copyright 2018 The HuggingFace Inc. team.
// Copyright 2019-2020 Guillaume Becquin
// Copyright 2020 Maarten van Gompel
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
use crate::BertVocab;
use crate::preprocessing::tokenizer::constants::{WHITESPACE_CHARS, ADDITIONAL_WHITESPACE_CHARS,
                                                 PUNCTUATION_CHARS, CONTROL_CHARS, ACCENT_MARKERS, BYTES_TO_UNICODE};
use unicode_normalization::char::decompose_canonical;
use std::char;
use std::char::REPLACEMENT_CHARACTER;
use std::error::Error;
use std::cmp::min;
use std::cell::RefCell;
use std::collections::{HashMap,HashSet};
use regex::Regex;
use crate::preprocessing::tokenizer::base_tokenizer::{TruncationStrategy,Offset,OffsetSize,Token,TokenRef,Mask};
use crate::preprocessing::vocab::bpe_vocab::{BpePairRef, BpePairVocab};


///Cleans text by removing control characters and normalizing whitespace
pub fn clean_text(text: &str, strict: bool) -> String {
    let mut output = String::new();
    for character in text.chars() {
        if is_control(&character, strict) || character == '\x00' || character == REPLACEMENT_CHARACTER {
            continue;
        }
        if is_whitespace(&character) {
            output.push(' ');
        } else {
            output.push(character);
        }
    }
    output
}

///Split a text on special tokens (like BOS/EOS/UNK markers), depending on the vocabulary
pub fn split_on_special_tokens<'a>(token: TokenRef<'a>, vocab: &impl Vocab) -> Vec<TokenRef<'a>> {
    let test_substr = |s: &str| {
        for special_value in vocab.special_values().keys() {
            if s.starts_with(special_value.as_str()) {
                return (special_value.len(), special_value.chars().count(),
                    if vocab.get_unknown_value() == special_value.as_str() {
                        Mask::Unknown
                    } else {
                        Mask::Special
                    });
            }
        }
        (0,0,Mask::None)
    };
    split_on_substr(token, test_substr, true)
}

///Tokenises CJK characters, each character will be a token
pub fn tokenize_cjk_chars<'a>(token: TokenRef<'a>) -> Vec<TokenRef<'a>> {
    split_on_char(token, is_cjk_char, true, Mask::CJK)
}

fn is_cjk_char(character: &char) -> bool {
    let u32_char = *character as u32;
    ((u32_char >= 0x4E00) & (u32_char <= 0x9FFF)) |
        ((u32_char >= 0x3400) & (u32_char <= 0x4DBF)) |
        ((u32_char >= 0x20000) & (u32_char <= 0x2A6DF)) |
        ((u32_char >= 0x2A700) & (u32_char <= 0x2B73F)) |
        ((u32_char >= 0x2B740) & (u32_char <= 0x2B81F)) |
        ((u32_char >= 0x2B820) & (u32_char <= 0x2CEAF)) |
        ((u32_char >= 0xF900) & (u32_char <= 0xFAFF)) |
        ((u32_char >= 0x2F800) & (u32_char <= 0x2FA1F))
}


pub fn is_whitespace(character: &char) -> bool {
    WHITESPACE_CHARS.contains(&(*character as u32))
}

///    This is a custom method to check if a character is a control character. The BERT tokenizer is
/// taking any character whose unicode category starts with `C` as a control character, which includes
/// the traditional control `Cc` category, but also the format `Cc`, private use `Co` and surrogate `Cs`.
/// The unassigned unicode category `Cn` has been skipped in order to avoid unnecessary checks.
///    A faster method may be called by setting strict to false and only check against the core control
/// characters. To match the original BERT tokenization, this should remain true.
pub fn is_control(character: &char, strict: bool) -> bool {
    if ADDITIONAL_WHITESPACE_CHARS.contains(character) {
        false
    } else {
        if strict {
            let u32_char = *character as u32;
            if (u32_char <= 0x001F) |
                ((u32_char >= 0x0080) & (u32_char <= 0x009F)) |
                ((u32_char >= 0xE0020) & (u32_char <= 0xE007F)) |
                ((u32_char >= 0xE000) & (u32_char <= 0xF8FF)) |
                ((u32_char >= 0xF0000) & (u32_char <= 0xFFFFD)) |
                ((u32_char >= 0x100000) & (u32_char <= 0x10FFFD)) |
                ((u32_char >= 0xD800) & (u32_char <= 0xDB7F)) |
                ((u32_char >= 0xDB80) & (u32_char <= 0xDBFF)) |
                ((u32_char >= 0xDC00) & (u32_char <= 0xDFFF)) |
                CONTROL_CHARS.contains(&u32_char)
            {
                true
            } else {
                false
            }
        } else {
            character.is_control()
        }
    }
}

pub fn is_punctuation(character: &char) -> bool {
    let u32_char = *character as u32;
    if ((u32_char >= 33) & (u32_char <= 47)) |
        ((u32_char >= 58) & (u32_char <= 64)) |
        ((u32_char >= 91) & (u32_char <= 96)) |
        ((u32_char >= 123) & (u32_char <= 126)) {
        true
    } else {
        PUNCTUATION_CHARS.contains(&u32_char)
    }
}


///Simple tokenisation based on whitespace only
pub fn whitespace_tokenize<'a>(token: TokenRef<'a>) -> Vec<TokenRef<'a>> {
    split_on_char(token, is_whitespace, false, Mask::Whitespace )
}


///Remove diacritics
pub fn strip_accents(text: String) -> String {
    let mut decomposed_string: String = String::with_capacity(text.capacity());
    for character in text.chars() {
        decompose_canonical(character, |c| if !ACCENT_MARKERS.contains(&(c as u32)) {
            decomposed_string.push(c)
        });
    }
    decomposed_string
}


///Split a token on punctuation
pub fn split_on_punct<'a>(token: TokenRef<'a>) -> Vec<TokenRef<'a>> {
    split_on_char(token, is_punctuation, true, Mask::Punctuation )
}

///Split a token on one or more characters (given a character test function)
/// * token: The token to split
/// * test_character: A function that borrows a `char` and returns a boolean. If true, a split will be made here
/// * add_separators: Add the separating characters to the tokens as well? (bool), separating tokens will be indicated in the returned mask by the value set in `set_mask`
pub fn split_on_char<'a, F>(token: TokenRef<'a>, test_character: F, add_separators: bool, set_mask: Mask) -> Vec<TokenRef<'a>>
    where F: Fn(&char) -> bool {
    let mut tokens: Vec<TokenRef<'a>> = Vec::new();
    let mut charbegin: usize = 0;
    let mut bytesbegin: usize = 0;
    let mut charcount: usize = 0;


    if token.mask == Mask::None {
        //iterate over all characters, returning the byte position with each
        for ( charidx , (bytesidx, c)) in token.text.char_indices().enumerate() {
            charcount += 1;
            if test_character(&c) {
                if charbegin < charidx {
                    //add previous token
                    tokens.push(TokenRef {
                        text: &token.text[bytesbegin..bytesbegin + (bytesidx - bytesbegin)],
                        offset: Offset { begin: token.offset.begin + charbegin as OffsetSize , end: token.offset.begin + charidx as OffsetSize },
                        mask: Mask::None
                    });
                }
                if add_separators {
                    //add seperator as a singleton token
                    tokens.push(TokenRef {
                        text: &token.text[bytesidx..bytesidx + c.len_utf8()],
                        offset: Offset { begin: token.offset.begin + charidx as OffsetSize, end: token.offset.begin + charidx as OffsetSize + 1},
                        mask: set_mask
                    });
                }
                //reset
                charbegin = charidx + 1;
                bytesbegin = bytesidx + c.len_utf8();
            }
        }
    }
    if charcount == 0 {
        //nothing done, return token as is
        tokens.push(token);
    } else if bytesbegin < token.text.len() {
        //add last buffered token if there is anything left
        if charcount == 0 {
            charcount = token.text.chars().count();
        }
        let bytesidx = token.text.len();
        tokens.push(TokenRef {
            text: &token.text[bytesbegin..bytesbegin + (bytesidx - bytesbegin)],
            offset: Offset { begin: token.offset.begin + charbegin as OffsetSize, end: token.offset.begin + charcount as OffsetSize },
            mask: Mask::None,
        });
    }
    tokens
}

pub fn split_on_regex_with_lookahead<'a>(token: TokenRef<'a>, pattern_lookahead: &Regex, pattern_tokenization: &Regex) -> Vec<TokenRef<'a>> {
    if token.mask == Mask::None {
        let mut sub_words: Vec<TokenRef<'a>> = vec!();
        let mut splits: Vec<TokenRef<'a>> = vec!();

        let mut beginbyte: usize = 0;
        let mut endbyte: usize;
        let mut beginchar: usize = 0; //chars
        let mut endchar: usize;
        for hit in pattern_lookahead.find_iter(token.text) {
            //eprintln!("hit in lookahead: {:?}", hit);
            endbyte = hit.end() - 1 - hit.as_str().chars().last().unwrap().len_utf8();
            let splittext = &token.text[beginbyte..endbyte];
            endchar = beginchar + splittext.chars().count();
            splits.push(TokenRef {
                text: splittext,
                offset: Offset::new(token.offset.begin + beginchar as OffsetSize, token.offset.begin + endchar as OffsetSize),
                mask: Mask::None,
            });
            beginbyte = endbyte;
            beginchar = endchar;
        }
        splits.push(TokenRef {
            text: &token.text[beginbyte..],
            offset: Offset::new(token.offset.begin + beginchar as OffsetSize, token.offset.begin + token.text.chars().count() as OffsetSize),
            mask: Mask::None,
        });

        for sub_word in splits {
            sub_words.extend(split_on_regex(sub_word, pattern_tokenization))
        }

        sub_words

    } else {
        vec!(token)
    }
}

pub fn split_on_regex<'a>(token: TokenRef<'a>, pattern_tokenization: &Regex) -> Vec<TokenRef<'a>> {
    let mut tokens: Vec<TokenRef<'a>> = Vec::new();
    let mut endchar: usize;
    let mut beginchar: usize = token.offset.begin as usize;
    for hit in pattern_tokenization.find_iter(token.text) {
        let startbyte = hit.start();
        if startbyte > 0 {
           beginchar = token.offset.begin as usize + token.text[..startbyte].chars().count();
        }
        endchar = beginchar + hit.as_str().chars().count();
        tokens.push( TokenRef {
            text: hit.as_str(),
            offset: Offset::new(beginchar as OffsetSize, endchar as OffsetSize),
            mask: Mask::None,
        });
        beginchar = endchar;
    }
    tokens
}

///Split a token on one or more substrings (given a substring test function)
/// * token: The token to split
/// * test_str: A function that contains the string buffer from the current point forward and
/// returns a 3-tuple with the length of the match in bytes, chars and the mask to set  (if the
/// length is zero then there is no match.
/// * add_separators: Add the separating characters to the tokens as well? (bool), separating tokens will be indicated in the returned mask by the value set in `set_mask`, which is returned by the test_substr function
pub fn split_on_substr<'a, F>(token: TokenRef<'a>, test_substr: F, add_separators: bool) -> Vec<TokenRef<'a>>
    where F: Fn(&'a str) -> (usize,usize,Mask) {
    let mut tokens: Vec<TokenRef<'a>> = Vec::new();
    let mut charbegin: usize = 0;
    let mut bytesbegin: usize = 0;
    let mut charcount: usize = 0;

    if token.mask == Mask::None { //don't process a token that already got marked in the mask
        //iterate over all characters, returning the byte position with each
        for ( charidx , (bytesidx, _)) in token.text.char_indices().enumerate() {
            charcount += 1;
            let (matched_bytes, matched_chars, set_mask): (usize, usize, Mask) = test_substr(&token.text[bytesidx..]);
            if matched_chars > 0 {
                if charbegin < charidx {
                    //add previous token
                    tokens.push(TokenRef {
                        text: &token.text[bytesbegin..bytesbegin + (bytesidx - bytesbegin)],
                        offset: Offset { begin: token.offset.begin + charbegin as OffsetSize , end: token.offset.begin + (charbegin+matched_chars) as OffsetSize },
                        mask: Mask::None
                    });
                }
                if add_separators {
                    //add seperator as a singleton token
                    tokens.push(TokenRef {
                        text: &token.text[bytesidx..bytesidx + matched_bytes],
                        offset: Offset { begin: token.offset.begin + charidx as OffsetSize, end: token.offset.begin + (charidx + matched_chars) as OffsetSize  },
                        mask: set_mask,
                    });
                }
                //reset
                charbegin = charidx + matched_chars;
                bytesbegin = bytesidx + matched_bytes;
            }
        }
    }
    if bytesbegin < token.text.len() {
        //add last buffered token if there is anything left
        if charcount == 0 {
            charcount = token.text.chars().count();
        }
        let bytesidx = token.text.len();
        tokens.push(TokenRef {
            text: &token.text[bytesbegin..bytesbegin + (bytesidx - bytesbegin)],
            offset: Offset { begin: token.offset.begin + charbegin as OffsetSize, end: token.offset.begin + charcount as OffsetSize },
            mask: Mask::None
        });
    }
    tokens
}


///Tokenize a token into word pieces according to the supplied vocabulary
///Continuation wordpieces will all have the suffix `##`
pub fn tokenize_wordpiece<'a>(token: TokenRef<'a>, vocab: &impl Vocab, max_word_len: usize) -> Vec<Token> {
    let mut tokens: Vec<Token> = Vec::new();

    if token.text.chars().count() > max_word_len {
        tokens.push( Token {
            text: BertVocab::unknown_value().to_owned(),
            offset: token.offset.clone(),
            mask: Mask::Unknown
        });
    } else {
        let char_indices: Vec<usize> = token.text.char_indices().map(|v| v.0).collect();
        let max_end: usize = char_indices.last().unwrap() + token.text.chars().last().unwrap().len_utf8();
        let mut start: usize = 0; //bytes
        let mut pos_begin = 0; //chars
        let mut pos_end;
        let mut end;
        while start < max_end { //bytes
            end = max_end;
            pos_end = char_indices.len(); //chars
            let mut is_unk: bool = true; //out of vocabulary? to be falsified
            while start < end {
                let mut substr = token.text[start..end].to_owned();
                let charlength = substr.chars().count();
                let suboffset = Offset {
                    begin: token.offset.begin + pos_begin as OffsetSize,
                    end: token.offset.begin + pos_begin as OffsetSize + charlength as OffsetSize
                };
                if start > 0 {
                    substr = format!("##{}", substr);
                }
                if vocab.values().contains_key(&substr) {
                    tokens.push( Token { text: substr, offset: suboffset, mask: if start > 0 { Mask::Continuation } else { token.mask }});
                    is_unk = false;
                    break;
                }
                pos_end = pos_end - 1;
                end = char_indices[pos_end];
            }
            if is_unk {
                return vec!( Token {
                    text: BertVocab::unknown_value().to_owned(),
                    offset: token.offset.clone(),
                    mask: Mask::Unknown
                });
            }
            start = end;
            pos_begin = pos_end;
        }

        //fix the mask, set Mask::Begin where a sequence of continuations is introduced
        for i in 1..(tokens.len() - 1) {
            if tokens[i].mask == Mask::Continuation && tokens[i-1].mask == Mask::None {
                if let Some(token) = tokens.get_mut(i-1) {
                    token.mask = Mask::Begin;
                }
            }
        }

    }

    tokens
}

/// # Truncates a sequence pair in place to the maximum length.
///
///   * tokens_1: list of tokenized input ids. Can be obtained from a string by chaining the
///       `tokenize` and `convert_tokens_to_ids` methods.
///   * tokens_2: Optional second list of input ids. Can be obtained from a string by chaining the
///       `tokenize` and `convert_tokens_to_ids` methods.
///   * offsets: list of offsets for tokens_1 (must be same length or empty if not used at all)
///   * offsets_2: optional second list of offsets for tokens_2 (must be same length or empty if not used at all)
///   * tokens_2: Optional second list of input ids. Can be obtained from a string by chaining the
///       `tokenize` and `convert_tokens_to_ids` methods.
///   * num_tokens_to_remove
///       number of tokens to remove using the truncation strategy
///   * truncation_strategy: truncation strategy
///       - TruncationStrategy::LongestFirst (default) Iteratively reduce the inputs sequence until the input is under max_length
///           starting from the longest one at each token (when there is a pair of input sequences).
///           Overflowing tokens only contains overflow from the first sequence.
///       - TruncationStrategy::OnlyFirst: Only truncate the first sequence. raise an error if the first sequence is shorter or equal to than num_tokens_to_remove.
///       - TruncationStrategy::OnlySecond: Only truncate the second sequence
///       - TruncationStrategy::DoNotTruncate: Does not truncate (raise an error if the input sequence is longer than max_length)
///   * stride
///       If set to a number along with max_length, the overflowing tokens returned will contain some tokens
///       from the main sequence returned. The value of this argument defines the number of additional tokens.
///
pub fn truncate_sequences(mut tokens_1: Vec<i64>, tokens_2: Option<Vec<i64>>,
                      mut offsets_1: Vec<Offset>, mut offsets_2: Option<Vec<Offset>>,
                      mut mask_1: Vec<Mask>, mut mask_2: Option<Vec<Mask>>,
                      num_tokens_to_remove: usize, truncation_strategy: &TruncationStrategy, stride: usize)
                      -> Result<(Vec<i64>, Option<Vec<i64>>, Vec<Offset>, Option<Vec<Offset>>, Vec<Mask>, Option<Vec<Mask>>, Vec<i64>, Vec<Offset>), Box<dyn Error>> {
    if num_tokens_to_remove == 0 {
        Ok((tokens_1, tokens_2, offsets_1, offsets_2, mask_1, mask_2, Vec::new(), Vec::new()))
    } else {
        match tokens_2 {
            Some(mut tokens_2) => {
                match truncation_strategy {
                    TruncationStrategy::LongestFirst => {
                        if (tokens_1.len() + tokens_2.len()) >= num_tokens_to_remove {
                            let mut overflow_tokens: Vec<i64> = Vec::with_capacity(num_tokens_to_remove + stride);
                            let mut overflow_offsets: Vec<Offset> = Vec::with_capacity(num_tokens_to_remove + stride);
                            for _ in 0..num_tokens_to_remove {
                                if tokens_1.len() >= tokens_2.len() {
                                    overflow_tokens.insert(0, tokens_1.pop().unwrap());
                                    if !offsets_1.is_empty() {
                                        overflow_offsets.insert(0, offsets_1.pop().unwrap());
                                    }
                                    if !mask_1.is_empty() {
                                        mask_1.pop();
                                    }
                                } else {
                                    tokens_2.pop();
                                    offsets_2 = offsets_2.map(|mut offsets_2| {
                                        offsets_2.pop();
                                        offsets_2
                                    });
                                    mask_2 = mask_2.map(|mut mask_2| {
                                        mask_2.pop();
                                        mask_2
                                    });
                                }

                            }
                            let window_len = min(tokens_1.len(), stride);
                            if window_len > 0 {
                                let slice: &[i64] = &tokens_1[&tokens_1.len() - window_len..];
                                overflow_tokens.splice(0..0, slice.iter().cloned());
                                if !offsets_1.is_empty() {
                                    let offset_slice: &[Offset] = &offsets_1[&offsets_1.len() - window_len..];
                                    overflow_offsets.splice(0..0, offset_slice.iter().cloned());
                                }
                            }
                            Ok((tokens_1, Some(tokens_2), offsets_1, offsets_2, mask_1, mask_2, overflow_tokens, overflow_offsets))
                        } else {
                            Err("Combined sequence length too short for requested truncation amount".into())
                        }
                    }
                    TruncationStrategy::OnlyFirst => {
                        if tokens_1.len() >= num_tokens_to_remove {
                            let (overflow_tokens, overflow_offsets) = truncate_with_overflow(&mut tokens_1, offsets_1.as_mut() , mask_1.as_mut(), num_tokens_to_remove, stride);
                            Ok((tokens_1, Some(tokens_2), offsets_1, offsets_2, mask_1, mask_2, overflow_tokens, overflow_offsets))
                        } else {
                            Err("First sequence too short for first only truncation".into())
                        }
                    }
                    TruncationStrategy::OnlySecond => {
                        if tokens_2.len() >= num_tokens_to_remove {
                            let (overflow_tokens, overflow_offsets) = truncate_with_overflow(&mut tokens_2, offsets_2.as_mut().unwrap_or(&mut vec!()), mask_2.as_mut().unwrap_or(&mut vec!()), num_tokens_to_remove, stride);
                            Ok((tokens_1, Some(tokens_2), offsets_1, offsets_2, mask_1, mask_2, overflow_tokens, overflow_offsets))
                        } else {
                            Err("Second sequence too short for second only truncation".into())
                        }
                    }
                    TruncationStrategy::DoNotTruncate => Err("Truncation needed but no truncation requested".into())
                }
            }
            None => {
                if tokens_1.len() >= num_tokens_to_remove {
                    match truncation_strategy {
                        TruncationStrategy::LongestFirst | TruncationStrategy::OnlyFirst => {
                            let (overflow_tokens, overflow_offsets) = truncate_with_overflow(&mut tokens_1, &mut offsets_1, &mut mask_1, num_tokens_to_remove, stride);
                            Ok((tokens_1, None, offsets_1, offsets_2, mask_1, mask_2, overflow_tokens, overflow_offsets))
                        }
                        TruncationStrategy::OnlySecond => Err("Invalid truncation strategy for single sentence truncation".into()),
                        TruncationStrategy::DoNotTruncate => Err("Truncation needed but no truncation requested".into())
                    }
                } else {
                    Err("First sequence too short for first only truncation".into())
                }
            }
        }
    }
}

fn truncate_with_overflow(sequence: &mut Vec<i64>, offsets: &mut Vec<Offset>, mask: &mut Vec<Mask>, num_tokens_to_remove: usize, stride: usize) -> (Vec<i64>,Vec<Offset>) {
    if !offsets.is_empty() {
        assert_eq!(sequence.len(), offsets.len());
    }
    if !mask.is_empty() {
        assert_eq!(sequence.len(), mask.len());
    }
    let cutoff = sequence.len() - num_tokens_to_remove;
    let mut overflow_tokens = sequence.split_off(cutoff);
    let mut overflow_offsets = if !offsets.is_empty() {
        offsets.split_off(cutoff)
    } else {
        Vec::new()
    };
    if !mask.is_empty() {
        mask.split_off(cutoff);
    }
    let window_len = min(sequence.len(), stride);
    if window_len > 0 {
        let slice: &[i64] = &sequence[&sequence.len() - window_len..];
        overflow_tokens.splice(0..0, slice.iter().cloned());
        if !offsets.is_empty() {
            let offset_slice: &[Offset] = &offsets[&offsets.len() - window_len..];
            overflow_offsets.splice(0..0, offset_slice.iter().cloned());
        }
    }
    (overflow_tokens, overflow_offsets)
}

pub fn get_pairs(token: &Vec<String>) -> Option<HashSet<BpePairRef>> {
    match token.len() {
        0 | 1 => None,
        _ => {
            let mut output: HashSet<BpePairRef> = HashSet::with_capacity(token.len());
            for idx in 0..token.len() - 1 {
                if let [byte_1, byte_2] = &token[idx..idx + 2] {
                    output.insert(BpePairRef { byte_1, byte_2 });
                }
            }
            Some(output)
        }
    }
}

pub fn group_common_pairs(tokens: Vec<Token>, bpe_ranks: &BpePairVocab) -> (Vec<Token>, bool) {
    if let Some(pairs) = get_pairs(&tokens.iter().map(|token| token.text.clone()).collect() ) {
        let bigram = pairs.iter().min_by_key(|pair|
            match bpe_ranks.byte_pair_to_id(pair) {
                Some(&rank) => rank,
                None => i64::max_value()
            }).unwrap();
        if bpe_ranks.byte_pair_to_id(bigram).is_none() {
            return (tokens, true);
        }
        let mut temp_sub_tokens: Vec<Token> = Vec::with_capacity(tokens.len());
        let mut i = 0;

        while i < tokens.len() {
            let j = if let Some(index) = &tokens[i..].iter().position(|r| r.text.as_str() == bigram.byte_1) {
                index + i
            } else {
                temp_sub_tokens.extend_from_slice(&tokens[i..]);
                break;
            };
            temp_sub_tokens.extend_from_slice(&tokens[i..j]);
            i = j;
            if (&tokens[i].text.as_str() == bigram.byte_1) & (i < tokens.len() - 1) {
                if &tokens[i + 1].text.as_str() == bigram.byte_2 {
                    let mut combined_bytes = String::with_capacity(bigram.byte_1.len() + bigram.byte_2.len());
                    combined_bytes.push_str(bigram.byte_1.as_str());
                    combined_bytes.push_str(bigram.byte_2.as_str());
                    temp_sub_tokens.push(Token {
                        text: combined_bytes,
                        offset: Offset { //offset always references the original situation! Not the modified one!
                            begin: tokens[i].offset.begin,
                            end: tokens[i+1].offset.end
                        },
                        mask: tokens[i].mask,
                    });
                    i += 2;
                } else {
                    temp_sub_tokens.push(Token {
                        text: bigram.byte_1.clone(),
                        offset: tokens[i].offset.clone(),
                        mask: tokens[i].mask,
                    });
                    i += 1;
                }
            } else {
                temp_sub_tokens.push(Token {
                    text: bigram.byte_1.clone(),
                    offset: tokens[i].offset.clone(),
                    mask: tokens[i].mask,
                });
                i += 1;
            }
        }
        if temp_sub_tokens.len() == 1 {
            return (temp_sub_tokens, true);
        }
        return (temp_sub_tokens, false);
    } else {
        return (tokens, true);
    }
}

pub fn ctrl_bpe<'a>(token: TokenRef<'a>, bpe_ranks: &BpePairVocab, exact_offsets: bool) -> Vec<Token> {
    let mut sub_tokens = bpe_get_subtokens(token, exact_offsets);

    if !sub_tokens.is_empty() {
        sub_tokens.last_mut().unwrap().text += "</w>";
    };

    let mut output = (sub_tokens, false);
    loop {
        output = group_common_pairs(output.0, &bpe_ranks);
        if output.1 {
            break;
        }
    }

    let length = output.0.len();
    for (i, token) in output.0.iter_mut().enumerate() {
        if i < length - 1 {
            token.text += "@@";
        } else if i == length - 1 {
            //strip the last </w> suffix again, we only needed it for group_common_pairs
            token.text = token.text.trim_end_matches("</w>").to_owned();
        }
    }

    bpe_fix_mask(output.0, exact_offsets)
}

pub fn openai_gpt_bpe<'a>(token: TokenRef<'a>, bpe_ranks: &BpePairVocab, exact_offsets: bool) -> Vec<Token> {
    let mut sub_tokens = bpe_get_subtokens(token, exact_offsets);

    //the addition of </w> is basically the only difference between this function and the default bpe:
    if !sub_tokens.is_empty() {
        sub_tokens.last_mut().unwrap().text += "</w>";
    };

    let mut output = (sub_tokens, false);
    loop {
        output = group_common_pairs(output.0, &bpe_ranks);
        if output.1 {
            break;
        }
    }

    bpe_fix_mask(output.0, exact_offsets)
}

///Default bpe function, as called by Roberta and GPT2
pub fn bpe<'a>(token: TokenRef<'a>, bpe_ranks: &BpePairVocab, exact_offsets: bool) -> Vec<Token> {
    let sub_tokens = bpe_get_subtokens(token, exact_offsets);
    let mut output = (sub_tokens, false);
    loop {
        output = group_common_pairs(output.0, &bpe_ranks);
        if output.1 {
            break;
        }
    }

    bpe_fix_mask(output.0, exact_offsets)
}

///Split the token into per-character subtokens prior for byte-pair encoding
pub fn bpe_get_subtokens<'a>(token: TokenRef<'a>, exact_offsets: bool) -> Vec<Token> {
    token.text.chars().enumerate().map(|(i,chr)|
        Token {
            text: chr.to_string(),
            offset: match exact_offsets {
                true => Offset::new( token.offset.begin + i as OffsetSize, token.offset.begin + i as OffsetSize + 1),
                false => token.offset.clone()
            },
            mask: match exact_offsets {
                true => Mask::Continuation,
                false => Mask::InexactContinuation,
            }
        }).collect()
}

pub fn bpe_fix_mask(mut tokens: Vec<Token>, exact_offsets: bool) -> Vec<Token> {
    if let Some(first_token)  = tokens.get_mut(0) {
        first_token.mask = match exact_offsets {
            true => Mask::Begin,
            false => Mask::InexactBegin
        }
    }
    tokens
}

pub fn split_on_bpe_pairs<'a, F>(token: TokenRef<'a>, bpe_function: F, bpe_ranks: &BpePairVocab, cache: &RefCell<HashMap<String, Vec<Token>>>) -> Vec<Token>
    where F: Fn(TokenRef,&BpePairVocab, bool) -> Vec<Token>
{
    let mut tokens: Vec<Token> = Vec::new();
    let text: String = token.text.as_bytes().iter().map(|v| BYTES_TO_UNICODE.get(&v).unwrap()).collect();
    let cached: bool = match cache.borrow().get(&text) {
        Some(cached_tokens) => {
            tokens.extend(cached_tokens.clone().into_iter().map(|mut t| {
                //the tokens from the cache have 0-based offsets, adapt the offset
                //according to the input offset
                t.offset.begin += token.offset.begin;
                t.offset.end += token.offset.begin;
                t
            }).collect::<Vec<Token>>());
            true
        }
        None => false
    };
    if !cached {
        //check if we there is a one to one mapping between the original text and the byte-text
        //(for now we just check if sizes are equal, could be improved)
        let exact_offsets: bool = text.chars().count() == token.text.chars().count();

        let bpe_output: Vec<Token> = bpe_function(TokenRef {
            text: text.as_str(),
            offset: Offset { //we reset the offset, so we can cache
                begin: 0,
                end: token.offset.end - token.offset.begin
            },
            mask: Mask::None //will be overwritten anyway
        }, bpe_ranks, exact_offsets);
        cache.borrow_mut().insert(text.to_owned(), bpe_output.clone());
        tokens.extend(bpe_output.into_iter().map(|mut t| {
            //the tokens from the bpe_output have 0-based offsets, adapt the offset
            //according to the input offset
            t.offset.begin += token.offset.begin;
            t.offset.end += token.offset.begin;
            t
        }).collect::<Vec<Token>>())
    }

    tokens

}


pub fn fix_mask(mut tokens: Vec<Token>) -> Vec<Token> {
    //fix mask
    if !tokens.is_empty() {
        if tokens.len() == 1 {
            if tokens[0].mask == Mask::InexactBegin && tokens[0].mask == Mask::Begin {
                tokens[0].mask = Mask::None;
            }
        } else {
            for i in 1..tokens.len() - 1 {
                if ((tokens[i].mask == Mask::InexactBegin || tokens[i].mask == Mask::Begin)) && ((tokens[i-1].mask == Mask::InexactBegin) || (tokens[i-1].mask == Mask::Begin)) {
                    tokens[i-1].mask = Mask::None;
                }
            }
        }
    }
    tokens
}

//==============================
// Unit tests
//==============================
#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::iter::FromIterator;
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
            ("[中]".to_owned(), 7),
            ("华".to_owned(), 8),
            ("人]".to_owned(), 9),
            ("[PAD]".to_owned(), 10),
            ("una".to_owned(), 10),
            ("##ffa".to_owned(), 10),
            ("##ble".to_owned(), 10)
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
    fn test_clean_text() {
//        Given
        let test_tuples = [
            (
                "Sentence with no special character.",
                "Sentence with no special character."
            ),
            (
                "Sentence with \n some \r\n line breaks.",
                "Sentence with   some    line breaks."
            ),
            (
                "Sentence with �replacement character.",
                "Sentence with replacement character."
            ),
            (
                "Sentence with \t \t tabs.",
                "Sentence with     tabs."
            ),
            (
                "Sentence with \x00null character.",
                "Sentence with null character."
            ),
        ];

//        When & Then
        for (source_text, expected_result) in test_tuples.iter() {
            assert_eq!(clean_text(*source_text, true), *expected_result);
        }

        for (source_text, expected_result) in test_tuples.iter() {
            assert_eq!(clean_text(*source_text, false), *expected_result);
        }
    }

    #[test]
    fn test_split_on_special_tokens() {
//        Given
        let vocab = generate_test_vocab();
        let test_tuples = [
            (
                "Sentence with [MASK] token.",
                vec!("Sentence with ", "[MASK]", " token.")
            ),
            (
                "[CLS]Sentence with [MASK] token.",
                vec!("[CLS]", "Sentence with ", "[MASK]", " token.")
            ),
            (
                "[CLS]",
                vec!("[CLS]")
            ),
            (
                "[CLS] [PAD]",
                vec!("[CLS]", " ", "[PAD]")
            ),
            (
                "[CLS]       [PAD]",
                vec!("[CLS]", "       ", "[PAD]")
            ),
            (
                "asdf[CLS]",
                vec!("asdf", "[CLS]")
            ),
            (
                "No special token in sentence",
                vec!("No special token in sentence")
            ),
            (
                "",
                vec!()
            ),
            (
                "[UNK]中华人民共和国 [PAD] asdf",
                vec!("[UNK]", "中华人民共和国 ", "[PAD]", " asdf")
            ),
        ];

//        When & Then
        for (source_text, expected_tokens) in test_tuples.iter() {
            let tokens: Vec<&str> = split_on_special_tokens(TokenRef::new(source_text), &vocab).into_iter().map(|t| t.text).collect();
            assert_eq!(tokens, *expected_tokens);
        }
    }

    #[test]
    fn test_tokenize_cjk_chars() {
//        Given
        let test_tuples = [
            (
                "[UNK]中华人民共和国 [PAD] asdf",
                vec!("[UNK]","中","华","人","民","共","和","国"," [PAD] asdf"),
                vec!(Offset { begin: 0, end: 5 }, Offset { begin: 5, end: 6 }, Offset { begin: 6, end: 7 }, Offset { begin: 7, end: 8 }, Offset { begin: 8, end: 9 }, Offset { begin: 9, end: 10 }, Offset { begin: 10, end: 11 }, Offset { begin: 11, end: 12 }, Offset { begin: 12, end: 23 }),
            ),
            (
                "中",
                vec!("中"),
                vec!(Offset::new(0,1)),
            ),
            (
                "中b华",
                vec!("中","b","华"),
                vec!(Offset::new(0,1), Offset::new(1,2), Offset::new(2,3)),
            ),
            (
                "中[PAD]华",
                vec!("中","[PAD]","华"),
                vec!(Offset::new(0,1), Offset::new(1,6), Offset::new(6,7)),
            ),
        ];

//        When & Then
        for (source_text, expected_tokens, expected_offsets) in test_tuples.iter() {
            let (tokens,offsets): (Vec<&str>,Vec<Offset>) = tokenize_cjk_chars(TokenRef::new(source_text)).into_iter().map(|t| (t.text,t.offset)).unzip();
            assert_eq!(tokens, *expected_tokens);
            assert_eq!(offsets, *expected_offsets);
        }
    }

    #[test]
    fn test_is_cjk_char() {
//        Given
        let chinese_chars = "的是不我一有大在人了中到資要可以這個你會好為上來就學交也用能如文時沒說他看提那問生過下請天們所多麼小想得之還電出工對都機自後子而訊站去心\
        只家知國台很信成章何同道地發法無然但嗎當於本現年前真最和新因果定意情點題其事方清科樣些吧三此位理行作經者什謝名日正華話開實再城愛與二動比高面又車力或種像應女教分手打已次\
        長太明己路起相主關鳳間呢覺該十外凰友才民系進使她著各少全兩回加將感第性球式把被老公龍程論及別給聽水重體做校裡常東風您灣啦見解等部原月美先管區錯音否啊找網樂讓通入期選較四\
        場由書它快從歡數表怎至立內合目望認幾社告更版度考喜頭難光買今身許弟若算記代統處完號接言政玩師字並男計誰山張黨每且結改非星連哈建放直轉報活設變指氣研陳試西五希取神化物王戰\
        近世受義反單死任跟便空林士臺卻北隊功必聲寫平影業金檔片討色容央妳向市則員興利強白價安呵特思叫總辦保花議傳元求份件持萬未究決投哪喔笑貓組獨級走支曾標流竹兄阿室卡馬共需海口\
        門般線語命觀視朋聯參格黃錢修失兒住八腦板吃另換即象料錄拿專遠速基幫形確候裝孩備歌界除南器畫訴差講類英案帶久乎掉迷量引整似耶奇制邊型超識雖怪飛始品運賽費夢故班權破驗眼滿念\
        造軍精務留服六圖收舍半讀願李底約雄課答令深票達演早賣棒夠黑院假曲火準百談勝碟術推存治離易往況晚示證段導傷調團七永剛哥甚德殺怕包列概照夜排客絕軟商根九切條集千落竟越待忘盡\
        據雙供稱座值消產紅跑嘛園附硬雲遊展執聞唱育斯某技唉息苦質油救效須介首助職例熱畢節害擊亂態嗯寶倒注停古輸規福親查復步舉魚斷終輕環練印隨依趣限響省局續司角簡極幹篇羅佛克陽武\
        疑送拉習源免志鳥煩足館仍低廣土呀樓壞兵顯率聖碼眾爭初誤楚責境野預具智壓係青貴順負魔適哇測慢懷懂史配嗚味亦醫迎舞戀細灌甲帝句屬靈評騎宜敗左追狂敢春狗際遇族群痛右康佳楊木病\
        戲項抓徵善官護博補石爾營歷隻按妹里編歲擇溫守血領尋田養謂居異雨止跳君爛優封拜惡啥浪核聊急狀陸激模攻忙良劇牛壘增維靜陣抱勢嚴詞亞夫簽悲密幕毒廠爽緣店吳蘭睡致江宿翻香蠻警控\
        趙冷威微坐週宗普登母絡午恐套巴雜創舊輯幸劍亮述堂酒麗牌仔腳突搞父俊暴防吉禮素招草周房餐慮充府背典仁漫景紹諸琴憶援尤缺扁罵純惜授皮松委湖誠麻置靠繼判益波姐既射欲刻堆釋含承\
        退莫劉昨旁紀趕製尚藝肉律鐵奏樹毛罪筆彩註歸彈虎衛刀皆鍵售塊險榮播施銘囉漢賞欣升葉螢載嘿弄鐘付寄鬼哦燈呆洋嘻布磁薦檢派構媽藍貼豬策紙暗巧努雷架享宣逢均擔啟濟罷呼劃偉島歉郭\
        訓穿詳沙督梅顧敵".chars();

        let japanese_chars = "一九七二人入八力十下三千上口土夕大女子小山川五天中六円手文日月木水火犬王正出本右四左玉生田白目石立百年休先名字早気竹糸耳虫村男町花見貝赤\
        足車学林空金雨青草音校森刀万丸才工弓内午少元今公分切友太引心戸方止毛父牛半市北古台兄冬外広母用矢交会合同回寺地多光当毎池米羽考肉自色行西来何作体弟図声売形汽社角言谷走近\
        里麦画東京夜直国姉妹岩店明歩知長門昼前南点室後春星海活思科秋茶計風食首夏弱原家帰時紙書記通馬高強教理細組船週野雪魚鳥黄黒場晴答絵買朝道番間雲園数新楽話遠電鳴歌算語読聞線\
        親頭曜顔丁予化区反央平申世由氷主仕他代写号去打皮皿礼両曲向州全次安守式死列羊有血住助医君坂局役投対決究豆身返表事育使命味幸始実定岸所放昔板泳注波油受物具委和者取服苦重乗\
        係品客県屋炭度待急指持拾昭相柱洋畑界発研神秒級美負送追面島勉倍真員宮庫庭旅根酒消流病息荷起速配院悪商動宿帳".chars();

//        When & Then
        for character in chinese_chars {
            assert!(is_cjk_char(&character));
        }

        for character in japanese_chars {
            assert!(is_cjk_char(&character));
        }
    }

    #[test]
    fn test_is_whitespace() {
//        Given
        let whitespace_chars: [u32; 17] = [
            0x0020, 0x00A0, 0x1680, 0x2000, 0x2001, 0x2002, 0x2003,
            0x2004, 0x2005, 0x2006, 0x2007, 0x2008, 0x2009, 0x200A,
            0x202F, 0x205F, 0x3000,
        ];

        let additional_whitespace_chars: [char; 4] = [
            ' ', '\n', '\r', '\t'
        ];

        let non_whitespace_chars: [char; 5] = ['a', '5', '♥', '_', '越'];

//        When & Then
        for character in whitespace_chars.iter() {
            assert!(is_whitespace(&char::from_u32(*character).unwrap()));
        }

        for character in additional_whitespace_chars.iter() {
            assert!(is_whitespace(character));
        }

        for character in non_whitespace_chars.iter() {
            assert!(!is_whitespace(character));
        }
    }

    #[test]
    fn test_is_control() {
//        Given
        let standard_control_chars_without_space_return_tab: [u32; 62] = [
            0x0000, 0x0001, 0x0002, 0x0003, 0x0004, 0x0005, 0x0006, 0x0007, 0x0008, 0x000B,
            0x000C, 0x000E, 0x000F, 0x0010, 0x0011, 0x0012, 0x0013, 0x0014, 0x0015, 0x0016,
            0x0017, 0x0018, 0x0019, 0x001A, 0x001B, 0x001C, 0x001D, 0x001E, 0x001F, 0x007F,
            0x0080, 0x0081, 0x0082, 0x0083, 0x0084, 0x0085, 0x0086, 0x0087, 0x0088, 0x0089,
            0x008A, 0x008B, 0x008C, 0x008D, 0x008E, 0x008F, 0x0090, 0x0091, 0x0092, 0x0093,
            0x0094, 0x0095, 0x0096, 0x0097, 0x0098, 0x0099, 0x009A, 0x009B, 0x009C, 0x009D,
            0x009E, 0x009F
        ];

        let extended_control_chars: [u32; 223] = [
            0x0000, 0x0001, 0x0002, 0x0003, 0x0004, 0x0005, 0x0006, 0x0007, 0x0008, 0x000B, 0x000C,
            0x000E, 0x000F, 0x0010, 0x0011, 0x0012, 0x0013, 0x0014, 0x0015, 0x0016, 0x0017, 0x0018,
            0x0019, 0x001A, 0x001B, 0x001C, 0x001D, 0x001E, 0x001F, 0x007F, 0x0080, 0x0081, 0x0082,
            0x0083, 0x0084, 0x0085, 0x0086, 0x0087, 0x0088, 0x0089, 0x008A,
            0x008B, 0x008C, 0x008D, 0x008E, 0x008F, 0x0090, 0x0091, 0x0092, 0x0093, 0x0094, 0x0095,
            0x0096, 0x0097, 0x0098, 0x0099, 0x009A, 0x009B, 0x009C, 0x009D, 0x009E, 0x009F, 0x00AD,
            0x0600, 0x0601, 0x0602, 0x0603, 0x0604, 0x0605, 0x061C, 0x06DD, 0x070F, 0x08E2, 0x180E,
            0x200B, 0x200C, 0x200D, 0x200E, 0x200F, 0x202A, 0x202B, 0x202C, 0x202D, 0x202E, 0x2060,
            0x2061, 0x2062, 0x2063, 0x2064, 0x2066, 0x2067, 0x2068, 0x2069, 0x206A, 0x206B, 0x206C,
            0x206D, 0x206E, 0x206F, 0xFEFF, 0xFFF9, 0xFFFA, 0xFFFB, 0x110BD, 0x110CD, 0x13430,
            0x13431, 0x13432, 0x13433, 0x13434, 0x13435, 0x13436, 0x13437, 0x13438, 0x1BCA0, 0x1BCA1,
            0x1BCA2, 0x1BCA3, 0x1D173, 0x1D174, 0x1D175, 0x1D176, 0x1D177, 0x1D178, 0x1D179, 0x1D17A,
            0xE0001, 0xE0020, 0xE0021, 0xE0022, 0xE0023, 0xE0024, 0xE0025, 0xE0026, 0xE0027, 0xE0028,
            0xE0029, 0xE002A, 0xE002B, 0xE002C, 0xE002D, 0xE002E, 0xE002F, 0xE0030, 0xE0031, 0xE0032,
            0xE0033, 0xE0034, 0xE0035, 0xE0036, 0xE0037, 0xE0038, 0xE0039, 0xE003A, 0xE003B, 0xE003C,
            0xE003D, 0xE003E, 0xE003F, 0xE0040, 0xE0041, 0xE0042, 0xE0043, 0xE0044, 0xE0045, 0xE0046,
            0xE0047, 0xE0048, 0xE0049, 0xE004A, 0xE004B, 0xE004C, 0xE004D, 0xE004E, 0xE004F, 0xE0050,
            0xE0051, 0xE0052, 0xE0053, 0xE0054, 0xE0055, 0xE0056, 0xE0057, 0xE0058, 0xE0059, 0xE005A,
            0xE005B, 0xE005C, 0xE005D, 0xE005E, 0xE005F, 0xE0060, 0xE0061, 0xE0062, 0xE0063, 0xE0064,
            0xE0065, 0xE0066, 0xE0067, 0xE0068, 0xE0069, 0xE006A, 0xE006B, 0xE006C, 0xE006D, 0xE006E,
            0xE006F, 0xE0070, 0xE0071, 0xE0072, 0xE0073, 0xE0074, 0xE0075, 0xE0076, 0xE0077, 0xE0078,
            0xE0079, 0xE007A, 0xE007B, 0xE007C, 0xE007D, 0xE007E, 0xE007F
        ];

        let additional_whitespace_chars: [char; 4] = [
            ' ', '\n', '\r', '\t'
        ];

        let non_control_chars: [char; 5] = ['a', '5', '♥', '_', '越'];

//        When & Then
        for character in standard_control_chars_without_space_return_tab.iter() {
            assert!(is_control(&char::from_u32(*character).unwrap(), true));
        }

        for character in extended_control_chars.iter() {
            assert!(is_control(&char::from_u32(*character).unwrap(), true));
        }

        for character in additional_whitespace_chars.iter() {
            assert!(!is_control(character, true));
        }

        for character in non_control_chars.iter() {
            assert!(!is_control(character, true));
        }

        for character in standard_control_chars_without_space_return_tab.iter() {
            assert!(is_control(&char::from_u32(*character).unwrap(), false));
        }

        for character in additional_whitespace_chars.iter() {
            assert!(!is_control(character, false));
        }

        for character in non_control_chars.iter() {
            assert!(!is_control(character, false));
        }
    }

    #[test]
    fn test_is_punctuation() {
        let punctuation_category_chars = [
            0x21, 0x22, 0x23, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x2c, 0x2d, 0x2e, 0x2f, 0x3a, 0x3b, 0x3f, 0x40, 0x5b, 0x5c, 0x5d, 0x5f, 0x7b, 0x7d, 0xa1, 0xa7,
            0xab, 0xb6, 0xb7, 0xbb, 0xbf, 0x37e, 0x387, 0x55a, 0x55b, 0x55c, 0x55d, 0x55e, 0x55f, 0x589, 0x58a, 0x5be, 0x5c0, 0x5c3, 0x5c6, 0x5f3, 0x5f4, 0x609,
            0x60a, 0x60c, 0x60d, 0x61b, 0x61e, 0x61f, 0x66a, 0x66b, 0x66c, 0x66d, 0x6d4, 0x700, 0x701, 0x702, 0x703, 0x704, 0x705, 0x706, 0x707, 0x708, 0x709, 0x70a,
            0x70b, 0x70c, 0x70d, 0x7f7, 0x7f8, 0x7f9, 0x830, 0x831, 0x832, 0x833, 0x834, 0x835, 0x836, 0x837, 0x838, 0x839, 0x83a, 0x83b, 0x83c, 0x83d, 0x83e, 0x85e,
            0x964, 0x965, 0x970, 0x9fd, 0xa76, 0xaf0, 0xc84, 0xdf4, 0xe4f, 0xe5a, 0xe5b, 0xf04, 0xf05, 0xf06, 0xf07, 0xf08, 0xf09, 0xf0a, 0xf0b, 0xf0c, 0xf0d, 0xf0e,
            0xf0f, 0xf10, 0xf11, 0xf12, 0xf14, 0xf3a, 0xf3b, 0xf3c, 0xf3d, 0xf85, 0xfd0, 0xfd1, 0xfd2, 0xfd3, 0xfd4, 0xfd9, 0xfda, 0x104a, 0x104b, 0x104c, 0x104d,
            0x104e, 0x104f, 0x10fb, 0x1360, 0x1361, 0x1362, 0x1363, 0x1364, 0x1365, 0x1366, 0x1367, 0x1368, 0x1400, 0x166d, 0x166e, 0x169b, 0x169c, 0x16eb, 0x16ec,
            0x16ed, 0x1735, 0x1736, 0x17d4, 0x17d5, 0x17d6, 0x17d8, 0x17d9, 0x17da, 0x1800, 0x1801, 0x1802, 0x1803, 0x1804, 0x1805, 0x1806, 0x1807, 0x1808, 0x1809,
            0x180a, 0x1944, 0x1945, 0x1a1e, 0x1a1f, 0x1aa0, 0x1aa1, 0x1aa2, 0x1aa3, 0x1aa4, 0x1aa5, 0x1aa6, 0x1aa8, 0x1aa9, 0x1aaa, 0x1aab, 0x1aac, 0x1aad, 0x1b5a,
            0x1b5b, 0x1b5c, 0x1b5d, 0x1b5e, 0x1b5f, 0x1b60, 0x1bfc, 0x1bfd, 0x1bfe, 0x1bff, 0x1c3b, 0x1c3c, 0x1c3d, 0x1c3e, 0x1c3f, 0x1c7e, 0x1c7f, 0x1cc0, 0x1cc1,
            0x1cc2, 0x1cc3, 0x1cc4, 0x1cc5, 0x1cc6, 0x1cc7, 0x1cd3, 0x2010, 0x2011, 0x2012, 0x2013, 0x2014, 0x2015, 0x2016, 0x2017, 0x2018, 0x2019, 0x201a, 0x201b,
            0x201c, 0x201d, 0x201e, 0x201f, 0x2020, 0x2021, 0x2022, 0x2023, 0x2024, 0x2025, 0x2026, 0x2027, 0x2030, 0x2031, 0x2032, 0x2033, 0x2034, 0x2035, 0x2036,
            0x2037, 0x2038, 0x2039, 0x203a, 0x203b, 0x203c, 0x203d, 0x203e, 0x203f, 0x2040, 0x2041, 0x2042, 0x2043, 0x2045, 0x2046, 0x2047, 0x2048, 0x2049, 0x204a,
            0x204b, 0x204c, 0x204d, 0x204e, 0x204f, 0x2050, 0x2051, 0x2053, 0x2054, 0x2055, 0x2056, 0x2057, 0x2058, 0x2059, 0x205a, 0x205b, 0x205c, 0x205d, 0x205e,
            0x207d, 0x207e, 0x208d, 0x208e, 0x2308, 0x2309, 0x230a, 0x230b, 0x2329, 0x232a, 0x2768, 0x2769, 0x276a, 0x276b, 0x276c, 0x276d, 0x276e, 0x276f, 0x2770,
            0x2771, 0x2772, 0x2773, 0x2774, 0x2775, 0x27c5, 0x27c6, 0x27e6, 0x27e7, 0x27e8, 0x27e9, 0x27ea, 0x27eb, 0x27ec, 0x27ed, 0x27ee, 0x27ef, 0x2983, 0x2984,
            0x2985, 0x2986, 0x2987, 0x2988, 0x2989, 0x298a, 0x298b, 0x298c, 0x298d, 0x298e, 0x298f, 0x2990, 0x2991, 0x2992, 0x2993, 0x2994, 0x2995, 0x2996, 0x2997,
            0x2998, 0x29d8, 0x29d9, 0x29da, 0x29db, 0x29fc, 0x29fd, 0x2cf9, 0x2cfa, 0x2cfb, 0x2cfc, 0x2cfe, 0x2cff, 0x2d70, 0x2e00, 0x2e01, 0x2e02, 0x2e03, 0x2e04,
            0x2e05, 0x2e06, 0x2e07, 0x2e08, 0x2e09, 0x2e0a, 0x2e0b, 0x2e0c, 0x2e0d, 0x2e0e, 0x2e0f, 0x2e10, 0x2e11, 0x2e12, 0x2e13, 0x2e14, 0x2e15, 0x2e16, 0x2e17,
            0x2e18, 0x2e19, 0x2e1a, 0x2e1b, 0x2e1c, 0x2e1d, 0x2e1e, 0x2e1f, 0x2e20, 0x2e21, 0x2e22, 0x2e23, 0x2e24, 0x2e25, 0x2e26, 0x2e27, 0x2e28, 0x2e29, 0x2e2a,
            0x2e2b, 0x2e2c, 0x2e2d, 0x2e2e, 0x2e30, 0x2e31, 0x2e32, 0x2e33, 0x2e34, 0x2e35, 0x2e36, 0x2e37, 0x2e38, 0x2e39, 0x2e3a, 0x2e3b, 0x2e3c, 0x2e3d, 0x2e3e,
            0x2e3f, 0x2e40, 0x2e41, 0x2e42, 0x2e43, 0x2e44, 0x2e45, 0x2e46, 0x2e47, 0x2e48, 0x2e49, 0x2e4a, 0x2e4b, 0x2e4c, 0x2e4d, 0x2e4e, 0x3001, 0x3002, 0x3003,
            0x3008, 0x3009, 0x300a, 0x300b, 0x300c, 0x300d, 0x300e, 0x300f, 0x3010, 0x3011, 0x3014, 0x3015, 0x3016, 0x3017, 0x3018, 0x3019, 0x301a, 0x301b, 0x301c,
            0x301d, 0x301e, 0x301f, 0x3030, 0x303d, 0x30a0, 0x30fb, 0xa4fe, 0xa4ff, 0xa60d, 0xa60e, 0xa60f, 0xa673, 0xa67e, 0xa6f2, 0xa6f3, 0xa6f4, 0xa6f5, 0xa6f6,
            0xa6f7, 0xa874, 0xa875, 0xa876, 0xa877, 0xa8ce, 0xa8cf, 0xa8f8, 0xa8f9, 0xa8fa, 0xa8fc, 0xa92e, 0xa92f, 0xa95f, 0xa9c1, 0xa9c2, 0xa9c3, 0xa9c4, 0xa9c5,
            0xa9c6, 0xa9c7, 0xa9c8, 0xa9c9, 0xa9ca, 0xa9cb, 0xa9cc, 0xa9cd, 0xa9de, 0xa9df, 0xaa5c, 0xaa5d, 0xaa5e, 0xaa5f, 0xaade, 0xaadf, 0xaaf0, 0xaaf1, 0xabeb,
            0xfd3e, 0xfd3f, 0xfe10, 0xfe11, 0xfe12, 0xfe13, 0xfe14, 0xfe15, 0xfe16, 0xfe17, 0xfe18, 0xfe19, 0xfe30, 0xfe31, 0xfe32, 0xfe33, 0xfe34, 0xfe35, 0xfe36,
            0xfe37, 0xfe38, 0xfe39, 0xfe3a, 0xfe3b, 0xfe3c, 0xfe3d, 0xfe3e, 0xfe3f, 0xfe40, 0xfe41, 0xfe42, 0xfe43, 0xfe44, 0xfe45, 0xfe46, 0xfe47, 0xfe48, 0xfe49,
            0xfe4a, 0xfe4b, 0xfe4c, 0xfe4d, 0xfe4e, 0xfe4f, 0xfe50, 0xfe51, 0xfe52, 0xfe54, 0xfe55, 0xfe56, 0xfe57, 0xfe58, 0xfe59, 0xfe5a, 0xfe5b, 0xfe5c, 0xfe5d,
            0xfe5e, 0xfe5f, 0xfe60, 0xfe61, 0xfe63, 0xfe68, 0xfe6a, 0xfe6b, 0xff01, 0xff02, 0xff03, 0xff05, 0xff06, 0xff07, 0xff08, 0xff09, 0xff0a, 0xff0c, 0xff0d,
            0xff0e, 0xff0f, 0xff1a, 0xff1b, 0xff1f, 0xff20, 0xff3b, 0xff3c, 0xff3d, 0xff3f, 0xff5b, 0xff5d, 0xff5f, 0xff60, 0xff61, 0xff62, 0xff63, 0xff64, 0xff65,
            0x10100, 0x10101, 0x10102, 0x1039f, 0x103d0, 0x1056f, 0x10857, 0x1091f, 0x1093f, 0x10a50, 0x10a51, 0x10a52, 0x10a53, 0x10a54, 0x10a55, 0x10a56, 0x10a57,
            0x10a58, 0x10a7f, 0x10af0, 0x10af1, 0x10af2, 0x10af3, 0x10af4, 0x10af5, 0x10af6, 0x10b39, 0x10b3a, 0x10b3b, 0x10b3c, 0x10b3d, 0x10b3e, 0x10b3f, 0x10b99,
            0x10b9a, 0x10b9b, 0x10b9c, 0x10f55, 0x10f56, 0x10f57, 0x10f58, 0x10f59, 0x11047, 0x11048, 0x11049, 0x1104a, 0x1104b, 0x1104c, 0x1104d, 0x110bb, 0x110bc,
            0x110be, 0x110bf, 0x110c0, 0x110c1, 0x11140, 0x11141, 0x11142, 0x11143, 0x11174, 0x11175, 0x111c5, 0x111c6, 0x111c7, 0x111c8, 0x111cd, 0x111db, 0x111dd,
            0x111de, 0x111df, 0x11238, 0x11239, 0x1123a, 0x1123b, 0x1123c, 0x1123d, 0x112a9, 0x1144b, 0x1144c, 0x1144d, 0x1144e, 0x1144f, 0x1145b, 0x1145d, 0x114c6,
            0x115c1, 0x115c2, 0x115c3, 0x115c4, 0x115c5, 0x115c6, 0x115c7, 0x115c8, 0x115c9, 0x115ca, 0x115cb, 0x115cc, 0x115cd, 0x115ce, 0x115cf, 0x115d0, 0x115d1,
            0x115d2, 0x115d3, 0x115d4, 0x115d5, 0x115d6, 0x115d7, 0x11641, 0x11642, 0x11643, 0x11660, 0x11661, 0x11662, 0x11663, 0x11664, 0x11665, 0x11666, 0x11667,
            0x11668, 0x11669, 0x1166a, 0x1166b, 0x1166c, 0x1173c, 0x1173d, 0x1173e, 0x1183b, 0x11a3f, 0x11a40, 0x11a41, 0x11a42, 0x11a43, 0x11a44, 0x11a45, 0x11a46,
            0x11a9a, 0x11a9b, 0x11a9c, 0x11a9e, 0x11a9f, 0x11aa0, 0x11aa1, 0x11aa2, 0x11c41, 0x11c42, 0x11c43, 0x11c44, 0x11c45, 0x11c70, 0x11c71, 0x11ef7, 0x11ef8,
            0x12470, 0x12471, 0x12472, 0x12473, 0x12474, 0x16a6e, 0x16a6f, 0x16af5, 0x16b37, 0x16b38, 0x16b39, 0x16b3a, 0x16b3b, 0x16b44, 0x16e97, 0x16e98, 0x16e99,
            0x16e9a, 0x1bc9f, 0x1da87, 0x1da88, 0x1da89, 0x1da8a, 0x1da8b, 0x1e95e, 0x1e95f
        ];

        let common_punctuation_chars = "!@$#%^&*()/?;:.,{}[]-_=+".chars();

        let non_punctuation_chars: [char; 5] = ['a', '5', '♥', '\t', '越'];

//        When & Then
        for character in punctuation_category_chars.iter() {
            assert!(is_punctuation(&char::from_u32(*character).unwrap()));
        }

        for character in common_punctuation_chars {
            assert!(is_punctuation(&character));
        }

        for character in non_punctuation_chars.iter() {
            assert!(!is_punctuation(&character));
        }
    }

    #[test]
    fn test_whitespace_tokenize() {
//        Given
        let test_tuples = [
            (
                "Sentence with 4 tokens.",
                (
                    vec!("Sentence", "with", "4", "tokens."),
                    vec!( Offset::new(0, 8), Offset::new(9 , 13), Offset::new(14, 15), Offset::new(16, 23))
                )

            ),
            (
                "Nowhitespacesinthissentence.",
                (
                    vec!("Nowhitespacesinthissentence."),
                    vec!(Offset::new(0,28))
                )
            ),
            (
                "Tab\tSeparated\tSentence", //behaviour changed, tabs are considered whitespace now!
                (
                    vec!("Tab","Separated","Sentence"),
                    vec!(Offset::new(0,3), Offset::new(4,13), Offset::new(14, 22) )
                )
            ),
            (
                "Newlines\nseparated\nsentence", //behaviour changed, newlines are considered whitespace now
                (
                    vec!("Newlines","separated","sentence"),
                    vec!(Offset::new(0,8), Offset::new(9,18), Offset::new(19,27))
                )
            ),
            (
                "Sentence with �replacement character.",
                (
                    vec!("Sentence","with","�replacement","character."), //is removed at a later stage, not in tokenize_with_offsets()
                    vec!(Offset::new(0,8), Offset::new(9,13), Offset::new(14,26), Offset::new(27,37))
                )
            ),
            (
                " leading and trailing spaces           ",
                (
                    vec!("leading", "and", "trailing", "spaces"),
                    vec!(Offset::new(1,8), Offset::new(9,12), Offset::new(13,21), Offset::new(22, 28))
                )
            ),
            (
                " Multiple spaces   in-between           ",
                (
                    vec!("Multiple", "spaces", "in-between"),
                    vec!(Offset::new(1,9), Offset::new(10,16), Offset::new(19,29) )
                )
            )
        ];

//        When & Then
        for (source_text, expected_result) in test_tuples.iter() {
            let (tokens,offsets): (Vec<&str>,Vec<Offset>) = whitespace_tokenize(TokenRef::new(source_text)).into_iter().map(|t| (t.text,t.offset)).unzip();
            assert_eq!(tokens, expected_result.0);
            assert_eq!(offsets, expected_result.1);
        }
    }

    #[test]
    fn test_strip_accents() {
        let test_tuples = [
            (
                "No accent here",
                "No accent here"
            ),
            (
                "çà",
                "ca"
            ),
            (
                "àgbọ̀n",
                "agbon"
            ),
            (
                "cùis",
                "cuis"
            ),
            (
                "Tiếng Việt",
                "Tieng Viet"
            ),
            (
                "château",
                "chateau"
            ),
            (
                "München",
                "Munchen"
            ),
        ];

//        When & Then
        for (source_text, expected_result) in test_tuples.iter() {
            assert_eq!(strip_accents(String::from(*source_text)), String::from(*expected_result));
        }
    }

    #[test]
    fn test_split_on_punct() {
//        Given
        let test_tuples = [
            (
                "Sentence One. Sentence Two",
                vec!("Sentence One", ".", " Sentence Two"),
                vec!(Offset::new(0,12), Offset::new(12,13), Offset::new(13,26))
            ),
            (
                "Sentence One.Sentence Two",
                vec!("Sentence One", ".", "Sentence Two"),
                vec!(Offset::new(0,12), Offset::new(12,13), Offset::new(13,25))
            ),
            (
                "Sentence One.!?Sentence Two",
                vec!("Sentence One", ".", "!", "?", "Sentence Two"),
                vec!(Offset::new(0,12), Offset::new(12,13), Offset::new(13,14), Offset::new(14,15), Offset::new(15,27))
            ),
        ];

//        When & Then
        for (source_text, expected_tokens, expected_offsets) in test_tuples.iter() {
            let (tokens,offsets): (Vec<String>,Vec<Offset>) = split_on_punct(TokenRef::new(source_text)).into_iter().map(|t| (t.text.to_string(),t.offset)).unzip();
            assert_eq!(tokens, expected_tokens.iter().map(|v| String::from(*v)).collect::<Vec<_>>());
            assert_eq!(offsets, *expected_offsets);
        }
    }

    #[test]
    fn test_wordpiece_tokenizer() {
//        Given
        let vocab = generate_test_vocab();
        let test_tuples = [
            (
                "unaffable",
                vec!("una", "##ffa", "##ble"),
                vec!(Offset::new(0,3), Offset::new(3,6), Offset::new(6,9) )
            ),
            (
                "hello",
                vec!("hello"),
                vec!(Offset::new(0,5))
            ),
            (
                "[PAD]",
                vec!("[PAD]"),
                vec!(Offset::new(0,5))
            ),
            (
                "51",
                vec!("[UNK]"),
                vec!(Offset::new(0,2))
            ),
        ];

//        When & Then
        for (source_text, expected_tokens, expected_offsets) in test_tuples.iter() {
            let (tokens,offsets): (Vec<String>,Vec<Offset>) = tokenize_wordpiece(TokenRef::new(source_text),&vocab, 100).into_iter().map(|t| (t.text,t.offset)).unzip();
            assert_eq!(tokens, expected_tokens.iter().map(|v| String::from(*v)).collect::<Vec<_>>());
            assert_eq!(offsets, *expected_offsets);
        }
    }

    #[test]
    fn test_truncate_single_sentence() {
//        Given
        let test_token_ids: Vec<i64> = (0..15).collect();
        let test_tuples: [((usize, &TruncationStrategy, usize), std::result::Result<(Vec<i64>, std::option::Option<Vec<i64>>, Vec<Offset>, Option<Vec<Offset>>, Vec<Mask>, Option<Vec<Mask>>, Vec<i64>, Vec<Offset>), Box<dyn Error>>);
            12] = [
//            Baseline
            (
                (5, &TruncationStrategy::LongestFirst, 0),
                Ok(((0..10).collect::<Vec<i64>>(), None::<Vec<i64>>, vec!(), None,vec!(), None, (10..15).collect::<Vec<i64>>(), vec!()))
            ),
//            With stride = 2
            (
                (5, &TruncationStrategy::LongestFirst, 2),
                Ok(((0..10).collect::<Vec<i64>>(), None::<Vec<i64>>, vec!(), None,vec!(), None, (8..15).collect::<Vec<i64>>(), vec!()))
            ),
//            Truncate entire sequence
            (
                (15, &TruncationStrategy::LongestFirst, 0),
                Ok(((0..0).collect::<Vec<i64>>(), None::<Vec<i64>>, vec!(), None ,vec!(), None, (0..15).collect::<Vec<i64>>(), vec!()))
            ),
//            Truncate amount larger than sequence length
            (
                (20, &TruncationStrategy::LongestFirst, 0),
                Err("First sequence too short for first only truncation".into())
            ),
//            Truncate entire sequence with stride = 2
            (
                (15, &TruncationStrategy::LongestFirst, 2),
                Ok(((0..0).collect::<Vec<i64>>(), None::<Vec<i64>>, vec!(), None ,vec!(), None, (0..15).collect::<Vec<i64>>(), vec!()))
            ),
//            stride larger than remaining elements
            (
                (10, &TruncationStrategy::LongestFirst, 7),
                Ok(((0..5).collect::<Vec<i64>>(), None::<Vec<i64>>, vec!(), None ,vec!(), None, (0..15).collect::<Vec<i64>>(), vec!()))
            ),
//            stride larger than all elements
            (
                (1, &TruncationStrategy::LongestFirst, 20),
                Ok(((0..14).collect::<Vec<i64>>(), None::<Vec<i64>>, vec!(), None,vec!(), None, (0..15).collect::<Vec<i64>>(), Vec::<Offset>::new()))
            ),
//            Truncate with OnlyFirst strategy
            (
                (10, &TruncationStrategy::OnlyFirst, 2),
                Ok(((0..5).collect::<Vec<i64>>(), None::<Vec<i64>>, vec!(), None ,vec!(), None, (3..15).collect::<Vec<i64>>(), vec!()))
            ),
//            No truncation
            (
                (0, &TruncationStrategy::LongestFirst, 0),
                Ok(((0..15).collect::<Vec<i64>>(), None::<Vec<i64>>, vec!(), None,vec!(), None, (15..15).collect::<Vec<i64>>(), vec!()))
            ),
//            No truncation requested, none needed
            (
                (0, &TruncationStrategy::DoNotTruncate, 0),
                Ok(((0..15).collect::<Vec<i64>>(), None::<Vec<i64>>, vec!(), None,vec!(), None, (15..15).collect::<Vec<i64>>(), vec!()))
            ),
//            No truncation requested, but needed
            (
                (1, &TruncationStrategy::DoNotTruncate, 0),
                Err("Truncation needed but no truncation requested".into())
            ),
//            Invalid truncation requested
            (
                (1, &TruncationStrategy::OnlySecond, 0),
                Err("Invalid truncation strategy for single sentence truncation".into())
            ),
        ];

        for (parameters, expected_outputs) in &test_tuples {
            let test_results = truncate_sequences(test_token_ids.clone(), None, vec!(), None, vec!(), None, parameters.0, parameters.1, parameters.2);
            match test_results {
                Ok(value) => assert_eq!(value, *expected_outputs.as_ref().unwrap()),
                Err(e) => assert_eq!(e.to_string(), (**expected_outputs.as_ref().err().unwrap()).to_string())
            }
        }
    }

    #[test]
    fn test_truncate_sentence_pair_longest_first() {
//        Given
        let test_token_ids: Vec<i64> = (0..15).collect();
        let test_pair_token_ids: Vec<i64> = (42..51).collect();
        let test_tuples: [((usize, &TruncationStrategy, usize), std::result::Result<(Vec<i64>, std::option::Option<Vec<i64>>, Vec<Offset>, Option<Vec<Offset>>, Vec<Mask>, Option<Vec<Mask>>, Vec<i64>, Vec<Offset>), Box<dyn Error>>);
            10] = [
//            Baseline
            (
                (5, &TruncationStrategy::LongestFirst, 0),
                Ok(((0..10).collect::<Vec<i64>>(), Some((42..51).collect::<Vec<i64>>()), vec!(),None,vec!(), None, (10..15).collect::<Vec<i64>>(), vec![]))
            ),
//            With stride = 2
            (
                (5, &TruncationStrategy::LongestFirst, 2),
                Ok(((0..10).collect::<Vec<i64>>(), Some((42..51).collect::<Vec<i64>>()), vec!(),None,vec!(), None, (8..15).collect::<Vec<i64>>(), vec![]))
            ),
//            Maximum value for only sentence 1 to be affected
            (
                (7, &TruncationStrategy::LongestFirst, 2),
                Ok(((0..8).collect::<Vec<i64>>(), Some((42..51).collect::<Vec<i64>>()), vec!(), None,vec!(), None, (6..15).collect::<Vec<i64>>(), vec![]))
            ),
//            Both sentences affected
            (
                (10, &TruncationStrategy::LongestFirst, 2),
                Ok(((0..7).collect::<Vec<i64>>(), Some((42..49).collect::<Vec<i64>>()), vec!(), None,vec!(), None, (5..15).collect::<Vec<i64>>(), vec![]))
            ),
//            Truncate entire sentence 1
            (
                (15 + 8, &TruncationStrategy::LongestFirst, 2),
                Ok(((0..0).collect::<Vec<i64>>(), Some((42..43).collect::<Vec<i64>>()), vec!(), None,vec!(), None, (0..15).collect::<Vec<i64>>(), vec![]))
            ),
//            Truncate both sentences entirely
            (
                (15 + 9, &TruncationStrategy::LongestFirst, 2),
                Ok(((0..0).collect::<Vec<i64>>(), Some((42..42).collect::<Vec<i64>>()), vec!(), None,vec!(), None, (0..15).collect::<Vec<i64>>(), vec![]))
            ),
//            Request truncation amount greater than combined length
            (
                (15 + 9 + 1, &TruncationStrategy::LongestFirst, 2),
                Err("Combined sequence length too short for requested truncation amount".into())
            ),
//            No truncation
            (
                (0, &TruncationStrategy::LongestFirst, 2),
                Ok(((0..15).collect::<Vec<i64>>(), Some((42..51).collect::<Vec<i64>>()), vec!(), None, vec!(), None,  (15..15).collect::<Vec<i64>>(), vec![]))
            ),
//            No truncation requested, none needed
            (
                (0, &TruncationStrategy::DoNotTruncate, 0),
                Ok(((0..15).collect::<Vec<i64>>(), Some((42..51).collect::<Vec<i64>>()),  vec!(),None, vec!(), None,   (15..15).collect::<Vec<i64>>(), vec![]))
            ),
//            No truncation requested, but needed
            (
                (1, &TruncationStrategy::DoNotTruncate, 0),
                Err("Truncation needed but no truncation requested".into())
            ),
        ];

        for (parameters, expected_outputs) in &test_tuples {
            let test_results = truncate_sequences(test_token_ids.clone(), Some(test_pair_token_ids.clone()), vec!(), None, vec!(), None, parameters.0, parameters.1, parameters.2);
            match test_results {
                Ok(value) => {
                    assert_eq!(value, *expected_outputs.as_ref().unwrap());
                },
                Err(e) => assert_eq!(e.to_string(), (**expected_outputs.as_ref().err().unwrap()).to_string())
            }
        }
    }

    #[test]
    fn test_truncate_sentence_pair_first_only() {
//        Given
        let test_token_ids: Vec<i64> = (0..15).collect();
        let test_pair_token_ids: Vec<i64> = (42..51).collect();
        let test_tuples: [((usize, &TruncationStrategy, usize), std::result::Result<(Vec<i64>, std::option::Option<Vec<i64>>, Vec<Offset>, Option<Vec<Offset>>, Vec<Mask>, Option<Vec<Mask>>, Vec<i64>, Vec<Offset>), Box<dyn Error>>);
            5] = [
//            Baseline
            (
                (5, &TruncationStrategy::OnlyFirst, 0),
                Ok(((0..10).collect::<Vec<i64>>(), Some((42..51).collect::<Vec<i64>>()), vec!(),None ,vec!(), None, (10..15).collect::<Vec<i64>>(), vec![]))
            ),
//            With stride = 2
            (
                (5, &TruncationStrategy::OnlyFirst, 2),
                Ok(((0..10).collect::<Vec<i64>>(), Some((42..51).collect::<Vec<i64>>()), vec!(),None ,vec!(), None, (8..15).collect::<Vec<i64>>(), vec![]))
            ),
//            Truncate entire sentence 1
            (
                (15, &TruncationStrategy::OnlyFirst, 2),
                Ok(((0..0).collect::<Vec<i64>>(), Some((42..51).collect::<Vec<i64>>()), vec!(), None ,vec!(), None, (0..15).collect::<Vec<i64>>(), vec![]))
            ),
//            Request truncation amount greater than sentence 1
            (
                (16, &TruncationStrategy::OnlyFirst, 2),
                Err("First sequence too short for first only truncation".into())
            ),
//            No truncation
            (
                (0, &TruncationStrategy::OnlyFirst, 2),
                Ok(((0..15).collect::<Vec<i64>>(), Some((42..51).collect::<Vec<i64>>()), vec!(),None ,vec!(), None, (15..15).collect::<Vec<i64>>(), vec![]))
            ),
        ];

        for (parameters, expected_outputs) in &test_tuples {
            let test_results = truncate_sequences(test_token_ids.clone(), Some(test_pair_token_ids.clone()), vec!(), None, vec!() ,None,parameters.0, parameters.1, parameters.2);
            match test_results {
                Ok(value) => assert_eq!(value, *expected_outputs.as_ref().unwrap()),
                Err(e) => assert_eq!(e.to_string(), (**expected_outputs.as_ref().err().unwrap()).to_string())
            }
        }
    }

    #[test]
    fn test_truncate_sentence_pair_second_only() {
//        Given
        let test_token_ids: Vec<i64> = (0..15).collect();
        let test_pair_token_ids: Vec<i64> = (42..51).collect();
        let test_tuples: [((usize, &TruncationStrategy, usize), std::result::Result<(Vec<i64>, std::option::Option<Vec<i64>>, Vec<Offset>, Option<Vec<Offset>>, Vec<Mask>, Option<Vec<Mask>>, Vec<i64>, Vec<Offset>), Box<dyn Error>>);
            5] = [
//            Baseline
            (
                (5, &TruncationStrategy::OnlySecond, 0),
                Ok(((0..15).collect::<Vec<i64>>(), Some((42..46).collect::<Vec<i64>>()), vec!() ,None, vec!() ,None,(46..51).collect::<Vec<i64>>(), vec![]))
            ),
//            With stride = 2
            (
                (5, &TruncationStrategy::OnlySecond, 2),
                Ok(((0..15).collect::<Vec<i64>>(), Some((42..46).collect::<Vec<i64>>()), vec!() ,None,vec!() ,None, (44..51).collect::<Vec<i64>>(), vec![]))
            ),
//            Truncate entire sentence 2
            (
                (9, &TruncationStrategy::OnlySecond, 2),
                Ok(((0..15).collect::<Vec<i64>>(), Some((42..42).collect::<Vec<i64>>()), vec!() ,None, vec!() ,None,(42..51).collect::<Vec<i64>>(), vec![]))
            ),
//            Request truncation amount greater than sentence 1
            (
                (10, &TruncationStrategy::OnlySecond, 2),
                Err("Second sequence too short for second only truncation".into())
            ),
//            No truncation
            (
                (0, &TruncationStrategy::OnlySecond, 2),
                Ok(((0..15).collect::<Vec<i64>>(), Some((42..51).collect::<Vec<i64>>()), vec!(), None, vec!() ,None,(42..42).collect::<Vec<i64>>(), vec![]))
            ),
        ];

        for (parameters, expected_outputs) in &test_tuples {
            let test_results = truncate_sequences(test_token_ids.clone(), Some(test_pair_token_ids.clone()), vec!(), None, vec!(), None,  parameters.0, parameters.1, parameters.2);
            match test_results {
                Ok(value) => assert_eq!(value, *expected_outputs.as_ref().unwrap()),
                Err(e) => assert_eq!(e.to_string(), (**expected_outputs.as_ref().err().unwrap()).to_string())
            }
        }
    }

    #[test]
    fn test_get_pair() {
//        Given
        let h = String::from("h");
        let e = String::from("e");
        let l = String::from("l");
        let o = String::from("o");
        let space = String::from(" ");

        let test_tuples = [
            (
                vec!(h.clone(), e.clone(), l.clone(), l.clone(), o.clone()),
                Some(HashSet::from_iter([
                    BpePairRef { byte_1: &h, byte_2: &e },
                    BpePairRef { byte_1: &e, byte_2: &l },
                    BpePairRef { byte_1: &l, byte_2: &l },
                    BpePairRef { byte_1: &l, byte_2: &o },
                ].iter().cloned()))
            ),
            (
                vec!(h.clone(), e.clone(), l.clone(), l.clone(), l.clone(), l.clone(), o.clone()),
                Some(HashSet::from_iter([
                    BpePairRef { byte_1: &h, byte_2: &e },
                    BpePairRef { byte_1: &e, byte_2: &l },
                    BpePairRef { byte_1: &l, byte_2: &l },
                    BpePairRef { byte_1: &l, byte_2: &o },
                ].iter().cloned()))
            ),
            (
                vec!(h.clone(), e.clone()),
                Some(HashSet::from_iter([
                    BpePairRef { byte_1: &h, byte_2: &e },
                ].iter().cloned()))
            ),
            (
                vec!(h.clone(), space.clone(), e.clone()),
                Some(HashSet::from_iter([
                    BpePairRef { byte_1: &h, byte_2: &space },
                    BpePairRef { byte_1: &space, byte_2: &e },
                ].iter().cloned()))
            ),
            (
                vec!(h.clone()),
                None
            ),
            (
                vec!(),
                None
            )
        ];

//        When & Then
        for (input, expected_output) in &test_tuples {
            assert_eq!(get_pairs(&input), *expected_output);
        }
    }

    fn generate_bpe_pair_vocab() -> BpePairVocab {
        let values: HashMap<(String, String), i64> = [
            (("e".to_owned(), "l".to_owned()), 0),
            (("e".to_owned(), "o".to_owned()), 1),
            (("h".to_owned(), "e".to_owned()), 2),
            (("el".to_owned(), "l".to_owned()), 3),
            (("o".to_owned(), "l".to_owned()), 4),
            (("l".to_owned(), "l".to_owned()), 5),
        ].iter().cloned().collect();

        BpePairVocab { values }
    }

    #[test]
    fn test_group_common_pairs() {
//        Given
        let bpe_pairs = generate_bpe_pair_vocab();

        let test_tuples = [
            (
                vec!(Token { text: "h".to_owned(), offset: Offset::new(0,1), mask: Mask::None }, //IN
                      Token { text: "e".to_owned(), offset: Offset::new(1,2), mask: Mask::None },
                      Token { text: "l".to_owned(), offset: Offset::new(2,3), mask: Mask::None },
                      Token { text: "l".to_owned(), offset: Offset::new(3,4), mask: Mask::None },
                      Token { text: "o".to_owned(), offset: Offset::new(4,5), mask: Mask::None }),
                (vec!(Token { text: "h".to_owned(), offset: Offset::new(0,1), mask: Mask::None }, //OUT
                      Token { text: "el".to_owned(), offset: Offset::new(1,3), mask: Mask::None },
                      Token { text: "l".to_owned(), offset: Offset::new(3,4), mask: Mask::None },
                      Token { text: "o".to_owned(), offset: Offset::new(4,5), mask: Mask::None })
                 , false)
            ),
            (
                vec!(Token { text: "h".to_owned(), offset: Offset::new(0,1), mask: Mask::None }, //IN
                      Token { text: "el".to_owned(), offset: Offset::new(1,3), mask: Mask::None },
                      Token { text: "l".to_owned(), offset: Offset::new(3,4), mask: Mask::None },
                      Token { text: "o".to_owned(), offset: Offset::new(4,5), mask: Mask::None }),

                (vec!(Token { text: "h".to_owned(), offset: Offset::new(0,1), mask: Mask::None }, //OUT
                      Token { text: "ell".to_owned(), offset: Offset::new(1,4), mask: Mask::None },
                      Token { text: "o".to_owned(), offset: Offset::new(4,5), mask: Mask::None })
                 , false)
            ),
            (
                vec!(Token { text: "h".to_owned(), offset: Offset::new(0,1), mask: Mask::None }, //IN
                      Token { text: "ell".to_owned(), offset: Offset::new(1,4), mask: Mask::None },
                      Token { text: "o".to_owned(), offset: Offset::new(4,5), mask: Mask::None }),

                (vec!(Token { text: "h".to_owned(), offset: Offset::new(0,1), mask: Mask::None }, //OUT
                      Token { text: "ell".to_owned(), offset: Offset::new(1,4), mask: Mask::None },
                      Token { text: "o".to_owned(), offset: Offset::new(4,5), mask: Mask::None })
                 , true)
            ),
            (
                vec!(Token { text: "h".to_owned(), offset: Offset::new(42,1), mask: Mask::None }), //IN

                (vec!(Token { text: "h".to_owned(), offset: Offset::new(42,1), mask: Mask::None }), //OUT
                          true)
            ),
            (
                vec!(Token { text: "h".to_owned(), offset: Offset::new(0,1), mask: Mask::None }, //IN
                      Token { text: "ello".to_owned(), offset: Offset::new(1,5), mask: Mask::None }),

                (vec!(Token { text: "h".to_owned(), offset: Offset::new(0,1), mask: Mask::None }, //OUT
                      Token { text: "ello".to_owned(), offset: Offset::new(1,5), mask: Mask::None })
                 , true)
            )
        ];

//        When & Then
        for (input, expected_output) in &test_tuples {
            assert_eq!(group_common_pairs(input.clone(), &bpe_pairs), *expected_output);
        }
    }

    #[test]
    fn test_bpe_exact() {
//        Given
        let bpe_pairs = generate_bpe_pair_vocab();

        let test_tuples = [
            (
                "hello",
                vec!(Token { text: "h@@".to_owned(), offset: Offset::new(0,1), mask: Mask::Begin }, //OUT
                      Token { text: "ell@@".to_owned(), offset: Offset::new(1,4), mask: Mask::Continuation },
                      Token { text: "o".to_owned(), offset: Offset::new(4,5), mask: Mask::Continuation })
            ),
            (
                "hellllo",
                vec!(Token { text: "h@@".to_owned(), offset: Offset::new(0,1), mask: Mask::Begin }, //OUT
                      Token { text: "ell@@".to_owned(), offset: Offset::new(1,4), mask: Mask::Continuation },
                      Token { text: "ll@@".to_owned(), offset: Offset::new(4,6), mask: Mask::Continuation },
                      Token { text: "o".to_owned(), offset: Offset::new(6,7), mask: Mask::Continuation })
            ),
            (
                "helo",
                vec!(Token { text: "h@@".to_owned(), offset: Offset::new(0,1), mask: Mask::Begin }, //OUT
                      Token { text: "el@@".to_owned(), offset: Offset::new(1,3), mask: Mask::Continuation },
                      Token { text: "o".to_owned(), offset: Offset::new(3,4), mask: Mask::Continuation })
            ),
            (
                "42",
                vec!(Token { text: "4@@".to_owned(), offset: Offset::new(0,1), mask: Mask::Begin }, //OUT
                      Token { text: "2".to_owned(), offset: Offset::new(1,2), mask: Mask::Continuation })
            ),
            (
                "1",
                vec!(Token { text: "1".to_owned(), offset: Offset::new(0,1), mask: Mask::Begin }), //OUT (mask is corrected to None in a later stage)
            ),
            (
                "",
                vec!(), //OUT (differs from original test that outputted a single EMPTY token!)
            ),
        ];

//        When & Then
        for (input, expected_output) in &test_tuples {
            let input: TokenRef = TokenRef { text: input, offset: Offset::new(0, input.chars().count() as OffsetSize), mask: Mask::None };
            assert_eq!(ctrl_bpe(input, &bpe_pairs, true), *expected_output);
        }
    }

    #[test]
    fn test_bpe_inexact() {
//        Given
        let bpe_pairs = generate_bpe_pair_vocab();

        let test_tuples = [
            (
                "hello",
                vec!(Token { text: "h@@".to_owned(), offset: Offset::new(0,5), mask: Mask::InexactBegin }, //OUT
                      Token { text: "ell@@".to_owned(), offset: Offset::new(0,5), mask: Mask::InexactContinuation },
                      Token { text: "o".to_owned(), offset: Offset::new(0,5), mask: Mask::InexactContinuation })
            ),
            (
                "hellllo",
                vec!(Token { text: "h@@".to_owned(), offset: Offset::new(0,7), mask: Mask::InexactBegin }, //OUT
                      Token { text: "ell@@".to_owned(), offset: Offset::new(0,7), mask: Mask::InexactContinuation },
                      Token { text: "ll@@".to_owned(), offset: Offset::new(0,7), mask: Mask::InexactContinuation },
                      Token { text: "o".to_owned(), offset: Offset::new(0,7), mask: Mask::InexactContinuation })
            ),
            (
                "helo",
                vec!(Token { text: "h@@".to_owned(), offset: Offset::new(0,4), mask: Mask::InexactBegin }, //OUT
                      Token { text: "el@@".to_owned(), offset: Offset::new(0,4), mask: Mask::InexactContinuation },
                      Token { text: "o".to_owned(), offset: Offset::new(0,4), mask: Mask::InexactContinuation })
            ),
            (
                "42",
                vec!(Token { text: "4@@".to_owned(), offset: Offset::new(0,2), mask: Mask::InexactBegin }, //OUT
                      Token { text: "2".to_owned(), offset: Offset::new(0,2), mask: Mask::InexactContinuation })
            ),
            (
                "1",
                vec!(Token { text: "1".to_owned(), offset: Offset::new(0,1), mask: Mask::InexactBegin }), //OUT (mask is correct to None in a later stage)
            ),
            (
                "",
                vec!(), //OUT (differs from original test that outputted a single EMPTY token!)
            ),
        ];

//        When & Then
        for (input, expected_output) in &test_tuples {
            let input: TokenRef = TokenRef { text: input, offset: Offset::new(0, input.chars().count() as OffsetSize), mask: Mask::None };
            assert_eq!(ctrl_bpe(input, &bpe_pairs, false), *expected_output);
        }
    }
}
