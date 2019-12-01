use crate::preprocessing::vocab::base_vocab::Vocab;
use crate::BertVocab;
use unicode_normalization::is_nfd;
use unicode_normalization::char::{decompose_canonical, is_combining_mark};


pub fn split_on_special_tokens<'a>(text: &'a str, vocab: &'a impl Vocab) -> Vec<&'a str> {
    let mut text_list: Vec<&str> = vec!(text);
    let mut temp_list: Vec<&str>;

    for special_value in vocab.special_values() {
        temp_list = vec!();
        for subtext in &text_list {
            let new_items = split_with_separator(subtext, special_value.0);
            temp_list.extend(new_items);
        }
        text_list = temp_list;
    }
    text_list
}

fn split_with_separator<'a>(text: &'a str, separator: &'a str) -> Vec<&'a str> {
    let split_text: Vec<&str> = text.split(separator).collect();
    let mut result: Vec<&str> = vec!();
    if text.is_empty() {
        result.push(text);
        return result;
    }
    for (i, subtext) in split_text.iter().enumerate() {
        let trimmed_subtext = subtext.trim();
        if (i == 0) & trimmed_subtext.is_empty() {
            result.push(separator);
        } else if i == split_text.len() - 1 {
            if !trimmed_subtext.is_empty() {
                result.push(trimmed_subtext);
            }
        } else {
            if !trimmed_subtext.is_empty() {
                result.push(trimmed_subtext);
            }
            result.push(separator);
        }
    }
    result
}

pub fn tokenize_cjk_chars(text: &str) -> String {
    let mut output = String::new();
    for character in text.chars() {
        if is_cjk_char(&character) {
            output.push(' ');
            output.push(character);
            output.push(' ');
        } else {
            output.push(character);
        }
    }
    output
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

//ToDo: Add the control chars to the list of whitespaces
pub fn whitespace_tokenize(text: &str) -> Vec<&str> {
    text.split_whitespace().collect()
}

pub fn strip_accents(text: String) -> String {
    if !is_nfd(&text) {
        let mut decomposed_string: String = String::with_capacity(text.capacity());
        for character in text.chars() {
            decompose_canonical(character, |c| if !is_combining_mark(c) { decomposed_string.push(c) });
        }
        decomposed_string
    } else {
        text
    }
}

pub fn split_on_punct(text: String, vocab: &impl Vocab) -> Vec<String> {
    let mut output: Vec<String> = Vec::new();
    let mut start_new_word: bool = true;
    let mut temp_string = String::new();
    if vocab.special_values().contains_key(&text) {
        output.push(text);
        output
    } else {
        for character in text.chars() {
            if character.is_ascii_punctuation() {
                if !&temp_string.is_empty() {
                    output.push(temp_string.clone());
                    temp_string = String::new();
                }
                output.push(character.to_string());
                start_new_word = true
            } else {
                if start_new_word {
                    temp_string = String::new();
                }
                start_new_word = false;
                temp_string.push(character);
            }
        }
        if !start_new_word & !&temp_string.is_empty() {
            output.push(temp_string.clone());
        }
        output
    }
}

pub fn tokenize_wordpiece(token: String, vocab: &impl Vocab, max_word_len: usize) -> Vec<String> {
    let mut tokenized_text: Vec<String> = Vec::new();
    if token.chars().count() > max_word_len {
        tokenized_text.push(BertVocab::unknown_value().to_owned());
    } else {
        let char_indices: Vec<usize> = token.char_indices().map(|v| v.0).collect();
        let max_end: usize = char_indices.last().unwrap() + token.chars().last().unwrap().len_utf8();
        let mut start: usize = 0;
        let mut pos_end;
        let mut end;
        while start < max_end {
            end = max_end;
            pos_end = char_indices.len() - 1;
            while start < end {
                let mut substr = token[start..end].to_owned();
                if start > 0 {
                    substr = format!("##{}", substr);
                }
                if vocab.values().contains_key(&substr) {
                    tokenized_text.push(substr);
                    break;
                }
                if pos_end == start {
                    let mut tokenized_text: Vec<String> = Vec::new();
                    tokenized_text.push(BertVocab::unknown_value().to_owned());
                    return tokenized_text;
                }
                pos_end = pos_end - 1;
                end = char_indices[pos_end + 1];
            }
            start = end;
        }
    }
    tokenized_text
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn generate_test_vocab() -> BertVocab {
        let values: HashMap<String, i64> = [
            ("hello".to_owned(), 0),
            ("world".to_owned(), 1),
            ("[UNK]".to_owned(), 2),
            ("!".to_owned(), 3),
            ("[CLS]".to_owned(), 4),
            ("[SEP]".to_owned(), 5),
            ("[MASK]".to_owned(), 6),
            ("[PAD]".to_owned(), 7)
        ].iter().cloned().collect();

        let special_values: HashMap<String, i64> = [
            ("[UNK]".to_owned(), 2),
            ("[CLS]".to_owned(), 4),
            ("[SEP]".to_owned(), 5),
            ("[MASK]".to_owned(), 6),
            ("[PAD]".to_owned(), 7)
        ].iter().cloned().collect();

        BertVocab { values, unknown_value: "[UNK]", special_values }
    }

    #[test]
    fn test_split_on_special_tokens() {
//        Given
        let vocab = generate_test_vocab();

//        When & Then
        assert_eq!(split_on_special_tokens("Sentence with [MASK] token.", &vocab),
                   vec!("Sentence with", "[MASK]", "token."));
        assert_eq!(split_on_special_tokens("[CLS]Sentence with [MASK] token.", &vocab),
                   vec!("[CLS]", "Sentence with", "[MASK]", "token."));
        assert_eq!(split_on_special_tokens("[CLS]", &vocab),
                   vec!("[CLS]"));
        assert_eq!(split_on_special_tokens("[CLS][PAD]", &vocab),
                   vec!("[CLS]", "[PAD]"));
        assert_eq!(split_on_special_tokens("[CLS] [PAD]", &vocab),
                   vec!("[CLS]", "[PAD]"));
        assert_eq!(split_on_special_tokens("[CLS]       [PAD]", &vocab),
                   vec!("[CLS]", "[PAD]"));
        assert_eq!(split_on_special_tokens("asdf[CLS]", &vocab),
                   vec!("asdf", "[CLS]"));
        assert_eq!(split_on_special_tokens("No special token in sentence", &vocab),
                   vec!("No special token in sentence"));
        assert_eq!(split_on_special_tokens("", &vocab),
                   vec!(""));
        assert_eq!(split_on_special_tokens("[UNK]中华人民共和国 [PAD] asdf", &vocab),
                   vec!("[UNK]", "中华人民共和国", "[PAD]", "asdf"));
    }
}