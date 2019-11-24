use crate::preprocessing::vocab::base_vocab::Vocab;
use crate::preprocessing::tokenizer::base_tokenizer::{split_on_special_tokens, tokenize_cjk_chars, whitespace_tokenize, strip_accents, split_on_punct};
use crate::BertVocab;

pub fn tokenize_bert(text: &str, vocab: &impl Vocab) -> Vec<String> {
    let tokenized_text: Vec<String> = {
        let temp_text = split_on_special_tokens(text, vocab);
        let temp_text: Vec<String> = temp_text.
            iter().
            map(|v| tokenize_cjk_chars(v)).
            collect();
        temp_text
    };

    let tokenized_text = tokenize_basic(tokenized_text, vocab);

    let tokenized_text: Vec<String> = tokenized_text
        .iter()
        .map(|v| tokenize_wordpiece(v.to_owned(), vocab, 100))
        .flatten()
        .map(|s| s.to_string())
        .collect();
    tokenized_text
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
                if match vocab.values().get(&substr) {
                    Some(_) => true,
                    None => false
                } {
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

fn tokenize_basic(tokens: Vec<String>, vocab: &impl Vocab) -> Vec<String> {
    let mut tokenized_text: Vec<String> = tokens
        .iter()
        .map(|v| whitespace_tokenize(&v))
        .flatten()
        .map(|s| s.to_string())
        .collect();

    for string in tokenized_text.iter_mut() {
        if !vocab.special_values().contains_key(string) {
            *string = string.to_lowercase();
            *string = strip_accents(string.to_owned());
        }
    }

    let tokenized_text: Vec<String> = tokenized_text
        .iter()
        .map(|v| split_on_punct(v.to_owned(), vocab))
        .flatten()
        .map(|s| s.to_string())
        .collect();

    let tokenized_text: Vec<String> = tokenized_text
        .iter()
        .map(|v| whitespace_tokenize(&v))
        .flatten()
        .map(|s| s.to_string())
        .collect();

    tokenized_text
}