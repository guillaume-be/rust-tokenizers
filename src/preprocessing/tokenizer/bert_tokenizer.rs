use crate::preprocessing::vocab::base_vocab::Vocab;
use crate::preprocessing::tokenizer::base_tokenizer::{split_on_special_tokens,
                                                      tokenize_cjk_chars,
                                                      whitespace_tokenize};

pub fn tokenize_bert(text: &str, vocab: &impl Vocab) -> Vec<String> {
    let tokenized_text: Vec<String> = {
        let temp_text = split_on_special_tokens(text, vocab);
        let temp_text: Vec<String> = temp_text.
            iter().
            map(|v| tokenize_cjk_chars(v)).
            collect();
        temp_text
    };

    let mut tokenized_text: Vec<String> = tokenized_text
        .iter()
        .map(|v| whitespace_tokenize(&v))
        .flatten()
        .map(|s| s.to_string())
        .collect();


    for string in tokenized_text.iter_mut() {
        if !vocab.special_values().contains_key(string) {
            *string = string.to_lowercase();
        }
    }

    tokenized_text
}