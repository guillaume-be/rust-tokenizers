use crate::preprocessing::vocab::base_vocab::Vocab;
use std::rc::Rc;
use crate::preprocessing::tokenizer::tokenization_utils::{split_on_special_tokens, tokenize_cjk_chars, whitespace_tokenize, strip_accents, split_on_punct};

pub trait Tokenizer {
    fn tokenize(&self, text: &str) -> Vec<String>;
}

pub struct BaseTokenizer<T: Vocab> {
    vocab: Rc<T>
}

impl<T: Vocab> BaseTokenizer<T> {
    pub fn from_file(path: &str) -> BaseTokenizer<T> {
        let vocab = T::from_file(path);
        BaseTokenizer { vocab: Rc::new(vocab) }
    }

    pub fn from_existing_vocab(vocab: Rc<T>) -> BaseTokenizer<T> {
        BaseTokenizer { vocab: vocab.clone() }
    }
}

impl<T: Vocab> Tokenizer for BaseTokenizer<T> {
    fn tokenize(&self, text: &str) -> Vec<String> {
        let tokenized_text: Vec<String> = {
            let temp_text = split_on_special_tokens(text, self.vocab.as_ref());
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
            if !self.vocab.as_ref().special_values().contains_key(string) {
                *string = string.to_lowercase();
                *string = strip_accents(string.to_owned());
            }
        }

        let tokenized_text: Vec<String> = tokenized_text
            .iter()
            .map(|v| split_on_punct(v.to_owned(), self.vocab.as_ref()))
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
}