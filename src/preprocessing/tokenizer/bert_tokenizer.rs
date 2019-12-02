use crate::preprocessing::vocab::base_vocab::Vocab;
use crate::preprocessing::tokenizer::base_tokenizer::{Tokenizer, BaseTokenizer};
use std::sync::Arc;
use rayon::prelude::*;
use crate::preprocessing::tokenizer::tokenization_utils::tokenize_wordpiece;

pub struct BertTokenizer<T: Vocab> {
    vocab: Arc<T>,
    base_tokenizer: BaseTokenizer<T>,
}

impl<T: Vocab> BertTokenizer<T> {
    pub fn from_file(path: &str) -> BertTokenizer<T> {
        let vocab = Arc::new(T::from_file(path));
        let base_tokenizer = BaseTokenizer::from_existing_vocab(vocab.clone());
        BertTokenizer { vocab, base_tokenizer }
    }

    pub fn from_existing_vocab(vocab: Arc<T>) -> BertTokenizer<T> {
        let base_tokenizer = BaseTokenizer::from_existing_vocab(vocab.clone());
        BertTokenizer { vocab: vocab.clone(), base_tokenizer }
    }
}

impl<T: Vocab + Sync + Send> Tokenizer for BertTokenizer<T> {
    fn tokenize(&self, text: &str) -> Vec<String> {
        let tokenized_text: Vec<String> = self.base_tokenizer.tokenize(text);
        let tokenized_text: Vec<String> = tokenized_text
            .iter()
            .map(|v| tokenize_wordpiece(v.to_owned(), self.vocab.as_ref(), 100))
            .flatten()
            .map(|s| s.to_string())
            .collect();
        tokenized_text
    }

    fn tokenize_list(&self, text_list: Vec<&str>) -> Vec<Vec<String>> {
        text_list.
            par_iter().
            map(|text| self.tokenize(text)).
            collect()
    }
}