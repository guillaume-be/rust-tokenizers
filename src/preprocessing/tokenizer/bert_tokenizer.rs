use crate::preprocessing::vocab::base_vocab::Vocab;
use crate::preprocessing::tokenizer::base_tokenizer::{Tokenizer, BaseTokenizer};
use std::sync::Arc;
use rayon::prelude::*;
use crate::preprocessing::tokenizer::tokenization_utils::tokenize_wordpiece;

pub struct BertTokenizer<T: Vocab> {
    vocab: Arc<T>,
    base_tokenizer: BaseTokenizer<T>,
}

impl<T: Vocab + Sync + Send> BertTokenizer<T> {
    pub fn from_file(path: &str) -> BertTokenizer<T> {
        let vocab = Arc::new(T::from_file(path));
        let base_tokenizer = BaseTokenizer::from_existing_vocab(vocab.clone());
        BertTokenizer { vocab, base_tokenizer }
    }

    pub fn from_existing_vocab(vocab: Arc<T>) -> BertTokenizer<T> {
        let base_tokenizer = BaseTokenizer::from_existing_vocab(vocab.clone());
        BertTokenizer { vocab: vocab.clone(), base_tokenizer }
    }

    pub fn convert_tokens_to_ids(&self, tokens: &Vec<String>) -> Vec<i64> {
        tokens.iter().map(|v| self.vocab.token_to_id(v)).collect()
    }

    pub fn encode(&self, text: &str) -> Vec<i64> {
        self.convert_tokens_to_ids(&self.tokenize(text))
    }

    pub fn encode_list(&self, text_list: Vec<&str>) -> Vec<Vec<i64>> {
        text_list
            .par_iter()
            .map(|text| self.tokenize(text))
            .map(|tokens| self.convert_tokens_to_ids(&tokens))
            .collect()
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
        text_list
            .par_iter()
            .map(|text| self.tokenize(text))
            .collect()
    }
}


//==============================
// Unit tests
//==============================
#[cfg(test)]
mod tests {
    use super::*;
    use crate::BertVocab;
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

        BertVocab { values, unknown_value: "[UNK]", special_values }
    }

    #[test]
    fn test_bert_tokenizer() {
//        Given
        let vocab = Arc::new(generate_test_vocab());
        let bert_tokenizer: BertTokenizer<BertVocab> = BertTokenizer::from_existing_vocab(vocab);
        let test_tuples = [
            (
                "Hello [MASK] world!",
                vec!("hello", "[MASK]", "world", "!")
            ),
            (
                "Hello, unaffable world!",
                vec!("hello", "[UNK]", "una", "##ffa", "##ble", "world", "!")
            ),
            (
                "[UNK]中华人民共和国 [PAD] asdf",
                vec!("[UNK]", "[UNK]", "华", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[PAD]", "[UNK]")
            )
        ];
        let source_texts: Vec<&str> = test_tuples.iter().map(|v| v.0).collect();
        let expected_results: Vec<Vec<&str>> = test_tuples.iter().map(|v| v.1.clone()).collect();

//        When & Then
        for (source_text, expected_result) in test_tuples.iter() {
            assert_eq!(bert_tokenizer.tokenize(*source_text), *expected_result);
        }

        assert_eq!(bert_tokenizer.tokenize_list(source_texts), expected_results);
    }
}