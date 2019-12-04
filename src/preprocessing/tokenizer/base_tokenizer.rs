use crate::preprocessing::vocab::base_vocab::Vocab;
use crate::preprocessing::tokenizer::tokenization_utils::{split_on_special_tokens, tokenize_cjk_chars, whitespace_tokenize, strip_accents, split_on_punct, clean_text};
use std::sync::Arc;
use rayon::prelude::*;
use itertools::Itertools;
use std::error::Error;

pub enum TruncationStrategy {
    LongestFirst,
    OnlyFirst,
    OnlySecond,
    DoNotTruncate,
}

pub trait Tokenizer<T: Vocab>
    where Self: std::marker::Sync {
    fn vocab(&self) -> &T;

    fn tokenize(&self, text: &str) -> Vec<String>;

    fn tokenize_list(&self, text_list: Vec<&str>) -> Vec<Vec<String>> {
        text_list.
            par_iter().
            map(|text| self.tokenize(text)).
            collect()
    }

    fn convert_tokens_to_ids(&self, tokens: &Vec<String>) -> Vec<i64> {
        tokens.iter().map(|v| self.vocab().token_to_id(v)).collect()
    }

    fn encode(&self, text_1: &str, text_2: Option<&str>, max_len: usize) -> Vec<i64> {
        let token_ids_1 = self.convert_tokens_to_ids(&self.tokenize(text_1));
        let len_1 = token_ids_1.len();
        let (token_ids_2, len_2, pair) = {
            if let Some(text) = text_2 {
                let token_ids_2: Vec<i64> = self.convert_tokens_to_ids(&self.tokenize(text));
                let len_2 = token_ids_2.len();
                (Some(token_ids_2), len_2, Some(vec!()))
            } else {
                (None, 0, None)
            }
        };
        let total_len = len_1 + len_2 + self.build_input_with_special_tokens(vec!(), pair).len();
        let num_tokens_to_remove = if total_len > max_len { total_len - max_len } else { 0 };
        let (token_ids_1,
            token_ids_2,
            _overflow_tokens) = self.truncate_sequences(token_ids_1,
                                                        token_ids_2,
                                                        num_tokens_to_remove,
                                                        TruncationStrategy::LongestFirst,
                                                        0).unwrap();

        self.build_input_with_special_tokens(token_ids_1,
                                             token_ids_2)
    }

    fn encode_list(&self, text_list: Vec<&str>, max_len: usize) -> Vec<Vec<i64>> {
        text_list
            .par_iter()
            .map(|text| self.encode(text, None, max_len))
            .collect()
    }

    fn encode_pair_list(&self, text_list: Vec<(&str, &str)>, max_len: usize) -> Vec<Vec<i64>> {
        text_list
            .par_iter()
            .map(|text| self.encode(text.0, Some(text.1), max_len))
            .collect()
    }


    fn build_input_with_special_tokens(&self, tokens_1: Vec<i64>, tokens_2: Option<Vec<i64>>) -> Vec<i64>;

    fn truncate_sequences(&self, mut tokens_1: Vec<i64>, tokens_2: Option<Vec<i64>>,
                          num_tokens_to_remove: usize, truncation_strategy: TruncationStrategy, stride: usize)
                          -> Result<(Vec<i64>, Option<Vec<i64>>, Vec<i64>), Box<dyn Error>> {
        if num_tokens_to_remove == 0 {
            Ok((tokens_1, tokens_2, Vec::new()))
        } else {
            match tokens_2 {
                Some(mut tokens_2) => {
                    match truncation_strategy {
                        TruncationStrategy::LongestFirst => {
                            let mut overflow_tokens: Vec<i64> = Vec::with_capacity(num_tokens_to_remove);
                            for _ in 0..num_tokens_to_remove {
                                if tokens_1.len() >= tokens_2.len() {
                                    overflow_tokens.push(tokens_1.pop().unwrap());
                                } else {
                                    overflow_tokens.push(tokens_2.pop().unwrap());
                                }
                            }
                            Ok((tokens_1, Some(tokens_2), overflow_tokens))
                        }
                        TruncationStrategy::OnlyFirst => {
                            if tokens_1.len() > num_tokens_to_remove {
                                let cutoff = tokens_1.len() - num_tokens_to_remove;
                                let overflow = tokens_1.split_off(cutoff);
                                Ok((tokens_1, Some(tokens_2), overflow.to_vec()))
                            } else {
                                Err("First sequence too short for first only truncation".into())
                            }
                        }
                        TruncationStrategy::OnlySecond => {
                            if tokens_2.len() > num_tokens_to_remove {
                                let cutoff = tokens_2.len() - num_tokens_to_remove;
                                let overflow = tokens_2.split_off(cutoff);
                                Ok((tokens_1, Some(tokens_2), overflow.to_vec()))
                            } else {
                                Err("First sequence too short for first only truncation".into())
                            }
                        }
                        TruncationStrategy::DoNotTruncate => Err("Truncation needed but no truncation requested".into())
                    }
                }
                None => {
                    match truncation_strategy {
                        TruncationStrategy::LongestFirst | TruncationStrategy::OnlyFirst => {
                            let cutoff = tokens_1.len() - num_tokens_to_remove;
                            let overflow = tokens_1.split_off(cutoff);
                            Ok((tokens_1, None, overflow.to_vec()))
                        }
                        _ => Err("Invalid truncation strategy for single sentence truncation".into())
                    }
                }
            }
        }
    }
}


pub struct BaseTokenizer<T: Vocab> {
    vocab: Arc<T>
}

impl<T: Vocab + Sync + Send> BaseTokenizer<T> {
    pub fn from_file(path: &str) -> BaseTokenizer<T> {
        let vocab = T::from_file(path);
        BaseTokenizer { vocab: Arc::new(vocab) }
    }

    pub fn from_existing_vocab(vocab: Arc<T>) -> BaseTokenizer<T> {
        BaseTokenizer { vocab: vocab.clone() }
    }
}

impl<T: Vocab + Sync + Send> Tokenizer<T> for BaseTokenizer<T> {
    fn vocab(&self) -> &T {
        &self.vocab
    }

    fn tokenize(&self, text: &str) -> Vec<String> {
        let tokenized_text: Vec<String> = {
            let temp_text = split_on_special_tokens(text, self.vocab.as_ref());
            let temp_text: Vec<String> = temp_text.
                iter().
                map(|v| clean_text(v, true)).
                map(|v| tokenize_cjk_chars(&v)).
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

        let tokenized_text: String = tokenized_text.iter().join(" ");
        let tokenized_text: Vec<String> = whitespace_tokenize(&tokenized_text)
            .iter()
            .map(|s| s.to_string())
            .collect();

        tokenized_text
    }

    fn build_input_with_special_tokens(&self, tokens_1: Vec<i64>, tokens_2: Option<Vec<i64>>) -> Vec<i64> {
        panic!("Not implemented, cannot combine {:?} and {:?}", tokens_1, tokens_2);
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

        BertVocab { values, unknown_value: "[UNK]", special_values }
    }

    #[test]
    fn test_base_tokenizer() {
//        Given
        let vocab = Arc::new(generate_test_vocab());
        let base_tokenizer: BaseTokenizer<BertVocab> = BaseTokenizer::from_existing_vocab(vocab);
        let test_tuples = [
            (
                "Sentence with [MASK] token.",
                vec!("sentence", "with", "[MASK]", "token", ".")
            ),
            (
                "[CLS]Sentence with [MASK] token.",
                vec!("[CLS]", "sentence", "with", "[MASK]", "token", ".")
            ),
            (
                "[CLS]",
                vec!("[CLS]")
            ),
            (
                "[CLS] [PAD]",
                vec!("[CLS]", "[PAD]")
            ),
            (
                "[CLS]       [PAD]",
                vec!("[CLS]", "[PAD]")
            ),
            (
                "asdf[CLS]",
                vec!("asdf", "[CLS]")
            ),
            (
                "",
                vec!()
            ),
            (
                "Allons, Flipote, allons; que d'eux je me délivre.",
                vec!("allons", ",", "flipote", ",", "allons", ";", "que", "d", "\'", "eux", "je", "me", "delivre", ".")
            ),
            (
                "[UNK]中华人民共和国 [PAD] asdf",
                vec!("[UNK]", "中", "华", "人", "民", "共", "和", "国", "[PAD]", "asdf")
            )
        ];
        let source_texts: Vec<&str> = test_tuples.iter().map(|v| v.0).collect();
        let expected_results: Vec<Vec<&str>> = test_tuples.iter().map(|v| v.1.clone()).collect();

//        When & Then
        for (source_text, expected_result) in test_tuples.iter() {
            assert_eq!(base_tokenizer.tokenize(*source_text), *expected_result);
        }

        assert_eq!(base_tokenizer.tokenize_list(source_texts), expected_results);
    }

    #[test]
    fn test_convert_tokens_to_ids() {
//        Given
        let vocab = Arc::new(generate_test_vocab());
        let base_tokenizer: BaseTokenizer<BertVocab> = BaseTokenizer::from_existing_vocab(vocab);
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
    fn test_encode() {
//        Given
        let vocab = Arc::new(generate_test_vocab());
        let base_tokenizer: BaseTokenizer<BertVocab> = BaseTokenizer::from_existing_vocab(vocab);
        let test_tuples = [
            (
                "hello[MASK] world!",
                vec!(0, 6, 1, 3)
            ),
            (
                "hello, unaffable world!",
                vec!(0, 2, 2, 1, 3)
            ),
            (
                "[UNK]中华人民共和国 [PAD] asdf",
                vec!(2, 2, 8, 2, 2, 2, 2, 2, 10, 2)
            )
        ];
        let source_texts: Vec<&str> = test_tuples.iter().map(|v| v.0).collect();
        let expected_results: Vec<Vec<i64>> = test_tuples.iter().map(|v| v.1.clone()).collect();

//        When & Then
        for (source_text, expected_result) in test_tuples.iter() {
            assert_eq!(base_tokenizer.encode(source_text, None, 128),
                       *expected_result);
        }
        assert_eq!(base_tokenizer.encode_list(source_texts, 128), expected_results);
    }
}
