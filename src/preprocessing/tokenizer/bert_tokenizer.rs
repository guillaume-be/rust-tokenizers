use crate::preprocessing::tokenizer::base_tokenizer::{Tokenizer, BaseTokenizer};
use std::sync::Arc;
use crate::preprocessing::tokenizer::tokenization_utils::tokenize_wordpiece;
use crate::preprocessing::vocab::base_vocab::Vocab;
use crate::BertVocab;

pub struct BertTokenizer {
    vocab: Arc<BertVocab>,
    base_tokenizer: BaseTokenizer<BertVocab>,
}

impl BertTokenizer {
    pub fn from_file(path: &str) -> BertTokenizer {
        let vocab = Arc::new(BertVocab::from_file(path));
        let base_tokenizer = BaseTokenizer::from_existing_vocab(vocab.clone());
        BertTokenizer { vocab, base_tokenizer }
    }

    pub fn from_existing_vocab(vocab: Arc<BertVocab>) -> BertTokenizer {
        let base_tokenizer = BaseTokenizer::from_existing_vocab(vocab.clone());
        BertTokenizer { vocab: vocab.clone(), base_tokenizer }
    }
}

impl Tokenizer<BertVocab> for BertTokenizer {
    fn vocab(&self) -> &BertVocab {
        &self.vocab
    }
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

    fn build_input_with_special_tokens(&self, tokens_1: Vec<i64>, tokens_2: Option<Vec<i64>>) -> (Vec<i64>, Vec<i8>, Vec<i8>) {
        let mut output: Vec<i64> = vec!();
        let mut token_segment_ids: Vec<i8> = vec!();
        let mut special_tokens_mask: Vec<i8> = vec!();
        special_tokens_mask.push(1);
        special_tokens_mask.extend(vec![0; tokens_1.len()]);
        special_tokens_mask.push(1);
        token_segment_ids.extend(vec![0; tokens_1.len() + 2]);
        output.push(self.vocab.token_to_id(BertVocab::cls_value()));
        output.extend(tokens_1);
        output.push(self.vocab.token_to_id(BertVocab::sep_value()));
        if let Some(add_tokens) = tokens_2 {
            special_tokens_mask.extend(vec![0; add_tokens.len()]);
            special_tokens_mask.push(1);
            token_segment_ids.extend(vec![1; add_tokens.len() + 1]);
            output.extend(add_tokens);
            output.push(self.vocab.token_to_id(BertVocab::sep_value()));
        }
        (output, token_segment_ids, special_tokens_mask)
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
    use crate::preprocessing::tokenizer::base_tokenizer::{TruncationStrategy, TokenizedInput};

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

        BertVocab { values, unknown_value: "[UNK]", special_values }
    }

    #[test]
    fn test_bert_tokenizer() {
//        Given
        let vocab = Arc::new(generate_test_vocab());
        let bert_tokenizer: BertTokenizer = BertTokenizer::from_existing_vocab(vocab);
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
                vec!("[UNK]", "中", "华", "人", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[PAD]", "[UNK]")
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

    #[test]
    fn test_encode() {
//        Given
        let vocab = Arc::new(generate_test_vocab());
        let bert_tokenizer: BertTokenizer = BertTokenizer::from_existing_vocab(vocab);
        let truncation_strategy = TruncationStrategy::LongestFirst;
        let test_tuples = [
            (
                "hello[MASK] world!",
                TokenizedInput { token_ids: vec!(4, 0, 6, 1, 3, 5), segment_ids: vec!(0, 0, 0, 0, 0, 0), special_tokens_mask: vec!(1, 0, 0, 0, 0, 1), overflowing_tokens: vec!(), num_truncated_tokens: 0 }
             ),
            (
                "hello, unaffable world!",
                TokenizedInput { token_ids: vec!(4, 0, 2, 11, 12, 13, 1, 3, 5), segment_ids: vec!(0, 0, 0, 0, 0, 0, 0, 0, 0), special_tokens_mask: vec!(1, 0, 0, 0, 0, 0, 0, 0, 1), overflowing_tokens: vec!(), num_truncated_tokens: 0 }
            ),
            (
                "[UNK]中华人民共和国 [PAD] asdf",
                TokenizedInput { token_ids: vec!(4, 2, 7, 8, 9, 2, 2, 2, 2, 10, 2, 5), segment_ids: vec!(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), special_tokens_mask: vec!(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1), overflowing_tokens: vec!(), num_truncated_tokens: 0 }
            )
        ];
        let source_texts: Vec<&str> = test_tuples.iter().map(|v| v.0).collect();
        let expected_results: Vec<TokenizedInput> = test_tuples.iter().map(|v| v.1.clone()).collect();

//        When & Then
        for (source_text, expected_result) in test_tuples.iter() {
            assert_eq!(bert_tokenizer.encode(source_text, None, 128, &truncation_strategy, 0),
                       *expected_result);
        }
        assert_eq!(bert_tokenizer.encode_list(source_texts, 128, &truncation_strategy, 0), expected_results);
    }

    #[test]
    fn test_encode_sentence_pair() {
//        Given
        let vocab = Arc::new(generate_test_vocab());
        let bert_tokenizer: BertTokenizer = BertTokenizer::from_existing_vocab(vocab);
        let truncation_strategy = TruncationStrategy::LongestFirst;
        let test_tuples = [
//            No truncation required
            (
                ("hello world", "This is the second sentence"),
                TokenizedInput { token_ids: vec!(4, 0, 1, 5, 2, 2, 2, 2, 2, 5), segment_ids: vec!(0, 0, 0, 0, 1, 1, 1, 1, 1, 1), special_tokens_mask: vec!(1, 0, 0, 1, 0, 0, 0, 0, 0, 1), overflowing_tokens: vec!(), num_truncated_tokens: 0 }
            ),
//            Truncation of sentence 2 (longest)
            (
                ("hello world", "!This is the second sentence!!!"),
                TokenizedInput { token_ids: vec!(4, 0, 1, 5, 3, 2, 2, 2, 2, 5), segment_ids: vec!(0, 0, 0, 0, 1, 1, 1, 1, 1, 1), special_tokens_mask: vec!(1, 0, 0, 1, 0, 0, 0, 0, 0, 1), overflowing_tokens: vec!(), num_truncated_tokens: 4 }
            ),
//            Truncation of sentence 1 (longest)
            (
                ("[UNK] hello  hello  hello  hello  hello  hello  hello  hello  hello  hello  hello", "!!!"),
                TokenizedInput { token_ids: vec!(4, 2, 0, 0, 0, 5, 3, 3, 3, 5), segment_ids: vec!(0, 0, 0, 0, 0, 0, 1, 1, 1, 1), special_tokens_mask: vec!(1, 0, 0, 0, 0, 1, 0, 0, 0, 1), overflowing_tokens: vec!(0, 0, 0, 0, 0, 0, 0, 0), num_truncated_tokens: 8 }
            ),
//            Truncation of both sentences (longest)
            (
                ("[UNK] hello  hello  hello  hello  hello", "!!!!!!!!"),
                TokenizedInput { token_ids: vec!(4, 2, 0, 0, 5, 3, 3, 3, 3, 5), segment_ids: vec!(0, 0, 0, 0, 0, 1, 1, 1, 1, 1), special_tokens_mask: vec!(1, 0, 0, 0, 1, 0, 0, 0, 0, 1), overflowing_tokens: vec!(0, 0, 0), num_truncated_tokens: 7 }
            )
        ];
        let source_texts: Vec<(&str, &str)> = test_tuples.iter().map(|v| v.0).collect();
        let expected_results: Vec<TokenizedInput> = test_tuples.iter().map(|v| v.1.clone()).collect();

//        When & Then
        for (source_text, expected_result) in test_tuples.iter() {
            assert_eq!(bert_tokenizer.encode(source_text.0, Some(source_text.1), 10, &truncation_strategy, 0),
                       *expected_result);
        }
        assert_eq!(bert_tokenizer.encode_pair_list(source_texts, 10, &truncation_strategy, 0), expected_results);
    }
}