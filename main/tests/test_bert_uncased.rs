use std::sync::Arc;
use rust_tokenizers::{Vocab, BertTokenizer, TruncationStrategy, Tokenizer, TokenizedInput};
use rust_tokenizers::preprocessing::tokenizer::base_tokenizer::Offset;
mod test_utils;
use test_utils::download_file_to_cache;


#[test]
fn test_bert_tokenization() {
    let vocab_path = download_file_to_cache("https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt",
                                            "bert-base-uncased_vocab.txt").unwrap();

    let vocab = Arc::new(rust_tokenizers::BertVocab::from_file(vocab_path.to_str().unwrap()));
    let bert_tokenizer: BertTokenizer = BertTokenizer::from_existing_vocab(vocab.clone(), true);


    let original_strings = [
        "This is a sample sentence to be tokénized",
        "Hello, y'all! How are you 😁 ?",
        "İs th!s 𩸽 Ϻ Šœ Ugljšić dấu nặng",
        "İs th!s   𩸽 [SEP] Ϻ Šœ  Uglj[SEP]šić   dấu nặng",
        "   İs th!s    𩸽 Ϻ Šœ   Ugljšić  dấu nặng     ",
        "  �� İs th!s   ���� 𩸽 Ϻ Šœ   Ugljšić  dấu nặng     ",
    ];

    let expected_results = [
        TokenizedInput {
            token_ids: vec!(101, 2023, 2003, 1037, 7099, 6251, 2000, 2022, 19204, 3550, 102),
            segment_ids: vec!(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            special_tokens_mask: vec!(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
            overflowing_tokens: vec!(),
            num_truncated_tokens: 0,
            token_offsets: vec!(None, Some(Offset { begin: 0, end: 4 }), Some(Offset { begin: 5, end: 7 }), Some(Offset { begin: 8, end: 9 }), Some(Offset { begin: 10, end: 16 }),
                                Some(Offset { begin: 17, end: 25 }), Some(Offset { begin: 26, end: 28 }), Some(Offset { begin: 29, end: 31 }), Some(Offset { begin: 32, end: 38 }),
                                Some(Offset { begin: 38, end: 42 }), None),
            reference_offsets: vec!(),
            mask: vec!(),
        },
        TokenizedInput {
            token_ids: vec!(101, 7592, 1010, 1061, 1005, 2035, 999, 2129, 2024, 2017, 100, 1029, 102),
            segment_ids: vec!(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            special_tokens_mask: vec!(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
            overflowing_tokens: vec!(),
            num_truncated_tokens: 0,
            token_offsets: vec!(None, Some(Offset { begin: 0, end: 5 }), Some(Offset { begin: 5, end: 6 }), Some(Offset { begin: 7, end: 8 }), Some(Offset { begin: 8, end: 9 }),
                                Some(Offset { begin: 9, end: 12 }), Some(Offset { begin: 12, end: 13 }), Some(Offset { begin: 14, end: 17 }), Some(Offset { begin: 18, end: 21 }),
                                Some(Offset { begin: 22, end: 25 }), Some(Offset { begin: 26, end: 27 }), Some(Offset { begin: 28, end: 29 }), None),
            reference_offsets: vec!(),
            mask: vec!(),
        },
        TokenizedInput {
            token_ids: vec!(101, 2003, 16215, 999, 1055, 100, 100, 1055, 29674, 1057, 23296, 22578, 2594, 4830, 2226, 16660, 2290, 102),
            segment_ids: vec!(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            special_tokens_mask: vec!(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
            overflowing_tokens: vec!(),
            num_truncated_tokens: 0,
            token_offsets: vec!(None, Some(Offset { begin: 0, end: 2 }), Some(Offset { begin: 3, end: 5 }), Some(Offset { begin: 5, end: 6 }), Some(Offset { begin: 6, end: 7 }),
                                Some(Offset { begin: 8, end: 9 }), Some(Offset { begin: 10, end: 11 }), Some(Offset { begin: 12, end: 13 }), Some(Offset { begin: 13, end: 14 }),
                                Some(Offset { begin: 15, end: 16 }), Some(Offset { begin: 16, end: 18 }), Some(Offset { begin: 18, end: 20 }), Some(Offset { begin: 20, end: 22 }),
                                Some(Offset { begin: 23, end: 25 }), Some(Offset { begin: 25, end: 26 }), Some(Offset { begin: 27, end: 30 }), Some(Offset { begin: 30, end: 31 }), None),
            reference_offsets: vec!(),
            mask: vec!(),
        },
        TokenizedInput {
            token_ids: vec!(101, 2003, 16215, 999, 1055, 100, 102, 100, 1055, 29674, 1057, 23296, 3501, 102, 14387, 4830, 2226, 16660, 2290, 102),
            segment_ids: vec!(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            special_tokens_mask: vec!(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
            overflowing_tokens: vec!(),
            num_truncated_tokens: 0,
            token_offsets: vec!(None, Some(Offset { begin: 0, end: 2 }), Some(Offset { begin: 3, end: 5 }), Some(Offset { begin: 5, end: 6 }), Some(Offset { begin: 6, end: 7 }),
                                Some(Offset { begin: 10, end: 11 }), Some(Offset { begin: 12, end: 17 }), Some(Offset { begin: 18, end: 19 }), Some(Offset { begin: 20, end: 21 }),
                                Some(Offset { begin: 21, end: 22 }), Some(Offset { begin: 24, end: 25 }), Some(Offset { begin: 25, end: 27 }), Some(Offset { begin: 27, end: 28 }),
                                Some(Offset { begin: 28, end: 33 }), Some(Offset { begin: 33, end: 36 }), Some(Offset { begin: 39, end: 41 }), Some(Offset { begin: 41, end: 42 }),
                                Some(Offset { begin: 43, end: 46 }), Some(Offset { begin: 46, end: 47 }),None),
            reference_offsets: vec!(),
            mask: vec!(),
        },
        TokenizedInput {
            token_ids: vec!(101, 2003, 16215, 999, 1055, 100, 100, 1055, 29674, 1057, 23296, 22578, 2594, 4830, 2226, 16660, 2290, 102),
            segment_ids: vec!(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            special_tokens_mask: vec!(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
            overflowing_tokens: vec!(),
            num_truncated_tokens: 0,
            token_offsets: vec!(None, Some(Offset { begin: 3, end: 5 }), Some(Offset { begin: 6, end: 8 }), Some(Offset { begin: 8, end: 9 }), Some(Offset { begin: 9, end: 10 }),
                                Some(Offset { begin: 14, end: 15 }), Some(Offset { begin: 16, end: 17 }), Some(Offset { begin: 18, end: 19 }), Some(Offset { begin: 19, end: 20 }),
                                Some(Offset { begin: 23, end: 24 }), Some(Offset { begin: 24, end: 26 }), Some(Offset { begin: 26, end: 28 }), Some(Offset { begin: 28, end: 30 }),
                                Some(Offset { begin: 32, end: 34 }), Some(Offset { begin: 34, end: 35 }), Some(Offset { begin: 36, end: 39 }), Some(Offset { begin: 39, end: 40 }), None),
            reference_offsets: vec!(),
            mask: vec!(),
        },
        TokenizedInput {
            token_ids: vec!(101, 2003, 16215, 999, 1055, 100, 100, 1055, 29674, 1057, 23296, 22578, 2594, 4830, 2226, 16660, 2290, 102),
            segment_ids: vec!(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            special_tokens_mask: vec!(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
            overflowing_tokens: vec!(),
            num_truncated_tokens: 0,
            token_offsets: vec!(None, Some(Offset { begin: 5, end: 7 }), Some(Offset { begin: 8, end: 10 }), Some(Offset { begin: 10, end: 11 }), Some(Offset { begin: 11, end: 12 }),
                                Some(Offset { begin: 20, end: 21 }), Some(Offset { begin: 22, end: 23 }), Some(Offset { begin: 24, end: 25 }), Some(Offset { begin: 25, end: 26 }),
                                Some(Offset { begin: 29, end: 30 }), Some(Offset { begin: 30, end: 32 }), Some(Offset { begin: 32, end: 34 }), Some(Offset { begin: 34, end: 36 }),
                                Some(Offset { begin: 38, end: 40 }), Some(Offset { begin: 40, end: 41 }), Some(Offset { begin: 42, end: 45 }), Some(Offset { begin: 45, end: 46 }), None),
            reference_offsets: vec!(),
            mask: vec!(),
        },
    ].to_vec();

    let output = bert_tokenizer.encode_list(original_strings.to_vec(),
                                            128,
                                            &TruncationStrategy::LongestFirst,
                                            0);

    for (_idx, (predicted, expected)) in output.iter().zip(expected_results.iter()).enumerate() {

//        let original_sentence_chars: Vec<char> = original_strings[_idx].chars().collect();
//        for offset in &predicted.token_offsets {
//            match offset {
//                Some(offset) => {
//                    let (start_char, end_char) = (offset.begin as usize, offset.end as usize);
//                    let text: String = original_sentence_chars[start_char..end_char].iter().collect();
//                    println!("{:?} -  {}", offset, text)
//                }
//                None => continue
//            }
//        };

        assert_eq!(predicted.token_ids, expected.token_ids);
        assert_eq!(predicted.special_tokens_mask, expected.special_tokens_mask);
        assert_eq!(predicted.token_offsets, expected.token_offsets);
    }
}
