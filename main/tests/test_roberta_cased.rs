use rust_tokenizers::{TruncationStrategy, Tokenizer, TokenizedInput, RobertaTokenizer};
use rust_tokenizers::preprocessing::tokenizer::base_tokenizer::Offset;

mod test_utils;

use test_utils::download_file_to_cache;


#[test]
fn test_roberta_tokenization() -> anyhow::Result<()> {
    let vocab_path = download_file_to_cache("https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-vocab.json",
                                            "roberta_vocab.json").unwrap();

    let merges_path = download_file_to_cache("https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-merges.txt",
                                             "roberta_merges.txt").unwrap();

    let roberta_tokenizer = RobertaTokenizer::from_file(vocab_path.to_str().unwrap(),
                                                        merges_path.to_str().unwrap(),
                                                        false)?;


    let original_strings = [
        "This is a sample sentence to be tokénized",
        "Wondering how this will get tokenized 🤔 ?",
        "İs th!s 𩸽 Ϻ Šœ Ugljšić dấu nặng",
        "İs th!s   𩸽 </s> Ϻ Šœ  Uglj</s>šić   dấu nặng",
        "   İs th!s    𩸽 Ϻ Šœ   Ugljšić  dấu nặng     ",
        "  �� İs th!s   ���� 𩸽 Ϻ Šœ   Ugljšić  dấu nặng     ",
    ];

    let expected_results = [
        TokenizedInput {
            token_ids: vec!(0, 152, 16, 10, 7728, 3645, 7, 28, 7, 1071, 44025, 10172, 282, 1538, 2),
            segment_ids: vec!(),
            special_tokens_mask: vec!(),
            overflowing_tokens: vec!(),
            num_truncated_tokens: 0,
            token_offsets: vec!(None, Some(Offset { begin: 0, end: 4 }), Some(Offset { begin: 4, end: 7 }), Some(Offset { begin: 7, end: 9 }), Some(Offset { begin: 9, end: 16 }),
                                Some(Offset { begin: 16, end: 25 }), Some(Offset { begin: 25, end: 28 }), Some(Offset { begin: 28, end: 31 }), Some(Offset { begin: 31, end: 34 }),
                                Some(Offset { begin: 34, end: 36 }), Some(Offset { begin: 36, end: 37 }), Some(Offset { begin: 36, end: 37 }), Some(Offset { begin: 37, end: 38 }),
                                Some(Offset { begin: 38, end: 42 }), None),
            reference_offsets: vec!(),
            mask: vec!(),
        },
        TokenizedInput {
            token_ids: vec!(0, 39980, 2961, 141, 42, 40, 120, 19233, 1538, 8103, 10470, 10674, 17487, 2),
            segment_ids: vec!(),
            special_tokens_mask: vec!(),
            overflowing_tokens: vec!(),
            num_truncated_tokens: 0,
            token_offsets: vec!(None, Some(Offset { begin: 0, end: 4 }), Some(Offset { begin: 4, end: 9 }), Some(Offset { begin: 9, end: 13 }), Some(Offset { begin: 13, end: 18 }),
                                Some(Offset { begin: 18, end: 23 }), Some(Offset { begin: 23, end: 27 }), Some(Offset { begin: 27, end: 33 }), Some(Offset { begin: 33, end: 37 }),
                                Some(Offset { begin: 37, end: 39 }), Some(Offset { begin: 38, end: 39 }), Some(Offset { begin: 38, end: 39 }), Some(Offset { begin: 39, end: 41 }), None),
            reference_offsets: vec!(),
            mask: vec!(),
        },
        TokenizedInput {
            token_ids: vec!(0, 4236, 7487, 29, 3553, 328, 29, 1437, 49585, 15375, 18537, 10809, 46927, 3070, 2742, 21402, 1277, 9085, 121, 7210, 267, 4654, 118, 4807, 385, 1376, 3070, 8210, 257, 295, 1376, 3070, 18400, 2590, 2),
            segment_ids: vec!(),
            special_tokens_mask: vec!(),
            overflowing_tokens: vec!(),
            num_truncated_tokens: 0,
            token_offsets: vec!(None, Some(Offset { begin: 0, end: 1 }), Some(Offset { begin: 0, end: 1 }), Some(Offset { begin: 1, end: 2 }), Some(Offset { begin: 2, end: 5 }),
                                Some(Offset { begin: 5, end: 6 }), Some(Offset { begin: 6, end: 7 }), Some(Offset { begin: 7, end: 8 }), Some(Offset { begin: 8, end: 9 }),
                                Some(Offset { begin: 8, end: 9 }), Some(Offset { begin: 8, end: 9 }), Some(Offset { begin: 8, end: 9 }), Some(Offset { begin: 9, end: 11 }),
                                Some(Offset { begin: 10, end: 11 }), Some(Offset { begin: 11, end: 13 }), Some(Offset { begin: 12, end: 13 }), Some(Offset { begin: 13, end: 14 }),
                                Some(Offset { begin: 13, end: 14 }), Some(Offset { begin: 14, end: 16 }), Some(Offset { begin: 16, end: 18 }), Some(Offset { begin: 18, end: 19 }),
                                Some(Offset { begin: 19, end: 20 }), Some(Offset { begin: 20, end: 21 }), Some(Offset { begin: 21, end: 22 }), Some(Offset { begin: 22, end: 24 }),
                                Some(Offset { begin: 24, end: 25 }), Some(Offset { begin: 24, end: 25 }), Some(Offset { begin: 24, end: 25 }), Some(Offset { begin: 25, end: 26 }),
                                Some(Offset { begin: 26, end: 28 }), Some(Offset { begin: 28, end: 29 }), Some(Offset { begin: 28, end: 29 }), Some(Offset { begin: 28, end: 29 }),
                                Some(Offset { begin: 29, end: 31 }), None),
            reference_offsets: vec!(),
            mask: vec!(),
        },
        TokenizedInput {
            token_ids: vec!(0, 4236, 7487, 29, 3553, 328, 29, 1437, 1437, 1437, 49585, 15375, 18537, 10809, 2, 46927, 3070, 2742, 21402, 1277, 9085, 1437, 121, 7210, 267, 2, 4654, 118, 4807, 1437, 1437, 385, 1376, 3070, 8210, 257, 295, 1376, 3070, 18400, 2590, 2),
            segment_ids: vec!(),
            special_tokens_mask: vec!(),
            overflowing_tokens: vec!(),
            num_truncated_tokens: 0,
            token_offsets: vec!(None, Some(Offset { begin: 0, end: 1 }), Some(Offset { begin: 0, end: 1 }), Some(Offset { begin: 1, end: 2 }), Some(Offset { begin: 2, end: 5 }),
                                Some(Offset { begin: 5, end: 6 }), Some(Offset { begin: 6, end: 7 }), Some(Offset { begin: 7, end: 8 }), Some(Offset { begin: 8, end: 9 }),
                                Some(Offset { begin: 9, end: 10 }), Some(Offset { begin: 10, end: 11 }), Some(Offset { begin: 10, end: 11 }), Some(Offset { begin: 10, end: 11 }),
                                Some(Offset { begin: 10, end: 11 }), Some(Offset { begin: 12, end: 16 }), Some(Offset { begin: 16, end: 18 }), Some(Offset { begin: 17, end: 18 }),
                                Some(Offset { begin: 18, end: 20 }), Some(Offset { begin: 19, end: 20 }), Some(Offset { begin: 20, end: 21 }), Some(Offset { begin: 20, end: 21 }),
                                Some(Offset { begin: 21, end: 22 }), Some(Offset { begin: 22, end: 24 }), Some(Offset { begin: 24, end: 26 }), Some(Offset { begin: 26, end: 27 }),
                                Some(Offset { begin: 27, end: 31 }), Some(Offset { begin: 31, end: 32 }), Some(Offset { begin: 32, end: 33 }), Some(Offset { begin: 33, end: 34 }),
                                Some(Offset { begin: 34, end: 35 }), Some(Offset { begin: 35, end: 36 }), Some(Offset { begin: 36, end: 38 }), Some(Offset { begin: 38, end: 39 }),
                                Some(Offset { begin: 38, end: 39 }), Some(Offset { begin: 38, end: 39 }), Some(Offset { begin: 39, end: 40 }), Some(Offset { begin: 40, end: 42 }),
                                Some(Offset { begin: 42, end: 43 }), Some(Offset { begin: 42, end: 43 }), Some(Offset { begin: 42, end: 43 }), Some(Offset { begin: 43, end: 45 }), None),
            reference_offsets: vec!(),
            mask: vec!(),
        },
        TokenizedInput {
            token_ids: vec!(0, 1437, 1437, 4236, 7487, 29, 3553, 328, 29, 1437, 1437, 1437, 1437, 49585, 15375, 18537, 10809, 46927, 3070, 2742, 21402, 1277, 9085, 1437, 1437, 121, 7210, 267, 4654, 118, 4807, 1437, 385, 1376, 3070, 8210, 257, 295, 1376, 3070, 18400, 2590, 2),
            segment_ids: vec!(),
            special_tokens_mask: vec!(),
            overflowing_tokens: vec!(),
            num_truncated_tokens: 0,
            token_offsets: vec!(None, Some(Offset { begin: 0, end: 1 }), Some(Offset { begin: 1, end: 2 }), Some(Offset { begin: 2, end: 4 }), Some(Offset { begin: 3, end: 4 }),
                                Some(Offset { begin: 4, end: 5 }), Some(Offset { begin: 5, end: 8 }), Some(Offset { begin: 8, end: 9 }), Some(Offset { begin: 9, end: 10 }),
                                Some(Offset { begin: 10, end: 11 }), Some(Offset { begin: 11, end: 12 }), Some(Offset { begin: 12, end: 13 }), Some(Offset { begin: 13, end: 14 }),
                                Some(Offset { begin: 14, end: 15 }), Some(Offset { begin: 14, end: 15 }), Some(Offset { begin: 14, end: 15 }), Some(Offset { begin: 14, end: 15 }),
                                Some(Offset { begin: 15, end: 17 }), Some(Offset { begin: 16, end: 17 }), Some(Offset { begin: 17, end: 19 }), Some(Offset { begin: 18, end: 19 }),
                                Some(Offset { begin: 19, end: 20 }), Some(Offset { begin: 19, end: 20 }), Some(Offset { begin: 20, end: 21 }), Some(Offset { begin: 21, end: 22 }),
                                Some(Offset { begin: 22, end: 24 }), Some(Offset { begin: 24, end: 26 }), Some(Offset { begin: 26, end: 27 }), Some(Offset { begin: 27, end: 28 }),
                                Some(Offset { begin: 28, end: 29 }), Some(Offset { begin: 29, end: 30 }), Some(Offset { begin: 30, end: 31 }), Some(Offset { begin: 31, end: 33 }),
                                Some(Offset { begin: 33, end: 34 }), Some(Offset { begin: 33, end: 34 }), Some(Offset { begin: 33, end: 34 }), Some(Offset { begin: 34, end: 35 }),
                                Some(Offset { begin: 35, end: 37 }), Some(Offset { begin: 37, end: 38 }), Some(Offset { begin: 37, end: 38 }), Some(Offset { begin: 37, end: 38 }),
                                Some(Offset { begin: 38, end: 40 }), None),
            reference_offsets: vec!(),
            mask: vec!(),
        },
        TokenizedInput {
            token_ids: vec!(0, 1437, 1437, 47119, 4236, 7487, 29, 3553, 328, 29, 1437, 1437, 48693, 1437, 49585, 15375, 18537, 10809, 46927, 3070, 2742, 21402, 1277, 9085, 1437, 1437, 121, 7210, 267, 4654, 118, 4807, 1437, 385, 1376, 3070, 8210, 257, 295, 1376, 3070, 18400, 2590, 2),
            segment_ids: vec!(),
            special_tokens_mask: vec!(),
            overflowing_tokens: vec!(),
            num_truncated_tokens: 0,
            token_offsets: vec!(None, Some(Offset { begin: 0, end: 1 }), Some(Offset { begin: 1, end: 2 }), Some(Offset { begin: 2, end: 4 }), Some(Offset { begin: 4, end: 6 }),
                                Some(Offset { begin: 5, end: 6 }), Some(Offset { begin: 6, end: 7 }), Some(Offset { begin: 7, end: 10 }), Some(Offset { begin: 10, end: 11 }),
                                Some(Offset { begin: 11, end: 12 }), Some(Offset { begin: 12, end: 13 }), Some(Offset { begin: 13, end: 14 }), Some(Offset { begin: 14, end: 19 }),
                                Some(Offset { begin: 19, end: 20 }), Some(Offset { begin: 20, end: 21 }), Some(Offset { begin: 20, end: 21 }), Some(Offset { begin: 20, end: 21 }),
                                Some(Offset { begin: 20, end: 21 }), Some(Offset { begin: 21, end: 23 }), Some(Offset { begin: 22, end: 23 }), Some(Offset { begin: 23, end: 25 }),
                                Some(Offset { begin: 24, end: 25 }), Some(Offset { begin: 25, end: 26 }), Some(Offset { begin: 25, end: 26 }), Some(Offset { begin: 26, end: 27 }),
                                Some(Offset { begin: 27, end: 28 }), Some(Offset { begin: 28, end: 30 }), Some(Offset { begin: 30, end: 32 }), Some(Offset { begin: 32, end: 33 }),
                                Some(Offset { begin: 33, end: 34 }), Some(Offset { begin: 34, end: 35 }), Some(Offset { begin: 35, end: 36 }), Some(Offset { begin: 36, end: 37 }),
                                Some(Offset { begin: 37, end: 39 }), Some(Offset { begin: 39, end: 40 }), Some(Offset { begin: 39, end: 40 }), Some(Offset { begin: 39, end: 40 }),
                                Some(Offset { begin: 40, end: 41 }), Some(Offset { begin: 41, end: 43 }), Some(Offset { begin: 43, end: 44 }), Some(Offset { begin: 43, end: 44 }),
                                Some(Offset { begin: 43, end: 44 }), Some(Offset { begin: 44, end: 46 }), None),
            reference_offsets: vec!(),
            mask: vec!(),
        },
    ].to_vec();

    let output = roberta_tokenizer.encode_list(original_strings.to_vec(),
                                               128,
                                               &TruncationStrategy::LongestFirst,
                                               0)?;


    for (_idx, (predicted, expected)) in output.iter().zip(expected_results.iter()).enumerate() {
        let original_sentence_chars: Vec<char> = original_strings[_idx].chars().collect();
        for (idx, offset) in predicted.token_offsets.iter().enumerate() {
            match offset {
                Some(offset) => {
                    let (start_char, end_char) = (offset.begin as usize, offset.end as usize);
                    let text: String = original_sentence_chars[start_char..end_char].iter().collect();
                    println!("{:<2?} | {:<10} | {:<10} | {:<10?}", offset, text, roberta_tokenizer.decode(vec!(predicted.token_ids[idx]), false, false), predicted.mask[idx])
                }
                None => continue
            }
        };

        assert_eq!(predicted.token_ids, expected.token_ids);
        assert_eq!(predicted.token_offsets, expected.token_offsets);
    }
    Ok(())
}
