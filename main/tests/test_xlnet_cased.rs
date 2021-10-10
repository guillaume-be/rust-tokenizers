mod test_utils;
use rust_tokenizers::tokenizer::{
    MultiThreadedTokenizer, Tokenizer, TruncationStrategy, XLNetTokenizer,
};
use rust_tokenizers::{Offset, TokenizedInput};
use test_utils::download_file_to_cache;

#[test]
fn test_xlnet_tokenization() -> anyhow::Result<()> {
    let vocab_path = download_file_to_cache(
        "https://cdn.huggingface.co/xlnet-base-cased-spiece.model",
        "xlnet-base-cased-spiece.model",
    )?;

    let xlnet_tokenizer = XLNetTokenizer::from_file(vocab_path.to_str().unwrap(), false, true)?;

    let original_strings = [
        "…",
        "This is a sample sentence to be tokénized",
        "Wondering how this will get tokenized 🤔 ?",
        "İs th!s 𩸽 Ϻ Šœ Ugljšić dấu nặng",
        "   İs th!s    𩸽 Ϻ Šœ   Ugljšić  dấu nặng     ",
        " � İs th!s �� 𩸽 Ϻ Šœ   Ugljšić  dấu nặng     ",
    ];

    let expected_results = [
        TokenizedInput {
            token_ids: vec![17, 9, 9, 9, 4, 3],
            segment_ids: vec![],
            special_tokens_mask: vec![],
            overflowing_tokens: vec![],
            num_truncated_tokens: 0,
            token_offsets: vec![
                Some(Offset { begin: 0, end: 1 }),
                Some(Offset { begin: 0, end: 1 }),
                Some(Offset { begin: 0, end: 1 }),
                Some(Offset { begin: 0, end: 1 }),
                None,
                None,
            ],
            reference_offsets: vec![],
            mask: vec![],
        },
        TokenizedInput {
            token_ids: vec![122, 27, 24, 4561, 3833, 22, 39, 17366, 1227, 4, 3],
            segment_ids: vec![],
            special_tokens_mask: vec![],
            overflowing_tokens: vec![],
            num_truncated_tokens: 0,
            token_offsets: vec![
                Some(Offset { begin: 0, end: 4 }),
                Some(Offset { begin: 4, end: 7 }),
                Some(Offset { begin: 7, end: 9 }),
                Some(Offset { begin: 9, end: 16 }),
                Some(Offset { begin: 16, end: 25 }),
                Some(Offset { begin: 25, end: 28 }),
                Some(Offset { begin: 28, end: 31 }),
                Some(Offset { begin: 31, end: 38 }),
                Some(Offset { begin: 38, end: 42 }),
                None,
                None,
            ],
            reference_offsets: vec![],
            mask: vec![],
        },
        TokenizedInput {
            token_ids: vec![
                14748, 56, 160, 52, 53, 133, 17366, 1227, 17, 0, 17, 82, 4, 3,
            ],
            segment_ids: vec![],
            special_tokens_mask: vec![],
            overflowing_tokens: vec![],
            num_truncated_tokens: 0,
            token_offsets: vec![
                Some(Offset { begin: 0, end: 6 }),
                Some(Offset { begin: 6, end: 9 }),
                Some(Offset { begin: 9, end: 13 }),
                Some(Offset { begin: 13, end: 18 }),
                Some(Offset { begin: 18, end: 23 }),
                Some(Offset { begin: 23, end: 27 }),
                Some(Offset { begin: 27, end: 33 }),
                Some(Offset { begin: 33, end: 37 }),
                Some(Offset { begin: 37, end: 38 }),
                Some(Offset { begin: 38, end: 39 }),
                Some(Offset { begin: 39, end: 40 }),
                Some(Offset { begin: 40, end: 41 }),
                None,
                None,
            ],
            reference_offsets: vec![],
            mask: vec![],
        },
        TokenizedInput {
            token_ids: vec![
                35, 23, 17, 138, 136, 23, 17, 0, 17, 0, 17, 83, 0, 128, 15222, 1315, 23, 556, 17,
                66, 2168, 17, 180, 2075, 4, 3,
            ],
            segment_ids: vec![],
            special_tokens_mask: vec![],
            overflowing_tokens: vec![],
            num_truncated_tokens: 0,
            token_offsets: vec![
                Some(Offset { begin: 0, end: 1 }),
                Some(Offset { begin: 1, end: 2 }),
                Some(Offset { begin: 2, end: 3 }),
                Some(Offset { begin: 3, end: 5 }),
                Some(Offset { begin: 5, end: 6 }),
                Some(Offset { begin: 6, end: 7 }),
                Some(Offset { begin: 7, end: 8 }),
                Some(Offset { begin: 8, end: 9 }),
                Some(Offset { begin: 9, end: 10 }),
                Some(Offset { begin: 10, end: 11 }),
                Some(Offset { begin: 11, end: 12 }),
                Some(Offset { begin: 12, end: 13 }),
                Some(Offset { begin: 13, end: 14 }),
                Some(Offset { begin: 14, end: 16 }),
                Some(Offset { begin: 16, end: 18 }),
                Some(Offset { begin: 18, end: 19 }),
                Some(Offset { begin: 19, end: 20 }),
                Some(Offset { begin: 20, end: 22 }),
                Some(Offset { begin: 22, end: 23 }),
                Some(Offset { begin: 23, end: 24 }),
                Some(Offset { begin: 24, end: 26 }),
                Some(Offset { begin: 26, end: 27 }),
                Some(Offset { begin: 27, end: 28 }),
                Some(Offset { begin: 28, end: 31 }),
                None,
                None,
            ],
            reference_offsets: vec![],
            mask: vec![],
        },
        TokenizedInput {
            token_ids: vec![
                17, 17, 35, 23, 17, 138, 136, 23, 17, 17, 17, 17, 0, 17, 0, 17, 83, 0, 17, 17, 128,
                15222, 1315, 23, 556, 17, 17, 66, 2168, 17, 180, 2075, 17, 17, 17, 17, 17, 4, 3,
            ],
            segment_ids: vec![],
            special_tokens_mask: vec![],
            overflowing_tokens: vec![],
            num_truncated_tokens: 0,
            token_offsets: vec![
                Some(Offset { begin: 0, end: 1 }),
                Some(Offset { begin: 1, end: 2 }),
                Some(Offset { begin: 2, end: 4 }),
                Some(Offset { begin: 4, end: 5 }),
                Some(Offset { begin: 5, end: 6 }),
                Some(Offset { begin: 6, end: 8 }),
                Some(Offset { begin: 8, end: 9 }),
                Some(Offset { begin: 9, end: 10 }),
                Some(Offset { begin: 10, end: 11 }),
                Some(Offset { begin: 11, end: 12 }),
                Some(Offset { begin: 12, end: 13 }),
                Some(Offset { begin: 13, end: 14 }),
                Some(Offset { begin: 14, end: 15 }),
                Some(Offset { begin: 15, end: 16 }),
                Some(Offset { begin: 16, end: 17 }),
                Some(Offset { begin: 17, end: 18 }),
                Some(Offset { begin: 18, end: 19 }),
                Some(Offset { begin: 19, end: 20 }),
                Some(Offset { begin: 20, end: 21 }),
                Some(Offset { begin: 21, end: 22 }),
                Some(Offset { begin: 22, end: 24 }),
                Some(Offset { begin: 24, end: 26 }),
                Some(Offset { begin: 26, end: 27 }),
                Some(Offset { begin: 27, end: 28 }),
                Some(Offset { begin: 28, end: 30 }),
                Some(Offset { begin: 30, end: 31 }),
                Some(Offset { begin: 31, end: 32 }),
                Some(Offset { begin: 32, end: 33 }),
                Some(Offset { begin: 33, end: 35 }),
                Some(Offset { begin: 35, end: 36 }),
                Some(Offset { begin: 36, end: 37 }),
                Some(Offset { begin: 37, end: 40 }),
                Some(Offset { begin: 40, end: 41 }),
                Some(Offset { begin: 41, end: 42 }),
                Some(Offset { begin: 42, end: 43 }),
                Some(Offset { begin: 43, end: 44 }),
                Some(Offset { begin: 44, end: 45 }),
                None,
                None,
            ],
            reference_offsets: vec![],
            mask: vec![],
        },
        TokenizedInput {
            token_ids: vec![
                17, 35, 23, 17, 138, 136, 23, 17, 17, 0, 17, 0, 17, 83, 0, 17, 17, 128, 15222,
                1315, 23, 556, 17, 17, 66, 2168, 17, 180, 2075, 17, 17, 17, 17, 17, 4, 3,
            ],
            segment_ids: vec![],
            special_tokens_mask: vec![],
            overflowing_tokens: vec![],
            num_truncated_tokens: 0,
            token_offsets: vec![
                Some(Offset { begin: 0, end: 1 }),
                Some(Offset { begin: 2, end: 4 }),
                Some(Offset { begin: 4, end: 5 }),
                Some(Offset { begin: 5, end: 6 }),
                Some(Offset { begin: 6, end: 8 }),
                Some(Offset { begin: 8, end: 9 }),
                Some(Offset { begin: 9, end: 10 }),
                Some(Offset { begin: 10, end: 11 }),
                Some(Offset { begin: 13, end: 14 }),
                Some(Offset { begin: 14, end: 15 }),
                Some(Offset { begin: 15, end: 16 }),
                Some(Offset { begin: 16, end: 17 }),
                Some(Offset { begin: 17, end: 18 }),
                Some(Offset { begin: 18, end: 19 }),
                Some(Offset { begin: 19, end: 20 }),
                Some(Offset { begin: 20, end: 21 }),
                Some(Offset { begin: 21, end: 22 }),
                Some(Offset { begin: 22, end: 24 }),
                Some(Offset { begin: 24, end: 26 }),
                Some(Offset { begin: 26, end: 27 }),
                Some(Offset { begin: 27, end: 28 }),
                Some(Offset { begin: 28, end: 30 }),
                Some(Offset { begin: 30, end: 31 }),
                Some(Offset { begin: 31, end: 32 }),
                Some(Offset { begin: 32, end: 33 }),
                Some(Offset { begin: 33, end: 35 }),
                Some(Offset { begin: 35, end: 36 }),
                Some(Offset { begin: 36, end: 37 }),
                Some(Offset { begin: 37, end: 40 }),
                Some(Offset { begin: 40, end: 41 }),
                Some(Offset { begin: 41, end: 42 }),
                Some(Offset { begin: 42, end: 43 }),
                Some(Offset { begin: 43, end: 44 }),
                Some(Offset { begin: 44, end: 45 }),
                None,
                None,
            ],
            reference_offsets: vec![],
            mask: vec![],
        },
    ]
    .to_vec();

    let output = MultiThreadedTokenizer::encode_list(
        &xlnet_tokenizer,
        &original_strings,
        128,
        &TruncationStrategy::LongestFirst,
        0,
    );

    for (_idx, (predicted, expected)) in output.iter().zip(expected_results.iter()).enumerate() {
        let original_sentence_chars: Vec<char> = original_strings[_idx].chars().collect();
        for (idx, offset) in predicted.token_offsets.iter().enumerate() {
            match offset {
                Some(offset) => {
                    let (start_char, end_char) = (offset.begin as usize, offset.end as usize);
                    let text: String = original_sentence_chars[start_char..end_char]
                        .iter()
                        .collect();
                    println!(
                        "{:<2?} | {:<10} | {:<10} | {:<10?}",
                        offset,
                        text,
                        xlnet_tokenizer.decode(vec!(predicted.token_ids[idx]), false, false),
                        predicted.mask[idx]
                    )
                }
                None => continue,
            }
        }

        assert_eq!(predicted.token_ids, expected.token_ids);
        assert_eq!(predicted.token_offsets, expected.token_offsets);
    }
    Ok(())
}
