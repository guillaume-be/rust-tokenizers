mod test_utils;

use rust_tokenizers::tokenizer::{
    M2M100Tokenizer, MultiThreadedTokenizer, Tokenizer, TruncationStrategy,
};
use rust_tokenizers::{Offset, TokenizedInput};
use test_utils::download_file_to_cache;

#[test]
fn test_m2m100_tokenization() -> anyhow::Result<()> {
    let vocab_path = download_file_to_cache(
        "https://huggingface.co/facebook/m2m100_418M/resolve/main/vocab.json",
        "m2m100_419M_vocab.json",
    )
    .unwrap();

    let merges_path = download_file_to_cache(
        "https://huggingface.co/facebook/m2m100_418M/resolve/main/sentencepiece.bpe.model",
        "m2m100_419M_spiece.model",
    )
    .unwrap();

    let mbart_tokenizer = M2M100Tokenizer::from_files(
        vocab_path.to_str().unwrap(),
        merges_path.to_str().unwrap(),
        false,
    )?;

    let original_strings = [
        ">>nl.<< …",
        ">>en.<< This is a sample sentence to be tokénized",
        ">>en.<< Wondering how this will get tokenized 🤔 ?",
        ">>fr.<< İs th!s 𩸽 Ϻ Šœ Ugljšić dấu nặng",
        ">>hi.<<   İs th!s    𩸽 Ϻ Šœ   Ugljšić  dấu nặng     ",
        ">>lt.<< � İs th!s �� 𩸽 Ϻ Šœ   Ugljšić  dấu nặng     ",
    ];

    let expected_results = [
        TokenizedInput {
            token_ids: vec![128067, 10, 2],
            segment_ids: vec![],
            special_tokens_mask: vec![],
            overflowing_tokens: vec![],
            num_truncated_tokens: 0,
            token_offsets: vec![None, Some(Offset { begin: 7, end: 9 }), None],
            reference_offsets: vec![],
            mask: vec![],
        },
        TokenizedInput {
            token_ids: vec![
                128022, 36606, 117, 8, 19580, 271, 8931, 6226, 128, 667, 6565, 1268, 68753, 2,
            ],
            segment_ids: vec![],
            special_tokens_mask: vec![],
            overflowing_tokens: vec![],
            num_truncated_tokens: 0,
            token_offsets: vec![
                None,
                Some(Offset { begin: 7, end: 12 }),
                Some(Offset { begin: 12, end: 15 }),
                Some(Offset { begin: 15, end: 17 }),
                Some(Offset { begin: 17, end: 22 }),
                Some(Offset { begin: 22, end: 24 }),
                Some(Offset { begin: 24, end: 29 }),
                Some(Offset { begin: 29, end: 33 }),
                Some(Offset { begin: 33, end: 36 }),
                Some(Offset { begin: 36, end: 39 }),
                Some(Offset { begin: 39, end: 43 }),
                Some(Offset { begin: 43, end: 46 }),
                Some(Offset { begin: 46, end: 50 }),
                None,
            ],
            reference_offsets: vec![],
            mask: vec![],
        },
        TokenizedInput {
            token_ids: vec![
                128022, 113315, 150, 40288, 15911, 13635, 6776, 6565, 49, 68753, 22, 125348, 375, 2,
            ],
            segment_ids: vec![],
            special_tokens_mask: vec![],
            overflowing_tokens: vec![],
            num_truncated_tokens: 0,
            token_offsets: vec![
                None,
                Some(Offset { begin: 7, end: 14 }),
                Some(Offset { begin: 14, end: 17 }),
                Some(Offset { begin: 17, end: 21 }),
                Some(Offset { begin: 21, end: 26 }),
                Some(Offset { begin: 26, end: 31 }),
                Some(Offset { begin: 31, end: 35 }),
                Some(Offset { begin: 35, end: 39 }),
                Some(Offset { begin: 39, end: 41 }),
                Some(Offset { begin: 41, end: 45 }),
                Some(Offset { begin: 45, end: 46 }),
                Some(Offset { begin: 46, end: 47 }),
                Some(Offset { begin: 47, end: 49 }),
                None,
            ],
            reference_offsets: vec![],
            mask: vec![],
        },
        TokenizedInput {
            token_ids: vec![
                128028, 42234, 1095, 30, 55, 22, 3, 22, 3, 2250, 64303, 16538, 6421, 63634, 48716,
                64883, 2,
            ],
            segment_ids: vec![],
            special_tokens_mask: vec![],
            overflowing_tokens: vec![],
            num_truncated_tokens: 0,
            token_offsets: vec![
                None,
                Some(Offset { begin: 7, end: 10 }),
                Some(Offset { begin: 10, end: 13 }),
                Some(Offset { begin: 13, end: 14 }),
                Some(Offset { begin: 14, end: 15 }),
                Some(Offset { begin: 15, end: 16 }),
                Some(Offset { begin: 16, end: 17 }),
                Some(Offset { begin: 17, end: 18 }),
                Some(Offset { begin: 18, end: 19 }),
                Some(Offset { begin: 19, end: 21 }),
                Some(Offset { begin: 21, end: 22 }),
                Some(Offset { begin: 22, end: 25 }),
                Some(Offset { begin: 25, end: 27 }),
                Some(Offset { begin: 27, end: 30 }),
                Some(Offset { begin: 30, end: 34 }),
                Some(Offset { begin: 34, end: 39 }),
                None,
            ],
            reference_offsets: vec![],
            mask: vec![],
        },
        TokenizedInput {
            token_ids: vec![
                128036, 42234, 1095, 30, 55, 22, 22, 22, 22, 3, 22, 3, 2250, 64303, 22, 22, 16538,
                6421, 63634, 22, 48716, 64883, 22, 22, 22, 22, 22, 2,
            ],
            segment_ids: vec![],
            special_tokens_mask: vec![],
            overflowing_tokens: vec![],
            num_truncated_tokens: 0,
            token_offsets: vec![
                None,
                Some(Offset { begin: 9, end: 12 }),
                Some(Offset { begin: 12, end: 15 }),
                Some(Offset { begin: 15, end: 16 }),
                Some(Offset { begin: 16, end: 17 }),
                Some(Offset { begin: 17, end: 18 }),
                Some(Offset { begin: 18, end: 19 }),
                Some(Offset { begin: 19, end: 20 }),
                Some(Offset { begin: 20, end: 21 }),
                Some(Offset { begin: 21, end: 22 }),
                Some(Offset { begin: 22, end: 23 }),
                Some(Offset { begin: 23, end: 24 }),
                Some(Offset { begin: 24, end: 26 }),
                Some(Offset { begin: 26, end: 27 }),
                Some(Offset { begin: 27, end: 28 }),
                Some(Offset { begin: 28, end: 29 }),
                Some(Offset { begin: 29, end: 32 }),
                Some(Offset { begin: 32, end: 34 }),
                Some(Offset { begin: 34, end: 37 }),
                Some(Offset { begin: 37, end: 38 }),
                Some(Offset { begin: 38, end: 42 }),
                Some(Offset { begin: 42, end: 47 }),
                Some(Offset { begin: 47, end: 48 }),
                Some(Offset { begin: 48, end: 49 }),
                Some(Offset { begin: 49, end: 50 }),
                Some(Offset { begin: 50, end: 51 }),
                Some(Offset { begin: 51, end: 52 }),
                None,
            ],
            reference_offsets: vec![],
            mask: vec![],
        },
        TokenizedInput {
            token_ids: vec![
                128057, 22, 42234, 1095, 30, 55, 22, 22, 3, 22, 3, 2250, 64303, 22, 22, 16538,
                6421, 63634, 22, 48716, 64883, 22, 22, 22, 22, 22, 2,
            ],
            segment_ids: vec![],
            special_tokens_mask: vec![],
            overflowing_tokens: vec![],
            num_truncated_tokens: 0,
            token_offsets: vec![
                None,
                Some(Offset { begin: 7, end: 8 }),
                Some(Offset { begin: 9, end: 12 }),
                Some(Offset { begin: 12, end: 15 }),
                Some(Offset { begin: 15, end: 16 }),
                Some(Offset { begin: 16, end: 17 }),
                Some(Offset { begin: 17, end: 18 }),
                Some(Offset { begin: 20, end: 21 }),
                Some(Offset { begin: 21, end: 22 }),
                Some(Offset { begin: 22, end: 23 }),
                Some(Offset { begin: 23, end: 24 }),
                Some(Offset { begin: 24, end: 26 }),
                Some(Offset { begin: 26, end: 27 }),
                Some(Offset { begin: 27, end: 28 }),
                Some(Offset { begin: 28, end: 29 }),
                Some(Offset { begin: 29, end: 32 }),
                Some(Offset { begin: 32, end: 34 }),
                Some(Offset { begin: 34, end: 37 }),
                Some(Offset { begin: 37, end: 38 }),
                Some(Offset { begin: 38, end: 42 }),
                Some(Offset { begin: 42, end: 47 }),
                Some(Offset { begin: 47, end: 48 }),
                Some(Offset { begin: 48, end: 49 }),
                Some(Offset { begin: 49, end: 50 }),
                Some(Offset { begin: 50, end: 51 }),
                Some(Offset { begin: 51, end: 52 }),
                None,
            ],
            reference_offsets: vec![],
            mask: vec![],
        },
    ]
    .to_vec();

    let output = MultiThreadedTokenizer::encode_list(
        &mbart_tokenizer,
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
                        mbart_tokenizer.decode(&[predicted.token_ids[idx]], false, false),
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
