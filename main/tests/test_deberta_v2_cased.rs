mod test_utils;

use rust_tokenizers::tokenizer::{DeBERTaV2Tokenizer, Tokenizer, TruncationStrategy};
use rust_tokenizers::{Offset, TokenizedInput};
use test_utils::download_file_to_cache;

#[test]
fn test_deberta_v2_tokenization() -> anyhow::Result<()> {
    let vocab_path = download_file_to_cache(
        "https://huggingface.co/microsoft/deberta-v3-base/resolve/main/spm.model",
        "deberta_v3_model.spm",
    )
    .unwrap();

    let deberta_v2_tokenizer = DeBERTaV2Tokenizer::from_file(vocab_path, false, false, false)?;

    let original_strings = [
        "…",
        "This is a sample sentence to be tokénized",
        "Wondering how this will get tokenized 🤔 ?",
        "İs th!s 𩸽 Ϻ Šœ Ugljšić dấu nặng",
        "İs th!s 𩸽 [MASK] Ϻ Šœ [MASK] dấu nặng",
    ];

    let expected_results = [
        TokenizedInput {
            token_ids: vec![1, 323, 260, 260, 2],
            segment_ids: vec![0, 0, 0, 0, 0],
            special_tokens_mask: vec![1, 0, 0, 0, 1],
            overflowing_tokens: vec![],
            num_truncated_tokens: 0,
            token_offsets: vec![
                None,
                Some(Offset { begin: 0, end: 1 }),
                Some(Offset { begin: 0, end: 1 }),
                Some(Offset { begin: 0, end: 1 }),
                None,
            ],
            reference_offsets: vec![],
            mask: vec![],
        },
        TokenizedInput {
            token_ids: vec![
                1, 329, 269, 266, 2783, 4378, 264, 282, 264, 1165, 28081, 4666, 2,
            ],
            segment_ids: vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            special_tokens_mask: vec![1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            overflowing_tokens: vec![],
            num_truncated_tokens: 0,
            token_offsets: vec![
                None,
                Some(Offset { begin: 0, end: 4 }),
                Some(Offset { begin: 4, end: 7 }),
                Some(Offset { begin: 7, end: 9 }),
                Some(Offset { begin: 9, end: 16 }),
                Some(Offset { begin: 16, end: 25 }),
                Some(Offset { begin: 25, end: 28 }),
                Some(Offset { begin: 28, end: 31 }),
                Some(Offset { begin: 31, end: 34 }),
                Some(Offset { begin: 34, end: 35 }),
                Some(Offset { begin: 35, end: 38 }),
                Some(Offset { begin: 38, end: 42 }),
                None,
            ],
            reference_offsets: vec![],
            mask: vec![],
        },
        TokenizedInput {
            token_ids: vec![
                1, 34559, 361, 291, 296, 350, 10704, 4666, 507, 123226, 1102, 2,
            ],
            segment_ids: vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            special_tokens_mask: vec![1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            overflowing_tokens: vec![],
            num_truncated_tokens: 0,
            token_offsets: vec![
                None,
                Some(Offset { begin: 0, end: 9 }),
                Some(Offset { begin: 9, end: 13 }),
                Some(Offset { begin: 13, end: 18 }),
                Some(Offset { begin: 18, end: 23 }),
                Some(Offset { begin: 23, end: 27 }),
                Some(Offset { begin: 27, end: 33 }),
                Some(Offset { begin: 33, end: 37 }),
                Some(Offset { begin: 37, end: 38 }),
                Some(Offset { begin: 38, end: 39 }),
                Some(Offset { begin: 39, end: 41 }),
                None,
            ],
            reference_offsets: vec![],
            mask: vec![],
        },
        TokenizedInput {
            token_ids: vec![
                1, 59070, 268, 6554, 300, 268, 507, 244, 173, 188, 193, 507, 211, 190, 37162,
                19078, 543, 18309, 3386, 120813, 1931, 97432, 1964, 2030, 117902, 5900, 2,
            ],
            segment_ids: vec![
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ],
            special_tokens_mask: vec![
                1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
            ],
            overflowing_tokens: vec![],
            num_truncated_tokens: 0,
            token_offsets: vec![
                None,
                Some(Offset { begin: 0, end: 1 }),
                Some(Offset { begin: 1, end: 2 }),
                Some(Offset { begin: 2, end: 5 }),
                Some(Offset { begin: 5, end: 6 }),
                Some(Offset { begin: 6, end: 7 }),
                Some(Offset { begin: 7, end: 8 }),
                Some(Offset { begin: 8, end: 9 }),
                Some(Offset { begin: 8, end: 9 }),
                Some(Offset { begin: 8, end: 9 }),
                Some(Offset { begin: 8, end: 9 }),
                Some(Offset { begin: 9, end: 10 }),
                Some(Offset { begin: 10, end: 11 }),
                Some(Offset { begin: 10, end: 11 }),
                Some(Offset { begin: 11, end: 13 }),
                Some(Offset { begin: 13, end: 14 }),
                Some(Offset { begin: 14, end: 16 }),
                Some(Offset { begin: 16, end: 18 }),
                Some(Offset { begin: 18, end: 19 }),
                Some(Offset { begin: 19, end: 22 }),
                Some(Offset { begin: 22, end: 24 }),
                Some(Offset { begin: 24, end: 25 }),
                Some(Offset { begin: 25, end: 26 }),
                Some(Offset { begin: 26, end: 28 }),
                Some(Offset { begin: 28, end: 29 }),
                Some(Offset { begin: 29, end: 31 }),
                None,
            ],
            reference_offsets: vec![],
            mask: vec![],
        },
        TokenizedInput {
            token_ids: vec![
                1, 59070, 268, 6554, 300, 268, 507, 244, 173, 188, 193, 128000, 507, 211, 190,
                37162, 19078, 128000, 1931, 97432, 1964, 2030, 117902, 5900, 2,
            ],
            segment_ids: vec![
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ],
            special_tokens_mask: vec![
                1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
            ],
            overflowing_tokens: vec![],
            num_truncated_tokens: 0,
            token_offsets: vec![
                None,
                Some(Offset { begin: 0, end: 1 }),
                Some(Offset { begin: 1, end: 2 }),
                Some(Offset { begin: 2, end: 5 }),
                Some(Offset { begin: 5, end: 6 }),
                Some(Offset { begin: 6, end: 7 }),
                Some(Offset { begin: 7, end: 8 }),
                Some(Offset { begin: 8, end: 9 }),
                Some(Offset { begin: 8, end: 9 }),
                Some(Offset { begin: 8, end: 9 }),
                Some(Offset { begin: 8, end: 9 }),
                Some(Offset { begin: 10, end: 16 }),
                Some(Offset { begin: 16, end: 17 }),
                Some(Offset { begin: 17, end: 18 }),
                Some(Offset { begin: 17, end: 18 }),
                Some(Offset { begin: 18, end: 20 }),
                Some(Offset { begin: 20, end: 21 }),
                Some(Offset { begin: 22, end: 28 }),
                Some(Offset { begin: 28, end: 30 }),
                Some(Offset { begin: 30, end: 31 }),
                Some(Offset { begin: 31, end: 32 }),
                Some(Offset { begin: 32, end: 34 }),
                Some(Offset { begin: 34, end: 35 }),
                Some(Offset { begin: 35, end: 37 }),
                None,
            ],
            reference_offsets: vec![],
            mask: vec![],
        },
    ]
    .to_vec();

    let output = deberta_v2_tokenizer.encode_list(
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
                        deberta_v2_tokenizer.decode(&[predicted.token_ids[idx]], false, false),
                        predicted.mask[idx]
                    )
                }
                None => continue,
            }
        }

        assert_eq!(predicted.token_ids, expected.token_ids);
        assert_eq!(predicted.special_tokens_mask, expected.special_tokens_mask);
        assert_eq!(predicted.token_offsets, expected.token_offsets);
    }
    Ok(())
}
