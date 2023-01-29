mod test_utils;

use rust_tokenizers::tokenizer::{DeBERTaTokenizer, Tokenizer, TruncationStrategy};
use rust_tokenizers::{Offset, TokenizedInput};
use test_utils::download_file_to_cache;

#[test]
fn test_deberta_tokenization() -> anyhow::Result<()> {
    let vocab_path = download_file_to_cache(
        "https://huggingface.co/microsoft/deberta-base/resolve/main/vocab.json",
    )
    .unwrap();

    let merges_path = download_file_to_cache(
        "https://huggingface.co/microsoft/deberta-base/resolve/main/merges.txt",
    )
    .unwrap();

    let deberta_tokenizer = DeBERTaTokenizer::from_file(vocab_path, merges_path, false)?;

    let original_strings = [
        "…",
        "This is a sample sentence to be tokénized",
        "Wondering how this will get tokenized 🤔 ?",
        "İs th!s 𩸽 Ϻ Šœ Ugljšić dấu nặng",
        "İs th!s 𩸽 [MASK] Ϻ Šœ [MASK] dấu nặng",
    ];

    let expected_results = [
        TokenizedInput {
            token_ids: vec![1, 1174, 2],
            segment_ids: vec![0, 0, 0],
            special_tokens_mask: vec![1, 0, 1],
            overflowing_tokens: vec![],
            num_truncated_tokens: 0,
            token_offsets: vec![None, Some(Offset { begin: 0, end: 1 }), None],
            reference_offsets: vec![],
            mask: vec![],
        },
        TokenizedInput {
            token_ids: vec![
                1, 713, 16, 10, 7728, 3645, 7, 28, 7, 1071, 44025, 10172, 282, 1538, 2,
            ],
            segment_ids: vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            special_tokens_mask: vec![1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
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
                Some(Offset { begin: 34, end: 36 }),
                Some(Offset { begin: 36, end: 37 }),
                Some(Offset { begin: 36, end: 37 }),
                Some(Offset { begin: 37, end: 38 }),
                Some(Offset { begin: 38, end: 42 }),
                None,
            ],
            reference_offsets: vec![],
            mask: vec![],
        },
        TokenizedInput {
            token_ids: vec![
                1, 771, 2832, 2961, 141, 42, 40, 120, 19233, 1538, 8103, 10470, 10674, 17487, 2,
            ],
            segment_ids: vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            special_tokens_mask: vec![1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            overflowing_tokens: vec![],
            num_truncated_tokens: 0,
            token_offsets: vec![
                None,
                Some(Offset { begin: 0, end: 1 }),
                Some(Offset { begin: 1, end: 4 }),
                Some(Offset { begin: 4, end: 9 }),
                Some(Offset { begin: 9, end: 13 }),
                Some(Offset { begin: 13, end: 18 }),
                Some(Offset { begin: 18, end: 23 }),
                Some(Offset { begin: 23, end: 27 }),
                Some(Offset { begin: 27, end: 33 }),
                Some(Offset { begin: 33, end: 37 }),
                Some(Offset { begin: 37, end: 39 }),
                Some(Offset { begin: 38, end: 39 }),
                Some(Offset { begin: 38, end: 39 }),
                Some(Offset { begin: 39, end: 41 }),
                None,
            ],
            reference_offsets: vec![],
            mask: vec![],
        },
        TokenizedInput {
            token_ids: vec![
                1, 649, 7487, 29, 3553, 328, 29, 1437, 49585, 15375, 18537, 10809, 46927, 3070,
                2742, 21402, 1277, 9085, 121, 7210, 267, 4654, 118, 4807, 385, 1376, 3070, 8210,
                257, 295, 1376, 3070, 18400, 2590, 2,
            ],
            segment_ids: vec![
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
            ],
            special_tokens_mask: vec![
                1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 1,
            ],
            overflowing_tokens: vec![],
            num_truncated_tokens: 0,
            token_offsets: vec![
                None,
                Some(Offset { begin: 0, end: 1 }),
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
                Some(Offset { begin: 9, end: 11 }),
                Some(Offset { begin: 10, end: 11 }),
                Some(Offset { begin: 11, end: 13 }),
                Some(Offset { begin: 12, end: 13 }),
                Some(Offset { begin: 13, end: 14 }),
                Some(Offset { begin: 13, end: 14 }),
                Some(Offset { begin: 14, end: 16 }),
                Some(Offset { begin: 16, end: 18 }),
                Some(Offset { begin: 18, end: 19 }),
                Some(Offset { begin: 19, end: 20 }),
                Some(Offset { begin: 20, end: 21 }),
                Some(Offset { begin: 21, end: 22 }),
                Some(Offset { begin: 22, end: 24 }),
                Some(Offset { begin: 24, end: 25 }),
                Some(Offset { begin: 24, end: 25 }),
                Some(Offset { begin: 24, end: 25 }),
                Some(Offset { begin: 25, end: 26 }),
                Some(Offset { begin: 26, end: 28 }),
                Some(Offset { begin: 28, end: 29 }),
                Some(Offset { begin: 28, end: 29 }),
                Some(Offset { begin: 28, end: 29 }),
                Some(Offset { begin: 29, end: 31 }),
                None,
            ],
            reference_offsets: vec![],
            mask: vec![],
        },
        TokenizedInput {
            token_ids: vec![
                1, 649, 7487, 29, 3553, 328, 29, 1437, 49585, 15375, 18537, 10809, 50264, 46927,
                3070, 2742, 21402, 1277, 9085, 50264, 385, 1376, 3070, 8210, 257, 295, 1376, 3070,
                18400, 2590, 2,
            ],
            segment_ids: vec![
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0,
            ],
            special_tokens_mask: vec![
                1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 1,
            ],
            overflowing_tokens: vec![],
            num_truncated_tokens: 0,
            token_offsets: vec![
                None,
                Some(Offset { begin: 0, end: 1 }),
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
                Some(Offset { begin: 16, end: 18 }),
                Some(Offset { begin: 17, end: 18 }),
                Some(Offset { begin: 18, end: 20 }),
                Some(Offset { begin: 19, end: 20 }),
                Some(Offset { begin: 20, end: 21 }),
                Some(Offset { begin: 20, end: 21 }),
                Some(Offset { begin: 22, end: 28 }),
                Some(Offset { begin: 28, end: 30 }),
                Some(Offset { begin: 30, end: 31 }),
                Some(Offset { begin: 30, end: 31 }),
                Some(Offset { begin: 30, end: 31 }),
                Some(Offset { begin: 31, end: 32 }),
                Some(Offset { begin: 32, end: 34 }),
                Some(Offset { begin: 34, end: 35 }),
                Some(Offset { begin: 34, end: 35 }),
                Some(Offset { begin: 34, end: 35 }),
                Some(Offset { begin: 35, end: 37 }),
                None,
            ],
            reference_offsets: vec![],
            mask: vec![],
        },
    ]
    .to_vec();

    let output =
        deberta_tokenizer.encode_list(&original_strings, 128, &TruncationStrategy::LongestFirst, 0);

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
                        deberta_tokenizer.decode(&[predicted.token_ids[idx]], false, false),
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
