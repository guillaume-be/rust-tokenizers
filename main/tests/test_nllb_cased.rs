mod test_utils;

use rust_tokenizers::tokenizer::{
    MultiThreadedTokenizer, NLLBTokenizer, Tokenizer, TruncationStrategy,
};
use rust_tokenizers::{Offset, TokenizedInput};
use test_utils::download_file_to_cache;

#[test]
fn test_nllb_tokenization() -> anyhow::Result<()> {
    let vocab_path = download_file_to_cache(
        "https://huggingface.co/facebook/nllb-200-distilled-600M/resolve/main/tokenizer.json",
        "nllb_tokenizer.json",
    )
    .unwrap();

    let merges_path = download_file_to_cache(
        "https://huggingface.co/facebook/nllb-200-distilled-600M/resolve/main/sentencepiece.bpe.model",
        "nllb_sentencepiece.model",
    )
    .unwrap();

    let special_path = download_file_to_cache(
        "htps://huggingface.co/facebook/nllb-200-distilled-600M/raw/main/special_tokens_map.json",
        "nllb_special.json",
    )
    .unwrap();

    let nllb_tokenizer = NLLBTokenizer::from_files(vocab_path, merges_path, special_path)?;

    let original_strings = [
        "nld_Latn …",
        "eng_Latn This is a sample sentence to be tokénized",
        "eng_Latn Wondering how this will get tokenized 🤔 ?",
        "fra_Latn İs th!s 𩸽 Ϻ Šœ Ugljšić dấu nặng",
        "hin_Deva   İs th!s    𩸽 Ϻ Šœ   Ugljšić  dấu nặng     ",
        "lit_Latn � İs th!s �� 𩸽 Ϻ Šœ   Ugljšić  dấu nặng     ",
    ];

    let expected_results = [
        TokenizedInput {
            token_ids: vec![256127, 622, 2],
            segment_ids: vec![],
            special_tokens_mask: vec![],
            overflowing_tokens: vec![],
            num_truncated_tokens: 0,
            token_offsets: vec![None, Some(Offset { begin: 8, end: 10 }), None],
            reference_offsets: vec![],
            mask: vec![],
        },
        TokenizedInput {
            token_ids: vec![
                256047, 9680, 248, 9, 183824, 109267, 202, 280, 1776, 107717, 18755, 2,
            ],
            segment_ids: vec![],
            special_tokens_mask: vec![],
            overflowing_tokens: vec![],
            num_truncated_tokens: 0,
            token_offsets: vec![
                None,
                Some(Offset { begin: 8, end: 13 }),
                Some(Offset { begin: 13, end: 16 }),
                Some(Offset { begin: 16, end: 18 }),
                Some(Offset { begin: 18, end: 25 }),
                Some(Offset { begin: 25, end: 34 }),
                Some(Offset { begin: 34, end: 37 }),
                Some(Offset { begin: 37, end: 40 }),
                Some(Offset { begin: 40, end: 44 }),
                Some(Offset { begin: 44, end: 48 }),
                Some(Offset { begin: 48, end: 51 }),
                None,
            ],
            reference_offsets: vec![],
            mask: vec![],
        },
        TokenizedInput {
            token_ids: vec![
                256047, 246, 527, 5334, 11657, 3423, 4062, 4023, 1776, 430, 18755, 248059, 254282,
                385, 2,
            ],
            segment_ids: vec![],
            special_tokens_mask: vec![],
            overflowing_tokens: vec![],
            num_truncated_tokens: 0,
            token_offsets: vec![
                None,
                Some(Offset { begin: 8, end: 10 }),
                Some(Offset { begin: 10, end: 13 }),
                Some(Offset { begin: 13, end: 18 }),
                Some(Offset { begin: 18, end: 22 }),
                Some(Offset { begin: 22, end: 27 }),
                Some(Offset { begin: 27, end: 32 }),
                Some(Offset { begin: 32, end: 36 }),
                Some(Offset { begin: 36, end: 40 }),
                Some(Offset { begin: 40, end: 43 }),
                Some(Offset { begin: 43, end: 46 }),
                Some(Offset { begin: 46, end: 47 }),
                Some(Offset { begin: 47, end: 48 }),
                Some(Offset { begin: 48, end: 50 }),
                None,
            ],
            reference_offsets: vec![],
            mask: vec![],
        },
        TokenizedInput {
            token_ids: vec![
                256057, 73808, 233, 248203, 248066, 248059, 3, 248059, 3, 3658, 250500, 12291,
                10117, 2139, 248701, 69085, 100655, 2,
            ],
            segment_ids: vec![],
            special_tokens_mask: vec![],
            overflowing_tokens: vec![],
            num_truncated_tokens: 0,
            token_offsets: vec![
                None,
                Some(Offset { begin: 8, end: 11 }),
                Some(Offset { begin: 11, end: 14 }),
                Some(Offset { begin: 14, end: 15 }),
                Some(Offset { begin: 15, end: 16 }),
                Some(Offset { begin: 16, end: 17 }),
                Some(Offset { begin: 17, end: 18 }),
                Some(Offset { begin: 18, end: 19 }),
                Some(Offset { begin: 19, end: 20 }),
                Some(Offset { begin: 20, end: 22 }),
                Some(Offset { begin: 22, end: 23 }),
                Some(Offset { begin: 23, end: 26 }),
                Some(Offset { begin: 26, end: 28 }),
                Some(Offset { begin: 28, end: 30 }),
                Some(Offset { begin: 30, end: 31 }),
                Some(Offset { begin: 31, end: 35 }),
                Some(Offset { begin: 35, end: 40 }),
                None,
            ],
            reference_offsets: vec![],
            mask: vec![],
        },
        TokenizedInput {
            token_ids: vec![
                256068, 73808, 233, 248203, 248066, 248059, 248059, 248059, 248059, 3, 248059, 3,
                3658, 250500, 248059, 248059, 12291, 10117, 2139, 248701, 248059, 69085, 100655,
                248059, 248059, 248059, 248059, 248059, 2,
            ],
            segment_ids: vec![],
            special_tokens_mask: vec![],
            overflowing_tokens: vec![],
            num_truncated_tokens: 0,
            token_offsets: vec![
                None,
                Some(Offset { begin: 10, end: 13 }),
                Some(Offset { begin: 13, end: 16 }),
                Some(Offset { begin: 16, end: 17 }),
                Some(Offset { begin: 17, end: 18 }),
                Some(Offset { begin: 18, end: 19 }),
                Some(Offset { begin: 19, end: 20 }),
                Some(Offset { begin: 20, end: 21 }),
                Some(Offset { begin: 21, end: 22 }),
                Some(Offset { begin: 22, end: 23 }),
                Some(Offset { begin: 23, end: 24 }),
                Some(Offset { begin: 24, end: 25 }),
                Some(Offset { begin: 25, end: 27 }),
                Some(Offset { begin: 27, end: 28 }),
                Some(Offset { begin: 28, end: 29 }),
                Some(Offset { begin: 29, end: 30 }),
                Some(Offset { begin: 30, end: 33 }),
                Some(Offset { begin: 33, end: 35 }),
                Some(Offset { begin: 35, end: 37 }),
                Some(Offset { begin: 37, end: 38 }),
                Some(Offset { begin: 38, end: 39 }),
                Some(Offset { begin: 39, end: 43 }),
                Some(Offset { begin: 43, end: 48 }),
                Some(Offset { begin: 48, end: 49 }),
                Some(Offset { begin: 49, end: 50 }),
                Some(Offset { begin: 50, end: 51 }),
                Some(Offset { begin: 51, end: 52 }),
                Some(Offset { begin: 52, end: 53 }),
                None,
            ],
            reference_offsets: vec![],
            mask: vec![],
        },
        TokenizedInput {
            token_ids: vec![
                256105, 248059, 73808, 233, 248203, 248066, 248059, 248059, 3, 248059, 3, 3658,
                250500, 248059, 248059, 12291, 10117, 2139, 248701, 248059, 69085, 100655, 248059,
                248059, 248059, 248059, 248059, 2,
            ],
            segment_ids: vec![],
            special_tokens_mask: vec![],
            overflowing_tokens: vec![],
            num_truncated_tokens: 0,
            token_offsets: vec![
                None,
                Some(Offset { begin: 8, end: 9 }),
                Some(Offset { begin: 10, end: 13 }),
                Some(Offset { begin: 13, end: 16 }),
                Some(Offset { begin: 16, end: 17 }),
                Some(Offset { begin: 17, end: 18 }),
                Some(Offset { begin: 18, end: 19 }),
                Some(Offset { begin: 21, end: 22 }),
                Some(Offset { begin: 22, end: 23 }),
                Some(Offset { begin: 23, end: 24 }),
                Some(Offset { begin: 24, end: 25 }),
                Some(Offset { begin: 25, end: 27 }),
                Some(Offset { begin: 27, end: 28 }),
                Some(Offset { begin: 28, end: 29 }),
                Some(Offset { begin: 29, end: 30 }),
                Some(Offset { begin: 30, end: 33 }),
                Some(Offset { begin: 33, end: 35 }),
                Some(Offset { begin: 35, end: 37 }),
                Some(Offset { begin: 37, end: 38 }),
                Some(Offset { begin: 38, end: 39 }),
                Some(Offset { begin: 39, end: 43 }),
                Some(Offset { begin: 43, end: 48 }),
                Some(Offset { begin: 48, end: 49 }),
                Some(Offset { begin: 49, end: 50 }),
                Some(Offset { begin: 50, end: 51 }),
                Some(Offset { begin: 51, end: 52 }),
                Some(Offset { begin: 52, end: 53 }),
                None,
            ],
            reference_offsets: vec![],
            mask: vec![],
        },
    ]
    .to_vec();

    let output = MultiThreadedTokenizer::encode_list(
        &nllb_tokenizer,
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
                        nllb_tokenizer.decode(&[predicted.token_ids[idx]], false, false),
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
