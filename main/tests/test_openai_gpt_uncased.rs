mod test_utils;

use rust_tokenizers::tokenizer::{OpenAiGptTokenizer, Tokenizer, TruncationStrategy};
use rust_tokenizers::{Offset, TokenizedInput};
use test_utils::download_file_to_cache;

#[test]
fn test_openai_gpt_tokenization() -> anyhow::Result<()> {
    let vocab_path = download_file_to_cache(
        "https://s3.amazonaws.com/models.huggingface.co/bert/openai-gpt-vocab.json",
        "openai-gpt_vocab.json",
    )
    .unwrap();

    let merges_path = download_file_to_cache(
        "https://s3.amazonaws.com/models.huggingface.co/bert/openai-gpt-merges.txt",
        "openai-gpt_merges.txt",
    )
    .unwrap();

    let openai_gpt_tokenizer = OpenAiGptTokenizer::from_file(
        vocab_path.to_str().unwrap(),
        merges_path.to_str().unwrap(),
        true,
    )?;

    let original_strings = [
        "This is a sample sentence to be tokénized",
        "Wondering how this will get tokenized 🤔 ?",
        "İs th!s 𩸽 Ϻ Šœ Ugljšić dấu nặng",
        "İs th!s   𩸽 <unk> Ϻ Šœ  Uglj<unk>šić   dấu nặng",
        "   İs th!s    𩸽 Ϻ Šœ   Ugljšić  dấu nặng     ",
        "  �� İs th!s   ���� 𩸽 Ϻ Šœ   Ugljšić  dấu nặng     ",
    ];

    let expected_results = [
        TokenizedInput {
            token_ids: vec![616, 544, 246, 12273, 5958, 485, 580, 571, 2987, 4780],
            segment_ids: vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            special_tokens_mask: vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            overflowing_tokens: vec![],
            num_truncated_tokens: 0,
            token_offsets: vec![
                Some(Offset { begin: 0, end: 4 }),
                Some(Offset { begin: 5, end: 7 }),
                Some(Offset { begin: 8, end: 9 }),
                Some(Offset { begin: 10, end: 16 }),
                Some(Offset { begin: 17, end: 25 }),
                Some(Offset { begin: 26, end: 28 }),
                Some(Offset { begin: 29, end: 31 }),
                Some(Offset { begin: 32, end: 34 }),
                Some(Offset { begin: 34, end: 38 }),
                Some(Offset { begin: 38, end: 42 }),
            ],
            reference_offsets: vec![],
            mask: vec![],
        },
        TokenizedInput {
            token_ids: vec![3039, 718, 616, 812, 727, 571, 2987, 4780, 0, 257],
            segment_ids: vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            special_tokens_mask: vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            overflowing_tokens: vec![],
            num_truncated_tokens: 0,
            token_offsets: vec![
                Some(Offset { begin: 0, end: 9 }),
                Some(Offset { begin: 10, end: 13 }),
                Some(Offset { begin: 14, end: 18 }),
                Some(Offset { begin: 19, end: 23 }),
                Some(Offset { begin: 24, end: 27 }),
                Some(Offset { begin: 28, end: 30 }),
                Some(Offset { begin: 30, end: 33 }),
                Some(Offset { begin: 33, end: 37 }),
                Some(Offset { begin: 38, end: 39 }),
                Some(Offset { begin: 40, end: 41 }),
            ],
            reference_offsets: vec![],
            mask: vec![],
        },
        TokenizedInput {
            token_ids: vec![
                544, 663, 267, 252, 0, 0, 14, 411, 16, 2041, 28, 2519, 643, 254, 10591, 268,
            ],
            segment_ids: vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            special_tokens_mask: vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            overflowing_tokens: vec![],
            num_truncated_tokens: 0,
            token_offsets: vec![
                Some(Offset { begin: 0, end: 2 }),
                Some(Offset { begin: 3, end: 5 }),
                Some(Offset { begin: 5, end: 6 }),
                Some(Offset { begin: 6, end: 7 }),
                Some(Offset { begin: 8, end: 9 }),
                Some(Offset { begin: 10, end: 11 }),
                Some(Offset { begin: 12, end: 13 }),
                Some(Offset { begin: 13, end: 14 }),
                Some(Offset { begin: 15, end: 16 }),
                Some(Offset { begin: 16, end: 18 }),
                Some(Offset { begin: 18, end: 19 }),
                Some(Offset { begin: 19, end: 22 }),
                Some(Offset { begin: 23, end: 25 }),
                Some(Offset { begin: 25, end: 26 }),
                Some(Offset { begin: 27, end: 30 }),
                Some(Offset { begin: 30, end: 31 }),
            ],
            reference_offsets: vec![],
            mask: vec![],
        },
        TokenizedInput {
            token_ids: vec![
                544, 663, 267, 252, 0, 0, 0, 14, 411, 16, 2041, 266, 0, 2519, 643, 254, 10591, 268,
            ],
            segment_ids: vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            special_tokens_mask: vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            overflowing_tokens: vec![],
            num_truncated_tokens: 0,
            token_offsets: vec![
                Some(Offset { begin: 0, end: 2 }),
                Some(Offset { begin: 3, end: 5 }),
                Some(Offset { begin: 5, end: 6 }),
                Some(Offset { begin: 6, end: 7 }),
                Some(Offset { begin: 10, end: 11 }),
                Some(Offset { begin: 12, end: 17 }),
                Some(Offset { begin: 18, end: 19 }),
                Some(Offset { begin: 20, end: 21 }),
                Some(Offset { begin: 21, end: 22 }),
                Some(Offset { begin: 24, end: 25 }),
                Some(Offset { begin: 25, end: 27 }),
                Some(Offset { begin: 27, end: 28 }),
                Some(Offset { begin: 28, end: 33 }),
                Some(Offset { begin: 33, end: 36 }),
                Some(Offset { begin: 39, end: 41 }),
                Some(Offset { begin: 41, end: 42 }),
                Some(Offset { begin: 43, end: 46 }),
                Some(Offset { begin: 46, end: 47 }),
            ],
            reference_offsets: vec![],
            mask: vec![],
        },
        TokenizedInput {
            token_ids: vec![
                544, 663, 267, 252, 0, 0, 14, 411, 16, 2041, 28, 2519, 643, 254, 10591, 268,
            ],
            segment_ids: vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            special_tokens_mask: vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            overflowing_tokens: vec![],
            num_truncated_tokens: 0,
            token_offsets: vec![
                Some(Offset { begin: 3, end: 5 }),
                Some(Offset { begin: 6, end: 8 }),
                Some(Offset { begin: 8, end: 9 }),
                Some(Offset { begin: 9, end: 10 }),
                Some(Offset { begin: 14, end: 15 }),
                Some(Offset { begin: 16, end: 17 }),
                Some(Offset { begin: 18, end: 19 }),
                Some(Offset { begin: 19, end: 20 }),
                Some(Offset { begin: 23, end: 24 }),
                Some(Offset { begin: 24, end: 26 }),
                Some(Offset { begin: 26, end: 27 }),
                Some(Offset { begin: 27, end: 30 }),
                Some(Offset { begin: 32, end: 34 }),
                Some(Offset { begin: 34, end: 35 }),
                Some(Offset { begin: 36, end: 39 }),
                Some(Offset { begin: 39, end: 40 }),
            ],
            reference_offsets: vec![],
            mask: vec![],
        },
        TokenizedInput {
            token_ids: vec![
                120, 358, 544, 663, 267, 252, 120, 120, 120, 358, 0, 0, 14, 411, 16, 2041, 28,
                2519, 643, 254, 10591, 268,
            ],
            segment_ids: vec![
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ],
            special_tokens_mask: vec![
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ],
            overflowing_tokens: vec![],
            num_truncated_tokens: 0,
            token_offsets: vec![
                Some(Offset { begin: 2, end: 3 }),
                Some(Offset { begin: 3, end: 4 }),
                Some(Offset { begin: 5, end: 7 }),
                Some(Offset { begin: 8, end: 10 }),
                Some(Offset { begin: 10, end: 11 }),
                Some(Offset { begin: 11, end: 12 }),
                Some(Offset { begin: 15, end: 16 }),
                Some(Offset { begin: 16, end: 17 }),
                Some(Offset { begin: 17, end: 18 }),
                Some(Offset { begin: 18, end: 19 }),
                Some(Offset { begin: 20, end: 21 }),
                Some(Offset { begin: 22, end: 23 }),
                Some(Offset { begin: 24, end: 25 }),
                Some(Offset { begin: 25, end: 26 }),
                Some(Offset { begin: 29, end: 30 }),
                Some(Offset { begin: 30, end: 32 }),
                Some(Offset { begin: 32, end: 33 }),
                Some(Offset { begin: 33, end: 36 }),
                Some(Offset { begin: 38, end: 40 }),
                Some(Offset { begin: 40, end: 41 }),
                Some(Offset { begin: 42, end: 45 }),
                Some(Offset { begin: 45, end: 46 }),
            ],
            reference_offsets: vec![],
            mask: vec![],
        },
    ]
    .to_vec();

    let output = openai_gpt_tokenizer.encode_list(
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
                        openai_gpt_tokenizer.decode(vec!(predicted.token_ids[idx]), false, false),
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
