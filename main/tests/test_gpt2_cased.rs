mod test_utils;

use rust_tokenizers::tokenizer::{Gpt2Tokenizer, Tokenizer, TruncationStrategy};
use rust_tokenizers::vocab::{BpePairVocab, Gpt2Vocab, Vocab};
use rust_tokenizers::{Offset, TokenizedInput};
use test_utils::download_file_to_cache;

#[test]
fn test_gpt2_tokenization() -> anyhow::Result<()> {
    let vocab_path = download_file_to_cache(
        "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json",
    )
    .unwrap();

    let merges_path = download_file_to_cache(
        "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt",
    )
    .unwrap();

    let vocab = Gpt2Vocab::from_file(vocab_path.as_path())?;
    let merges = BpePairVocab::from_file(merges_path.as_path())?;

    let gpt2_tokenizer = Gpt2Tokenizer::from_existing_vocab_and_merges(vocab, merges, false);

    let original_strings = [
        "…",
        "This is a sample sentence to be tokénized",
        "Wondering how this will get tokenized 🤔 ?",
        "İs th!s 𩸽 Ϻ Šœ Ugljšić dấu nặng",
        "İs th!s   𩸽 <|endoftext|> Ϻ Šœ  Uglj<|endoftext|>šić   dấu nặng",
        "   İs th!s    𩸽 Ϻ Šœ   Ugljšić  dấu nặng     ",
        "  �� İs th!s   ���� 𩸽 Ϻ Šœ   Ugljšić  dấu nặng     ",
    ];

    let expected_results = [
        TokenizedInput {
            token_ids: vec![1399],
            segment_ids: vec![0],
            special_tokens_mask: vec![0],
            overflowing_tokens: vec![],
            num_truncated_tokens: 0,
            token_offsets: vec![Some(Offset { begin: 0, end: 1 })],
            reference_offsets: vec![],
            mask: vec![],
        },
        TokenizedInput {
            token_ids: vec![
                1212, 318, 257, 6291, 6827, 284, 307, 284, 365, 136, 223, 77, 1143,
            ],
            segment_ids: vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            special_tokens_mask: vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
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
                Some(Offset { begin: 31, end: 34 }),
                Some(Offset { begin: 34, end: 36 }),
                Some(Offset { begin: 36, end: 37 }),
                Some(Offset { begin: 36, end: 37 }),
                Some(Offset { begin: 37, end: 38 }),
                Some(Offset { begin: 38, end: 42 }),
            ],
            reference_offsets: vec![],
            mask: vec![],
        },
        TokenizedInput {
            token_ids: vec![
                54, 623, 1586, 703, 428, 481, 651, 11241, 1143, 12520, 97, 242, 5633,
            ],
            segment_ids: vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            special_tokens_mask: vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            overflowing_tokens: vec![],
            num_truncated_tokens: 0,
            token_offsets: vec![
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
            ],
            reference_offsets: vec![],
            mask: vec![],
        },
        TokenizedInput {
            token_ids: vec![
                128, 108, 82, 294, 0, 82, 220, 172, 102, 116, 121, 18074, 118, 25370, 254, 129,
                241, 471, 4743, 73, 32790, 72, 38325, 288, 157, 118, 98, 84, 299, 157, 118, 115,
                782,
            ],
            segment_ids: vec![
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0,
            ],
            special_tokens_mask: vec![
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0,
            ],
            overflowing_tokens: vec![],
            num_truncated_tokens: 0,
            token_offsets: vec![
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
            ],
            reference_offsets: vec![],
            mask: vec![],
        },
        TokenizedInput {
            token_ids: vec![
                128, 108, 82, 294, 0, 82, 220, 220, 220, 172, 102, 116, 121, 50256, 18074, 118,
                25370, 254, 129, 241, 220, 471, 4743, 73, 50256, 32790, 72, 38325, 220, 220, 288,
                157, 118, 98, 84, 299, 157, 118, 115, 782,
            ],
            segment_ids: vec![
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ],
            special_tokens_mask: vec![
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ],
            overflowing_tokens: vec![],
            num_truncated_tokens: 0,
            token_offsets: vec![
                Some(Offset { begin: 0, end: 1 }),
                Some(Offset { begin: 0, end: 1 }),
                Some(Offset { begin: 1, end: 2 }),
                Some(Offset { begin: 2, end: 5 }),
                Some(Offset { begin: 5, end: 6 }),
                Some(Offset { begin: 6, end: 7 }),
                Some(Offset { begin: 7, end: 8 }),
                Some(Offset { begin: 8, end: 9 }),
                Some(Offset { begin: 9, end: 10 }),
                Some(Offset { begin: 10, end: 11 }),
                Some(Offset { begin: 10, end: 11 }),
                Some(Offset { begin: 10, end: 11 }),
                Some(Offset { begin: 10, end: 11 }),
                Some(Offset { begin: 12, end: 25 }),
                Some(Offset { begin: 25, end: 27 }),
                Some(Offset { begin: 26, end: 27 }),
                Some(Offset { begin: 27, end: 29 }),
                Some(Offset { begin: 28, end: 29 }),
                Some(Offset { begin: 29, end: 30 }),
                Some(Offset { begin: 29, end: 30 }),
                Some(Offset { begin: 30, end: 31 }),
                Some(Offset { begin: 31, end: 33 }),
                Some(Offset { begin: 33, end: 35 }),
                Some(Offset { begin: 35, end: 36 }),
                Some(Offset { begin: 36, end: 49 }),
                Some(Offset { begin: 49, end: 50 }),
                Some(Offset { begin: 50, end: 51 }),
                Some(Offset { begin: 51, end: 52 }),
                Some(Offset { begin: 52, end: 53 }),
                Some(Offset { begin: 53, end: 54 }),
                Some(Offset { begin: 54, end: 56 }),
                Some(Offset { begin: 56, end: 57 }),
                Some(Offset { begin: 56, end: 57 }),
                Some(Offset { begin: 56, end: 57 }),
                Some(Offset { begin: 57, end: 58 }),
                Some(Offset { begin: 58, end: 60 }),
                Some(Offset { begin: 60, end: 61 }),
                Some(Offset { begin: 60, end: 61 }),
                Some(Offset { begin: 60, end: 61 }),
                Some(Offset { begin: 61, end: 63 }),
            ],
            reference_offsets: vec![],
            mask: vec![],
        },
        TokenizedInput {
            token_ids: vec![
                220, 220, 34754, 108, 82, 294, 0, 82, 220, 220, 220, 220, 172, 102, 116, 121,
                18074, 118, 25370, 254, 129, 241, 220, 220, 471, 4743, 73, 32790, 72, 38325, 220,
                288, 157, 118, 98, 84, 299, 157, 118, 115, 782, 220, 220, 220, 220, 220,
            ],
            segment_ids: vec![
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ],
            special_tokens_mask: vec![
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ],
            overflowing_tokens: vec![],
            num_truncated_tokens: 0,
            token_offsets: vec![
                Some(Offset { begin: 0, end: 1 }),
                Some(Offset { begin: 1, end: 2 }),
                Some(Offset { begin: 2, end: 4 }),
                Some(Offset { begin: 3, end: 4 }),
                Some(Offset { begin: 4, end: 5 }),
                Some(Offset { begin: 5, end: 8 }),
                Some(Offset { begin: 8, end: 9 }),
                Some(Offset { begin: 9, end: 10 }),
                Some(Offset { begin: 10, end: 11 }),
                Some(Offset { begin: 11, end: 12 }),
                Some(Offset { begin: 12, end: 13 }),
                Some(Offset { begin: 13, end: 14 }),
                Some(Offset { begin: 14, end: 15 }),
                Some(Offset { begin: 14, end: 15 }),
                Some(Offset { begin: 14, end: 15 }),
                Some(Offset { begin: 14, end: 15 }),
                Some(Offset { begin: 15, end: 17 }),
                Some(Offset { begin: 16, end: 17 }),
                Some(Offset { begin: 17, end: 19 }),
                Some(Offset { begin: 18, end: 19 }),
                Some(Offset { begin: 19, end: 20 }),
                Some(Offset { begin: 19, end: 20 }),
                Some(Offset { begin: 20, end: 21 }),
                Some(Offset { begin: 21, end: 22 }),
                Some(Offset { begin: 22, end: 24 }),
                Some(Offset { begin: 24, end: 26 }),
                Some(Offset { begin: 26, end: 27 }),
                Some(Offset { begin: 27, end: 28 }),
                Some(Offset { begin: 28, end: 29 }),
                Some(Offset { begin: 29, end: 30 }),
                Some(Offset { begin: 30, end: 31 }),
                Some(Offset { begin: 31, end: 33 }),
                Some(Offset { begin: 33, end: 34 }),
                Some(Offset { begin: 33, end: 34 }),
                Some(Offset { begin: 33, end: 34 }),
                Some(Offset { begin: 34, end: 35 }),
                Some(Offset { begin: 35, end: 37 }),
                Some(Offset { begin: 37, end: 38 }),
                Some(Offset { begin: 37, end: 38 }),
                Some(Offset { begin: 37, end: 38 }),
                Some(Offset { begin: 38, end: 40 }),
                Some(Offset { begin: 40, end: 41 }),
                Some(Offset { begin: 41, end: 42 }),
                Some(Offset { begin: 42, end: 43 }),
                Some(Offset { begin: 43, end: 44 }),
                Some(Offset { begin: 44, end: 45 }),
            ],
            reference_offsets: vec![],
            mask: vec![],
        },
        TokenizedInput {
            token_ids: vec![
                220, 220, 6353, 34754, 108, 82, 294, 0, 82, 220, 220, 26825, 220, 172, 102, 116,
                121, 18074, 118, 25370, 254, 129, 241, 220, 220, 471, 4743, 73, 32790, 72, 38325,
                220, 288, 157, 118, 98, 84, 299, 157, 118, 115, 782, 220, 220, 220, 220, 220,
            ],
            segment_ids: vec![
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ],
            special_tokens_mask: vec![
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ],
            overflowing_tokens: vec![],
            num_truncated_tokens: 0,
            token_offsets: vec![
                Some(Offset { begin: 0, end: 1 }),
                Some(Offset { begin: 1, end: 2 }),
                Some(Offset { begin: 2, end: 4 }),
                Some(Offset { begin: 4, end: 6 }),
                Some(Offset { begin: 5, end: 6 }),
                Some(Offset { begin: 6, end: 7 }),
                Some(Offset { begin: 7, end: 10 }),
                Some(Offset { begin: 10, end: 11 }),
                Some(Offset { begin: 11, end: 12 }),
                Some(Offset { begin: 12, end: 13 }),
                Some(Offset { begin: 13, end: 14 }),
                Some(Offset { begin: 14, end: 19 }),
                Some(Offset { begin: 19, end: 20 }),
                Some(Offset { begin: 20, end: 21 }),
                Some(Offset { begin: 20, end: 21 }),
                Some(Offset { begin: 20, end: 21 }),
                Some(Offset { begin: 20, end: 21 }),
                Some(Offset { begin: 21, end: 23 }),
                Some(Offset { begin: 22, end: 23 }),
                Some(Offset { begin: 23, end: 25 }),
                Some(Offset { begin: 24, end: 25 }),
                Some(Offset { begin: 25, end: 26 }),
                Some(Offset { begin: 25, end: 26 }),
                Some(Offset { begin: 26, end: 27 }),
                Some(Offset { begin: 27, end: 28 }),
                Some(Offset { begin: 28, end: 30 }),
                Some(Offset { begin: 30, end: 32 }),
                Some(Offset { begin: 32, end: 33 }),
                Some(Offset { begin: 33, end: 34 }),
                Some(Offset { begin: 34, end: 35 }),
                Some(Offset { begin: 35, end: 36 }),
                Some(Offset { begin: 36, end: 37 }),
                Some(Offset { begin: 37, end: 39 }),
                Some(Offset { begin: 39, end: 40 }),
                Some(Offset { begin: 39, end: 40 }),
                Some(Offset { begin: 39, end: 40 }),
                Some(Offset { begin: 40, end: 41 }),
                Some(Offset { begin: 41, end: 43 }),
                Some(Offset { begin: 43, end: 44 }),
                Some(Offset { begin: 43, end: 44 }),
                Some(Offset { begin: 43, end: 44 }),
                Some(Offset { begin: 44, end: 46 }),
                Some(Offset { begin: 46, end: 47 }),
                Some(Offset { begin: 47, end: 48 }),
                Some(Offset { begin: 48, end: 49 }),
                Some(Offset { begin: 49, end: 50 }),
                Some(Offset { begin: 50, end: 51 }),
            ],
            reference_offsets: vec![],
            mask: vec![],
        },
    ]
    .to_vec();

    let output =
        gpt2_tokenizer.encode_list(&original_strings, 128, &TruncationStrategy::LongestFirst, 0);

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
                        gpt2_tokenizer.decode(&[predicted.token_ids[idx]], false, false),
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
