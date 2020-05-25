use rust_tokenizers::preprocessing::tokenizer::marian_tokenizer::MarianTokenizer;
use rust_tokenizers::{Tokenizer, TruncationStrategy};

fn main() {
    let vocab_file = "E:/Coding/cache/tokenizers/marian_vocab.json";
    let model_file = "E:/Coding/cache/tokenizers/en.spm";

    let text = "the";
    let tokenizer = MarianTokenizer::from_files(vocab_file, model_file, false);

    let output = tokenizer.encode(text, None, 128, &TruncationStrategy::LongestFirst, 0);

    println!("{:?}", output.token_ids);
}