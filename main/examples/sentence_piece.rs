extern crate radix_trie;

use rust_tokenizers::preprocessing::tokenizer::sentence_piece_tokenizer::SentencePieceTokenizer;


fn main() {
    let model_path = "E:/Coding/notebooks/xlnet-base-cased-spiece.model";
    let sentence_piece_tokenizer = SentencePieceTokenizer::from_file(model_path, false);

//    let common_prefixes = sentence_piece_tokenizer.vocab().common_prefix_search("absolutely");
    sentence_piece_tokenizer.tokenize_to_pieces("\u{2581}🤔Supercalifragil🤔🤔isticexpialidocious🤔");
//    sentence_piece_tokenizer.tokenize_to_pieces("\u{2581}🤔");
//    println!("{:?}", output);
}
