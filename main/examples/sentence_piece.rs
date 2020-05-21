
extern crate radix_trie;


use rust_tokenizers::preprocessing::vocab::sentence_piece_vocab::SentencePieceVocab;


fn main() {

    let model_path = "E:/Coding/notebooks/xlnet-base-cased-spiece.model";
    let spiece_vocab = SentencePieceVocab::from_file(model_path);

    let common_prefixes = spiece_vocab.common_prefix_search("absolutely");
    println!("{:?}", common_prefixes);
}
