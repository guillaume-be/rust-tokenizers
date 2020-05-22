extern crate radix_trie;

use rust_tokenizers::preprocessing::tokenizer::sentence_piece_tokenizer::SentencePieceTokenizer;
use std::time::Instant;

fn main() {
    let model_path = "E:/Coding/notebooks/xlnet-base-cased-spiece.model";
    let sentence_piece_tokenizer = SentencePieceTokenizer::from_file(model_path, false);

    let now = Instant::now();
    let text = "\u{2581}One of the most famous people born in Warsaw was Maria Skłodowska-Curie, who achieved international recognition for her research on radioactivity and was the first female recipient of the Nobel Prize. Famous musicians include Władysław Szpilman and Frédéric Chopin. Though Chopin was born in the village of Żelazowa Wola, about 60 km (37 mi) from Warsaw, he moved to the city with his family when he was seven months old. Casimir Pulaski, a Polish general and hero of the American Revolutionary War, was born here in 1745.";
//    let text = "\u{2581}Supercalifragilisticexpialidocious";
    sentence_piece_tokenizer.tokenize_to_pieces(text);

    println!("{:?}", now.elapsed());

}
