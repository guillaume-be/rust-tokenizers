use radix_trie::Trie;
use crate::preprocessing::vocab::sentencepiece_proto::sentencepiece_model::ModelProto;
use protobuf::parse_from_bytes;
use std::fs::File;
use std::io::Read;

pub struct SentencePieceVocab {
    trie: Trie<String, (f32, i64)>
}

impl SentencePieceVocab {
    pub fn from_file(path: &str) -> SentencePieceVocab {
        let mut f = File::open(path).unwrap();
        let mut contents = Vec::new();
        f.read_to_end(&mut contents).unwrap();

        let proto = parse_from_bytes::<ModelProto>(contents.as_slice()).unwrap();
        SentencePieceVocab::from_proto(&proto)
    }

    pub fn from_proto(proto: &ModelProto) -> SentencePieceVocab {
        let mut trie = Trie::new();
        for (idx, piece) in proto.get_pieces().iter().enumerate() {
            trie.insert(piece.get_piece().to_owned(), (piece.get_score(), idx as i64));
        }
        SentencePieceVocab { trie }
    }

    pub fn common_prefix_search<'a>(&'a self, text: &'a str) -> Vec<(&'a str, f32, i64)> {
        let mut results = vec!();
        for i in 1..text.len() + 1 {
            if let Some(sub_trie) = self.trie.get(&text[..i]) {
                results.push((&text[..i], sub_trie.0, sub_trie.1));
            };
        }
        results
    }
}