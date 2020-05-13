use protobuf::parse_from_bytes;
use rust_tokenizers::preprocessing::vocab::sentencepiece_proto::sentencepiece_model::ModelProto;


fn main() {

    let _contents = include_bytes!("E:/Coding/notebooks/toy.model");
    let sentencepiece_model = parse_from_bytes::<ModelProto>(_contents).unwrap();

    println!("{:?}", sentencepiece_model.get_pieces());

}
