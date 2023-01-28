#[cfg(feature = "proto-compile")]
use std::fs;

fn main() {
    #[cfg(feature = "proto-compile")]
    {
        let out_path = "src/vocab/sentencepiece_proto";
        let out_file_name = "src/vocab/sentencepiece_proto/sentencepiece_model.proto";
        let proto_path = "protos/sentencepiece_model.proto";

        let metadata = fs::metadata(out_file_name);

        if metadata.is_err() {
            protobuf_codegen_pure::Codegen::new()
                .out_dir(out_path)
                .inputs([proto_path])
                .include("protos")
                .run()
                .expect("Codegen failed.");
        }
    }
}
