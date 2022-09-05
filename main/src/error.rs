//! # Tokenizer error variants

#[derive(Debug, snafu::Snafu)]
#[snafu(visibility(pub(crate)))]
pub enum TokenizerError {
    #[snafu(display("{location}: failed to perform IO operation with {}: {source}", path.display()))]
    IO {
        source: std::io::Error,
        path: std::path::PathBuf,
        location: snafu::Location,
    },

    #[snafu(display("{location}: failed to deserialize JSON: {source}"))]
    JsonDeserialize {
        source: serde_json::Error,
        location: snafu::Location,
    },

    #[snafu(display("{location}: failed to deserialize protobuf: {source}"))]
    ProtobufDeserialize {
        source: protobuf::ProtobufError,
        location: snafu::Location,
    },

    #[snafu(display("{location}: {message}"))]
    VocabularyValidation {
        message: String,
        location: snafu::Location,
    },

    #[snafu(display("{location}: index not found: {index}"))]
    IndexNotFound {
        index: usize,
        location: snafu::Location,
    },

    #[snafu(display("{location}: token not found: {token}"))]
    TokenNotFound {
        token: String,
        location: snafu::Location,
    },

    #[snafu(display("{location}: value error: {message}"))]
    Value {
        message: String,
        location: snafu::Location,
    },

    #[snafu(display("{location}: CSV deserialization error: {source}"))]
    CSVDeserialize {
        source: csv::Error,
        location: snafu::Location,
    },
}
