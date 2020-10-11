//! # Tokenizer error variants
use thiserror::Error;

#[derive(Error, Debug)]
pub enum TokenizerError {
    #[error("File not found error: {0}")]
    FileNotFound(String),

    #[error("Error when loading vocabulary file, the file may be corrupted or does not match the expected format: {0}")]
    VocabularyParsingError(String),

    #[error("Token index not found in vocabulary: {0}")]
    IndexNotFound(String),

    #[error("Token not found in vocabulary: {0}")]
    TokenNotFound(String),

    #[error("Tokenization error: {0}")]
    TokenizationError(String),

    #[error("Value error: {0}")]
    ValueError(String),

    #[error("IO error: {0}")]
    IOError(String),
}

impl From<csv::Error> for TokenizerError {
    fn from(error: csv::Error) -> Self {
        TokenizerError::IOError(error.to_string())
    }
}
