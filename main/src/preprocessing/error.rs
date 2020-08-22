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
}