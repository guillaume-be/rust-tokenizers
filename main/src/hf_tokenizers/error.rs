use crate::error::TokenizerError;

impl From<tokenizers::tokenizer::Error> for TokenizerError {
    fn from(error: tokenizers::tokenizer::Error) -> Self {
        TokenizerError::TokenizationError(error.to_string())
    }
}
