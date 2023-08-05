use crate::vocab::base_vocab::AddedToken;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

/// # Utility to deserialize JSON config files
pub trait Config
where
    for<'de> Self: Deserialize<'de>,
{
    /// Loads a `Config` object from a JSON file.
    /// The parsing will fail if non-optional keys expected by the model are missing.
    ///
    /// # Arguments
    ///
    /// * `path` - `Path` to the configuration JSON file.
    ///
    /// ```
    fn from_file<P: AsRef<Path>>(path: P) -> Self {
        let f = File::open(path).expect("Could not open configuration file.");
        let br = BufReader::new(f);
        let config: Self = serde_json::from_reader(br).expect("could not parse configuration");
        config
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
/// # Tokenizer Model configuration (tokenizer.json)
pub struct TokenizerConfig {
    added_tokens: Vec<AddedToken>,
}

impl Config for TokenizerConfig {}
