use std::collections::{HashMap, HashSet};

use serde::Deserialize;

use crate::error::TokenizerError;

use super::{base_vocab::swap_key_values, Vocab};

pub const FAIRSEQ_LANGUAGE_CODES: [&str; 202] = [
    "ace", "ace", "acm", "acq", "aeb", "afr", "ajp", "aka", "amh", "apc", "arb", "ars", "ary",
    "arz", "asm", "ast", "awa", "ayr", "azb", "azj", "bak", "bam", "ban", "bel", "bem", "ben",
    "bho", "bjn", "bjn", "bod", "bos", "bug", "bul", "cat", "ceb", "ces", "cjk", "ckb", "crh",
    "cym", "dan", "deu", "dik", "dyu", "dzo", "ell", "eng", "epo", "est", "eus", "ewe", "fao",
    "pes", "fij", "fin", "fon", "fra", "fur", "fuv", "gla", "gle", "glg", "grn", "guj", "hat",
    "hau", "heb", "hin", "hne", "hrv", "hun", "hye", "ibo", "ilo", "ind", "isl", "ita", "jav",
    "jpn", "kab", "kac", "kam", "kan", "kas", "kas", "kat", "knc", "knc", "kaz", "kbp", "kea",
    "khm", "kik", "kin", "kir", "kmb", "kon", "kor", "kmr", "lao", "lvs", "lij", "lim", "lin",
    "lit", "lmo", "ltg", "ltz", "lua", "lug", "luo", "lus", "mag", "mai", "mal", "mar", "min",
    "mkd", "plt", "mlt", "mni", "khk", "mos", "mri", "zsm", "mya", "nld", "nno", "nob", "npi",
    "nso", "nus", "nya", "oci", "gaz", "ory", "pag", "pan", "pap", "pol", "por", "prs", "pbt",
    "quy", "ron", "run", "rus", "sag", "san", "sat", "scn", "shn", "sin", "slk", "slv", "smo",
    "sna", "snd", "som", "sot", "spa", "als", "srd", "srp", "ssw", "sun", "swe", "swh", "szl",
    "tam", "tat", "tel", "tgk", "tgl", "tha", "tir", "taq", "taq", "tpi", "tsn", "tso", "tuk",
    "tum", "tur", "twi", "tzm", "uig", "ukr", "umb", "urd", "uzn", "vec", "vie", "war", "wol",
    "xho", "ydd", "yor", "yue", "zho", "zho", "zul",
];

pub struct NLLBVocab {
    /// A mapping of tokens as string to indices (i.e. the encoder base)
    pub values: HashMap<String, i64>,

    /// A mapping of token IDs to strings (i.e. the decoder base)
    pub indices: HashMap<i64, String>,

    /// A mapping of special value tokens as strings to IDs (i.e. the encoder base for special
    /// values), special values typically include things like BOS/EOS markers, class markers, mask
    /// markers and padding markers
    pub special_values: HashMap<String, i64>,

    /// A mapping of special value tokens as IDs to strings (i.e. the decoder base for special values)
    pub special_indices: HashMap<i64, String>,

    /// Language code stored as bytes for extraction of the prefix in input sequences
    pub language_codes_bytes: HashSet<Vec<u8>>,
}

impl NLLBVocab {
    /// The beginning of sequence token that was used during pretraining.
    /// Can be used a sequence classifier token.
    pub fn bos_value() -> &'static str {
        "<s>"
    }

    /// End of sequence token.
    pub fn eos_value() -> &'static str {
        "</s>"
    }

    /// Returns the SEP token for M2M100 (`</s>`)
    pub fn sep_value() -> &'static str {
        "</s>"
    }

    /// Returns the PAD token for M2M100 (`<pad>`)
    pub fn pad_value() -> &'static str {
        "<pad>"
    }
}

impl Vocab for NLLBVocab {
    fn unknown_value() -> &'static str {
        "<unk>"
    }

    fn get_unknown_value(&self) -> &'static str {
        "<unk>"
    }

    fn values(&self) -> &HashMap<String, i64> {
        &self.values
    }

    fn indices(&self) -> &HashMap<i64, String> {
        &self.indices
    }

    fn special_values(&self) -> &HashMap<String, i64> {
        &self.special_values
    }

    fn special_indices(&self) -> &HashMap<i64, String> {
        &self.special_indices
    }

    fn from_file(path: &str) -> Result<Self, TokenizerError> {
        let reader = std::fs::File::open(path).map_err(|e| {
            TokenizerError::FileNotFound(format!("{} vocabulary file not found :{}", path, e))
        })?;

        let reader = std::io::BufReader::new(reader);
        let mut tokenizer: Tokenizer = serde_json::from_reader(reader)
            .map_err(|e| TokenizerError::VocabularyParsingError(e.to_string()))?;

        let mut special_values = HashMap::with_capacity(FAIRSEQ_LANGUAGE_CODES.len() + 10);

        let values = &mut tokenizer.model.vocab;
        for language_code in FAIRSEQ_LANGUAGE_CODES.iter() {
            let language_code = if language_code.len() == 3 {
                format!(">>{language_code}<<")
            } else {
                return Err(TokenizerError::VocabularyParsingError(
                    "NLLB Vocab only supports language code of length 8".to_string(),
                ));
            };
            values.insert(language_code.clone(), values.len() as i64);
            NLLBVocab::_register_as_special_value(
                language_code.as_str(),
                values,
                &mut special_values,
            )?;
        }

        // TODO: remove it (it's already contained in `added_tokens`):
        let vocab = &tokenizer.model.vocab;
        Self::_register_as_special_value(Self::unknown_value(), vocab, &mut special_values)?;
        Self::_register_as_special_value(Self::sep_value(), vocab, &mut special_values)?;
        Self::_register_as_special_value(Self::bos_value(), vocab, &mut special_values)?;
        Self::_register_as_special_value(Self::eos_value(), vocab, &mut special_values)?;
        Self::_register_as_special_value(Self::pad_value(), vocab, &mut special_values)?;

        let indices = swap_key_values(vocab);
        let special_indices = swap_key_values(&special_values);
        let language_codes_bytes = FAIRSEQ_LANGUAGE_CODES
            .iter()
            .map(|f| format!(">>{f}<<").as_bytes().to_vec())
            .collect::<HashSet<Vec<u8>>>();

        Ok(Self {
            indices,
            language_codes_bytes,
            special_indices,
            special_values,
            values: vocab.clone(),
        })
    }

    fn token_to_id(&self, token: &str) -> i64 {
        self._token_to_id(
            token,
            self.values(),
            self.special_values(),
            self.get_unknown_value(),
        )
    }

    fn id_to_token(&self, id: &i64) -> String {
        self._id_to_token(
            id,
            self.indices(),
            self.special_indices(),
            self.get_unknown_value(),
        )
    }
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct Token {
    id: i64,
    content: String,
    single_word: bool,
    lstrip: bool,
    rstrip: bool,
    normalized: bool,
    special: bool,
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct Model {
    #[serde(rename = "type")]
    model_type: String,

    vocab: HashMap<String, i64>,
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct Tokenizer {
    version: String,
    truncation: Option<usize>,
    padding: Option<usize>,
    added_tokens: Vec<Token>,
    model: Model,
}
