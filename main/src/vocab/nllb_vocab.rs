use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::BufReader,
    path::Path,
};

use serde::Deserialize;
use snafu::ResultExt;

use crate::error::*;

use super::{
    base_vocab::{swap_key_values, SpecialTokens},
    Vocab,
};

pub const FAIRSEQ_LANGUAGE_CODES: [&str; 202] = [
    "ace_Arab", "ace_Latn", "acm_Arab", "acq_Arab", "aeb_Arab", "afr_Latn", "ajp_Arab", "aka_Latn",
    "amh_Ethi", "apc_Arab", "arb_Arab", "ars_Arab", "ary_Arab", "arz_Arab", "asm_Beng", "ast_Latn",
    "awa_Deva", "ayr_Latn", "azb_Arab", "azj_Latn", "bak_Cyrl", "bam_Latn", "ban_Latn", "bel_Cyrl",
    "bem_Latn", "ben_Beng", "bho_Deva", "bjn_Arab", "bjn_Latn", "bod_Tibt", "bos_Latn", "bug_Latn",
    "bul_Cyrl", "cat_Latn", "ceb_Latn", "ces_Latn", "cjk_Latn", "ckb_Arab", "crh_Latn", "cym_Latn",
    "dan_Latn", "deu_Latn", "dik_Latn", "dyu_Latn", "dzo_Tibt", "ell_Grek", "eng_Latn", "epo_Latn",
    "est_Latn", "eus_Latn", "ewe_Latn", "fao_Latn", "pes_Arab", "fij_Latn", "fin_Latn", "fon_Latn",
    "fra_Latn", "fur_Latn", "fuv_Latn", "gla_Latn", "gle_Latn", "glg_Latn", "grn_Latn", "guj_Gujr",
    "hat_Latn", "hau_Latn", "heb_Hebr", "hin_Deva", "hne_Deva", "hrv_Latn", "hun_Latn", "hye_Armn",
    "ibo_Latn", "ilo_Latn", "ind_Latn", "isl_Latn", "ita_Latn", "jav_Latn", "jpn_Jpan", "kab_Latn",
    "kac_Latn", "kam_Latn", "kan_Knda", "kas_Arab", "kas_Deva", "kat_Geor", "knc_Arab", "knc_Latn",
    "kaz_Cyrl", "kbp_Latn", "kea_Latn", "khm_Khmr", "kik_Latn", "kin_Latn", "kir_Cyrl", "kmb_Latn",
    "kon_Latn", "kor_Hang", "kmr_Latn", "lao_Laoo", "lvs_Latn", "lij_Latn", "lim_Latn", "lin_Latn",
    "lit_Latn", "lmo_Latn", "ltg_Latn", "ltz_Latn", "lua_Latn", "lug_Latn", "luo_Latn", "lus_Latn",
    "mag_Deva", "mai_Deva", "mal_Mlym", "mar_Deva", "min_Latn", "mkd_Cyrl", "plt_Latn", "mlt_Latn",
    "mni_Beng", "khk_Cyrl", "mos_Latn", "mri_Latn", "zsm_Latn", "mya_Mymr", "nld_Latn", "nno_Latn",
    "nob_Latn", "npi_Deva", "nso_Latn", "nus_Latn", "nya_Latn", "oci_Latn", "gaz_Latn", "ory_Orya",
    "pag_Latn", "pan_Guru", "pap_Latn", "pol_Latn", "por_Latn", "prs_Arab", "pbt_Arab", "quy_Latn",
    "ron_Latn", "run_Latn", "rus_Cyrl", "sag_Latn", "san_Deva", "sat_Beng", "scn_Latn", "shn_Mymr",
    "sin_Sinh", "slk_Latn", "slv_Latn", "smo_Latn", "sna_Latn", "snd_Arab", "som_Latn", "sot_Latn",
    "spa_Latn", "als_Latn", "srd_Latn", "srp_Cyrl", "ssw_Latn", "sun_Latn", "swe_Latn", "swh_Latn",
    "szl_Latn", "tam_Taml", "tat_Cyrl", "tel_Telu", "tgk_Cyrl", "tgl_Latn", "tha_Thai", "tir_Ethi",
    "taq_Latn", "taq_Tfng", "tpi_Latn", "tsn_Latn", "tso_Latn", "tuk_Latn", "tum_Latn", "tur_Latn",
    "twi_Latn", "tzm_Tfng", "uig_Arab", "ukr_Cyrl", "umb_Latn", "urd_Arab", "uzn_Latn", "vec_Latn",
    "vie_Latn", "war_Latn", "wol_Latn", "xho_Latn", "ydd_Hebr", "yor_Latn", "yue_Hant", "zho_Hans",
    "zho_Hant", "zul_Latn",
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

    pub special_token_storage: Option<SpecialTokens>,
}

impl NLLBVocab {
    /// The beginning of sequence token that was used during pretraining.
    /// Can be used a sequence classifier token.
    pub fn bos_value(&self) -> &str {
        self.special_token_storage
            .as_ref()
            .map(|s| s.begin_of_sequence_token.as_str())
            .unwrap_or("<s>")
    }

    /// End of sequence token.
    pub fn eos_value(&self) -> &str {
        self.special_token_storage
            .as_ref()
            .map(|s| s.end_of_sequence_token.as_str())
            .unwrap_or("</s>")
    }

    /// Returns the SEP token for M2M100 (`</s>`)
    pub fn sep_value(&self) -> &str {
        self.special_token_storage
            .as_ref()
            .map(|s| s.end_of_sequence_token.as_str())
            .unwrap_or("</s>")
    }

    /// Returns the PAD token for M2M100 (`<pad>`)
    pub fn pad_value(&self) -> &str {
        self.special_token_storage
            .as_ref()
            .map(|s| s.padding_token.as_str())
            .unwrap_or("<pad>")
    }
}

impl Vocab for NLLBVocab {
    fn unknown_value() -> &'static str {
        "<unk>"
    }

    fn get_unknown_value(&self) -> &str {
        self.special_token_storage
            .as_ref()
            .map(|s| s.unknown_token.as_str())
            .unwrap_or("<unk>")
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

    fn from_file<V: AsRef<Path>, S: AsRef<Path>>(
        vocab: V,
        special: Option<S>,
    ) -> Result<Self, TokenizerError> {
        let special_token_storage: Option<SpecialTokens> = special
            .map(|p| {
                File::open(&p)
                    .context(IOSnafu { path: p.as_ref() })
                    .and_then(|r| serde_json::from_reader(r).context(JsonDeserializeSnafu))
            })
            .transpose()?;

        let mut tokenizer: Tokenizer = File::open(&vocab)
            .context(IOSnafu {
                path: vocab.as_ref(),
            })
            .map(BufReader::new)
            .and_then(|r| serde_json::from_reader(r).context(JsonDeserializeSnafu))?;

        let mut special_values = HashMap::with_capacity(FAIRSEQ_LANGUAGE_CODES.len() + 10);

        let values = &mut tokenizer.model.vocab;
        for language_code in FAIRSEQ_LANGUAGE_CODES.iter() {
            let language_code = if language_code.len() == 3 {
                format!(">>{language_code}<<")
            } else {
                return VocabularyValidationSnafu {
                    message: "NLLB Vocab only supports language code of length 8",
                }
                .fail();
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
            values: tokenizer.model.vocab,
            special_token_storage,
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
