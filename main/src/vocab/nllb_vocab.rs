use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::BufReader,
    path::Path,
};

use serde::{Deserialize, Deserializer};

use crate::error::*;

use super::{
    base_vocab::{register_as_special_value, swap_key_values, SpecialTokenMap},
    Vocab,
};

pub const EXTENDED_FAIRSEQ_LANGUAGE_CODES: [&str; 202] = [
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

#[derive(Debug, Default, Clone, Deserialize)]
pub struct NLLBSpecialTokenMap {
    pub unk_token: String,
    pub pad_token: Option<String>,
    pub bos_token: Option<String>,
    pub sep_token: Option<String>,
    pub cls_token: Option<String>,
    pub eos_token: Option<String>,
    #[serde(deserialize_with = "get_nllb_mask")]
    pub mask_token: Option<String>,
    pub additional_special_tokens: Option<HashSet<String>>,
}

fn get_nllb_mask<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    struct MaskHelper {
        mask_token: Option<String>,
        _lstrip: Option<bool>,
        _normalized: Option<bool>,
        _rstrip: Option<bool>,
        _single_word: Option<bool>,
    }
    let helper = MaskHelper::deserialize(deserializer)?;
    Ok(helper.mask_token)
}

impl From<NLLBSpecialTokenMap> for SpecialTokenMap {
    fn from(value: NLLBSpecialTokenMap) -> Self {
        SpecialTokenMap {
            unk_token: value.unk_token,
            pad_token: value.pad_token,
            bos_token: value.bos_token,
            sep_token: value.sep_token,
            cls_token: value.cls_token,
            eos_token: value.eos_token,
            mask_token: value.mask_token,
            additional_special_tokens: value.additional_special_tokens,
        }
    }
}

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

    pub special_token_map: SpecialTokenMap,
}

const DEFAULT_UNK_TOKEN: &str = "<unk>";
const DEFAULT_PAD_TOKEN: &str = "<pad>";
const DEFAULT_BOS_TOKEN: &str = "<s>";
const DEFAULT_SEP_TOKEN: &str = "</s>";
const DEFAULT_EOS_TOKEN: &str = "</s>";

impl NLLBVocab {
    /// The beginning of sequence token that was used during pretraining.
    /// Can be used a sequence classifier token.
    pub fn get_bos_value(&self) -> &str {
        self.special_token_map
            .bos_token
            .as_deref()
            .unwrap_or(DEFAULT_BOS_TOKEN)
    }

    /// End of sequence token.
    pub fn get_eos_value(&self) -> &str {
        self.special_token_map
            .eos_token
            .as_deref()
            .unwrap_or(DEFAULT_EOS_TOKEN)
    }

    /// Returns the SEP token for NLLB (`</s>`)
    pub fn get_sep_value(&self) -> &str {
        self.special_token_map
            .sep_token
            .as_deref()
            .unwrap_or(DEFAULT_SEP_TOKEN)
    }

    /// Returns the PAD token for NLLB (`<pad>`)
    pub fn get_pad_value(&self) -> &str {
        self.special_token_map
            .pad_token
            .as_deref()
            .unwrap_or(DEFAULT_PAD_TOKEN)
    }
}

impl Vocab for NLLBVocab {
    fn get_unknown_value(&self) -> &str {
        &self.special_token_map.unk_token
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

    fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, TokenizerError> {
        let values = Tokenizer::deserialize(path)?.model.vocab;

        let special_token_map = SpecialTokenMap {
            unk_token: DEFAULT_UNK_TOKEN.to_string(),
            pad_token: Some(DEFAULT_PAD_TOKEN.to_string()),
            bos_token: Some(DEFAULT_BOS_TOKEN.to_string()),
            sep_token: Some(DEFAULT_SEP_TOKEN.to_string()),
            cls_token: None,
            eos_token: Some(DEFAULT_EOS_TOKEN.to_string()),
            mask_token: None,
            additional_special_tokens: None,
        };

        Self::from_values_and_special_token_map(values, special_token_map)
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

    fn from_file_with_special_token_mapping<P: AsRef<Path>, S: AsRef<Path>>(
        path: P,
        special_token_mapping_path: S,
    ) -> Result<Self, TokenizerError>
    where
        Self: Sized,
    {
        let values = Tokenizer::deserialize(path)?.model.vocab;
        let f = File::open(&special_token_mapping_path).map_err(|e| {
            TokenizerError::FileNotFound(format!(
                "{} vocabulary file not found :{}",
                special_token_mapping_path.as_ref().display(),
                e
            ))
        })?;
        let br = BufReader::new(f);
        let special_config: NLLBSpecialTokenMap = serde_json::from_reader(br).map_err(|e| {
            TokenizerError::FileNotFound(format!("Invalid special token mapping file {}", e))
        })?;

        Self::from_values_and_special_token_map(values, special_config.into())
    }

    fn from_values_and_special_token_map(
        values: HashMap<String, i64>,
        special_token_map: SpecialTokenMap,
    ) -> Result<Self, TokenizerError>
    where
        Self: Sized,
    {
        let mut result = Self {
            values,
            indices: HashMap::new(),
            special_values: HashMap::new(),
            special_indices: HashMap::new(),
            language_codes_bytes: HashSet::new(),
            special_token_map,
        };

        let mut special_values = HashMap::new();
        let mut language_code_bytes = HashSet::new();

        let mut reserve_special =
            |t| register_as_special_value(t, &result.values, &mut special_values);

        reserve_special(result.get_bos_value())?;
        reserve_special(result.get_eos_value())?;
        reserve_special(result.get_sep_value())?;
        reserve_special(result.get_pad_value())?;
        reserve_special(result.get_unknown_value())?;

        if let Some(languages) = result.special_token_map.additional_special_tokens.as_ref() {
            for language in languages {
                reserve_special(language)?;
                language_code_bytes.insert(language.as_bytes().to_vec());
            }
        } else {
            for language in EXTENDED_FAIRSEQ_LANGUAGE_CODES {
                reserve_special(language)?;
                language_code_bytes.insert(language.as_bytes().to_vec());
            }
        }

        let indices = swap_key_values(&result.values);
        let special_indices = swap_key_values(&special_values);
        result.indices = indices;
        result.special_indices = special_indices;
        result.special_values = special_values;
        result.language_codes_bytes = language_code_bytes;
        Ok(result)
    }
}

#[derive(Deserialize)]
struct Model {
    vocab: HashMap<String, i64>,
}

#[derive(Deserialize)]
struct Tokenizer {
    model: Model,
}

impl Tokenizer {
    fn deserialize<P: AsRef<Path>>(path: P) -> Result<Self, TokenizerError> {
        let file = File::open(&path).map_err(|e| {
            TokenizerError::FileNotFound(format!(
                "{} tokenizer file not found: {}",
                path.as_ref().display(),
                e
            ))
        })?;

        let reader = BufReader::new(file);

        serde_json::from_reader(reader)
            .map_err(|e| TokenizerError::VocabularyParsingError(e.to_string()))
    }
}
