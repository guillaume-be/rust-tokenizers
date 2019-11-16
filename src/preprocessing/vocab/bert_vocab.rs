use std::collections::HashMap;
use crate::preprocessing::vocab::base_vocab::Vocab;
use std::process;

pub struct BertVocab {
    pub values: HashMap<String, i64>,
    pub unknown_value: &'static str,
    pub special_values: HashMap<String, i64>,
}

impl BertVocab {
    fn pad_value() -> &'static str { "[PAD]" }
    fn sep_value() -> &'static str { "[SEP]" }
    fn cls_value() -> &'static str { "[CLS]" }
    fn mask_value() -> &'static str { "[MASK]" }
}

impl Vocab for BertVocab {
    fn unknown_value() -> &'static str { "[UNK]" }

    fn from_file(path: &str) -> BertVocab {
        let values = BertVocab::read_vocab_file(path);
        let mut special_values = HashMap::new();

        let unknown_value = BertVocab::unknown_value();
        BertVocab::_register_as_special_value(unknown_value, &values, &mut special_values);

        let pad_value = BertVocab::pad_value();
        BertVocab::_register_as_special_value(pad_value, &values, &mut special_values);

        let sep_value = BertVocab::sep_value();
        BertVocab::_register_as_special_value(sep_value, &values, &mut special_values);

        let cls_value = BertVocab::cls_value();
        BertVocab::_register_as_special_value(cls_value, &values, &mut special_values);

        let mask_value = BertVocab::mask_value();
        BertVocab::_register_as_special_value(mask_value, &values, &mut special_values);

        BertVocab { values, unknown_value, special_values }
    }

    fn token_to_id(&self, token: &str) -> i64 {
        match self._token_to_id(token, &self.values, &self.special_values, &self.unknown_value) {
            Ok(index) => index,
            Err(err) => {
                println!("{}", err);
                process::exit(1);
            }
        }
    }
}