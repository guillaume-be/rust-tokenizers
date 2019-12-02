pub mod preprocessing;
pub use preprocessing::vocab::{base_vocab::BaseVocab, bert_vocab::BertVocab};
pub use preprocessing::tokenizer::bert_tokenizer;
use pyo3::prelude::*;
use crate::preprocessing::tokenizer::bert_tokenizer::BertTokenizer;
use crate::preprocessing::tokenizer::base_tokenizer::Tokenizer;

#[macro_use]
extern crate lazy_static;

#[pyclass(module = "rust_transformers")]
struct PyBertTokenizer {
    tokenizer: BertTokenizer<BertVocab>,
}

#[pymethods]
impl PyBertTokenizer {
    #[new]
    fn new(obj: &PyRawObject, path: String) {
        obj.init(PyBertTokenizer {
            tokenizer: BertTokenizer::from_file(&path),
        });
    }

    fn tokenize(&self, text: &str) -> PyResult<Vec<String>> {
        Ok(self.tokenizer.tokenize(&text))
    }

    fn tokenize_list(&self, text_list: Vec<&str>) -> PyResult<Vec<Vec<String>>> {
        Ok(self.tokenizer.tokenize_list(text_list))
    }
}


#[pymodule]
fn rust_transformers(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyBertTokenizer>()?;

    Ok(())
}