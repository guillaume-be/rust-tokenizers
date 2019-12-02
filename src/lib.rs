pub mod preprocessing;
pub use preprocessing::vocab::{base_vocab::BaseVocab, bert_vocab::BertVocab};
pub use preprocessing::tokenizer::bert_tokenizer;
use pyo3::prelude::*;
//use pyo3::wrap_pyfunction;
use crate::preprocessing::tokenizer::bert_tokenizer::BertTokenizer;
use crate::preprocessing::tokenizer::base_tokenizer::Tokenizer;

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

    fn tokenize(&self, text: String) -> PyResult<Vec<String>> {
        Ok(self.tokenizer.tokenize(&text))
    }
}


#[pymodule]
fn rust_transformers(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyBertTokenizer>()?;

    Ok(())
}