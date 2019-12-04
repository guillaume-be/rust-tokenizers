pub mod preprocessing;

pub use preprocessing::vocab::{base_vocab::BaseVocab, bert_vocab::BertVocab};
pub use preprocessing::tokenizer::bert_tokenizer;
use pyo3::prelude::*;
use crate::preprocessing::tokenizer::bert_tokenizer::BertTokenizer;
use crate::preprocessing::tokenizer::base_tokenizer::{Tokenizer, TruncationStrategy};
use pyo3::exceptions;

#[macro_use]
extern crate lazy_static;

#[pyclass(module = "rust_transformers")]
struct PyBertTokenizer {
    tokenizer: BertTokenizer,
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

    fn encode(&self, text: &str, max_len: usize, truncation_strategy: &str, stride: usize) -> PyResult<Vec<i64>> {
        let truncation_strategy = match truncation_strategy {
            "only_first" => Ok(TruncationStrategy::OnlyFirst),
            _ => Err("Invalid truncation strategy provided")
        };
        match truncation_strategy {
            Ok(truncation_strategy) => Ok(self.tokenizer.encode(&text, None, max_len, &truncation_strategy, stride)),
            Err(e) => Err(exceptions::ValueError::py_err(e))
        }
    }

    fn encode_list(&self, text_list: Vec<&str>, max_len: usize, truncation_strategy: &str, stride: usize) -> PyResult<Vec<Vec<i64>>> {
        let truncation_strategy = match truncation_strategy {
            "only_first" => Ok(TruncationStrategy::OnlyFirst),
            _ => Err("Invalid truncation strategy provided")
        };
        match truncation_strategy {
            Ok(truncation_strategy) => Ok(self.tokenizer.encode_list(text_list, max_len, &truncation_strategy, stride)),
            Err(e) => Err(exceptions::ValueError::py_err(e))
        }
    }
}


#[pymodule]
fn rust_transformers(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyBertTokenizer>()?;

    Ok(())
}