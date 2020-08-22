use pyo3::{PyResult, PyRawObject, Python};
use pyo3::prelude::*;
use pyo3::exceptions;
use rust_tokenizers::{Tokenizer, Vocab, TruncationStrategy, MultiThreadedTokenizer, BertTokenizer, BertVocab, CtrlTokenizer, OpenAiGptVocab, Gpt2Tokenizer, Gpt2Vocab, RobertaTokenizer, RobertaVocab, OpenAiGptTokenizer, XLMRobertaVocab};
use rust_tokenizers::preprocessing::tokenizer::sentence_piece_tokenizer::SentencePieceTokenizer;
use rust_tokenizers::preprocessing::vocab::sentence_piece_vocab::SentencePieceVocab;
use rust_tokenizers::preprocessing::vocab::albert_vocab::AlbertVocab;
use rust_tokenizers::preprocessing::tokenizer::albert_tokenizer::AlbertTokenizer;
use rust_tokenizers::preprocessing::tokenizer::t5_tokenizer::T5Tokenizer;
use rust_tokenizers::preprocessing::vocab::t5_vocab::T5Vocab;
use rust_tokenizers::preprocessing::tokenizer::xlm_roberta_tokenizer::XLMRobertaTokenizer;

#[pyclass]
#[derive(Debug, PartialEq, PartialOrd, Clone)]
pub struct PyTokenizedInput {
    #[pyo3(get)]
    pub token_ids: Vec<i64>,
    #[pyo3(get)]
    pub segment_ids: Vec<i8>,
    #[pyo3(get)]
    pub special_tokens_mask: Vec<i8>,
    #[pyo3(get)]
    pub overflowing_tokens: Vec<i64>,
    #[pyo3(get)]
    pub num_truncated_tokens: usize,
}


trait PyTokenizer<T: Tokenizer<U>, U: Vocab> {
    fn tokenizer(&self) -> &T;

    fn tokenize(&self, text: &str) -> PyResult<Vec<String>> {
        Ok(self.tokenizer().tokenize(&text))
    }

    fn tokenize_list(&self, text_list: Vec<&str>) -> PyResult<Vec<Vec<String>>> {
        Ok(self.tokenizer().tokenize_list(text_list))
    }

    fn encode(&self, text: &str, max_len: usize, truncation_strategy: &str, stride: usize) -> PyResult<PyTokenizedInput> {
        let truncation_strategy = match truncation_strategy {
            "longest_first" => Ok(TruncationStrategy::LongestFirst),
            "only_first" => Ok(TruncationStrategy::OnlyFirst),
            "only_second" => Ok(TruncationStrategy::OnlySecond),
            "do_not_truncate" => Ok(TruncationStrategy::DoNotTruncate),
            _ => Err("Invalid truncation strategy provided. Must be one of `longest_first`, `only_first`, `only_second` or `do_not_truncate`")
        };
        match truncation_strategy {
            Ok(truncation_strategy) => {
                let tokenized_input = self.tokenizer().encode(&text, None, max_len, &truncation_strategy, stride);
                Ok(PyTokenizedInput {
                    token_ids: tokenized_input.token_ids,
                    segment_ids: tokenized_input.segment_ids,
                    special_tokens_mask: tokenized_input.special_tokens_mask,
                    overflowing_tokens: tokenized_input.overflowing_tokens,
                    num_truncated_tokens: tokenized_input.num_truncated_tokens,
                })
            }
            Err(e) => Err(exceptions::ValueError::py_err(e))
        }
    }

    fn encode_pair(&self, text_a: &str, text_b: &str, max_len: usize, truncation_strategy: &str, stride: usize) -> PyResult<PyTokenizedInput> {
        let truncation_strategy = match truncation_strategy {
            "longest_first" => Ok(TruncationStrategy::LongestFirst),
            "only_first" => Ok(TruncationStrategy::OnlyFirst),
            "only_second" => Ok(TruncationStrategy::OnlySecond),
            "do_not_truncate" => Ok(TruncationStrategy::DoNotTruncate),
            _ => Err("Invalid truncation strategy provided. Must be one of `longest_first`, `only_first`, `only_second` or `do_not_truncate`")
        };
        match truncation_strategy {
            Ok(truncation_strategy) => {
                let tokenized_input = self.tokenizer().encode(&text_a, Some(&text_b), max_len, &truncation_strategy, stride);
                Ok(PyTokenizedInput {
                    token_ids: tokenized_input.token_ids,
                    segment_ids: tokenized_input.segment_ids,
                    special_tokens_mask: tokenized_input.special_tokens_mask,
                    overflowing_tokens: tokenized_input.overflowing_tokens,
                    num_truncated_tokens: tokenized_input.num_truncated_tokens,
                })
            }
            Err(e) => Err(exceptions::ValueError::py_err(e))
        }
    }

    fn encode_list(&self, text_list: Vec<&str>, max_len: usize, truncation_strategy: &str, stride: usize) -> PyResult<Vec<PyTokenizedInput>> {
        let truncation_strategy = match truncation_strategy {
            "longest_first" => Ok(TruncationStrategy::LongestFirst),
            "only_first" => Ok(TruncationStrategy::OnlyFirst),
            "only_second" => Ok(TruncationStrategy::OnlySecond),
            "do_not_truncate" => Ok(TruncationStrategy::DoNotTruncate),
            _ => Err("Invalid truncation strategy provided. Must be one of `longest_first`, `only_first`, `only_second` or `do_not_truncate`")
        };
        match truncation_strategy {
            Ok(truncation_strategy) => {
                let tokenized_inputs = self.tokenizer().encode_list(text_list, max_len, &truncation_strategy, stride);
                Ok(tokenized_inputs
                    .into_iter()
                    .map(|tokenized_input| PyTokenizedInput {
                        token_ids: tokenized_input.token_ids,
                        segment_ids: tokenized_input.segment_ids,
                        special_tokens_mask: tokenized_input.special_tokens_mask,
                        overflowing_tokens: tokenized_input.overflowing_tokens,
                        num_truncated_tokens: tokenized_input.num_truncated_tokens,
                    })
                    .collect::<Vec<PyTokenizedInput>>())
            }
            Err(e) => Err(exceptions::ValueError::py_err(e))
        }
    }

    fn encode_pair_list(&self, text_list: Vec<(&str, &str)>, max_len: usize, truncation_strategy: &str, stride: usize) -> PyResult<Vec<PyTokenizedInput>> {
        let truncation_strategy = match truncation_strategy {
            "longest_first" => Ok(TruncationStrategy::LongestFirst),
            "only_first" => Ok(TruncationStrategy::OnlyFirst),
            "only_second" => Ok(TruncationStrategy::OnlySecond),
            "do_not_truncate" => Ok(TruncationStrategy::DoNotTruncate),
            _ => Err("Invalid truncation strategy provided. Must be one of `longest_first`, `only_first`, `only_second` or `do_not_truncate`")
        };
        match truncation_strategy {
            Ok(truncation_strategy) => {
                let tokenized_inputs = self.tokenizer().encode_pair_list(text_list, max_len, &truncation_strategy, stride);
                Ok(tokenized_inputs
                    .into_iter()
                    .map(|tokenized_input| PyTokenizedInput {
                        token_ids: tokenized_input.token_ids,
                        segment_ids: tokenized_input.segment_ids,
                        special_tokens_mask: tokenized_input.special_tokens_mask,
                        overflowing_tokens: tokenized_input.overflowing_tokens,
                        num_truncated_tokens: tokenized_input.num_truncated_tokens,
                    })
                    .collect::<Vec<PyTokenizedInput>>())
            }
            Err(e) => Err(exceptions::ValueError::py_err(e))
        }
    }
}

trait PyMultiThreadTokenizer<T: MultiThreadedTokenizer<U>, U: Vocab>
    where Self: PyTokenizer<T, U> {
    fn tokenize_list(&self, text_list: Vec<&str>) -> PyResult<Vec<Vec<String>>> {
        Ok(MultiThreadedTokenizer::tokenize_list(self.tokenizer(), text_list))
    }

    fn encode_list(&self, text_list: Vec<&str>, max_len: usize, truncation_strategy: &str, stride: usize) -> PyResult<Vec<PyTokenizedInput>> {
        let truncation_strategy = match truncation_strategy {
            "longest_first" => Ok(TruncationStrategy::LongestFirst),
            "only_first" => Ok(TruncationStrategy::OnlyFirst),
            "only_second" => Ok(TruncationStrategy::OnlySecond),
            "do_not_truncate" => Ok(TruncationStrategy::DoNotTruncate),
            _ => Err("Invalid truncation strategy provided. Must be one of `longest_first`, `only_first`, `only_second` or `do_not_truncate`")
        };
        match truncation_strategy {
            Ok(truncation_strategy) => {
                let tokenized_inputs = MultiThreadedTokenizer::encode_list(self.tokenizer(), text_list, max_len, &truncation_strategy, stride);
                Ok(tokenized_inputs
                    .into_iter()
                    .map(|tokenized_input| PyTokenizedInput {
                        token_ids: tokenized_input.token_ids,
                        segment_ids: tokenized_input.segment_ids,
                        special_tokens_mask: tokenized_input.special_tokens_mask,
                        overflowing_tokens: tokenized_input.overflowing_tokens,
                        num_truncated_tokens: tokenized_input.num_truncated_tokens,
                    })
                    .collect::<Vec<PyTokenizedInput>>())
            }
            Err(e) => Err(exceptions::ValueError::py_err(e))
        }
    }

    fn encode_pair_list(&self, text_list: Vec<(&str, &str)>, max_len: usize, truncation_strategy: &str, stride: usize) -> PyResult<Vec<PyTokenizedInput>> {
        let truncation_strategy = match truncation_strategy {
            "longest_first" => Ok(TruncationStrategy::LongestFirst),
            "only_first" => Ok(TruncationStrategy::OnlyFirst),
            "only_second" => Ok(TruncationStrategy::OnlySecond),
            "do_not_truncate" => Ok(TruncationStrategy::DoNotTruncate),
            _ => Err("Invalid truncation strategy provided. Must be one of `longest_first`, `only_first`, `only_second` or `do_not_truncate`")
        };
        match truncation_strategy {
            Ok(truncation_strategy) => {
                let tokenized_inputs = MultiThreadedTokenizer::encode_pair_list(self.tokenizer(), text_list, max_len, &truncation_strategy, stride);
                Ok(tokenized_inputs
                    .into_iter()
                    .map(|tokenized_input| PyTokenizedInput {
                        token_ids: tokenized_input.token_ids,
                        segment_ids: tokenized_input.segment_ids,
                        special_tokens_mask: tokenized_input.special_tokens_mask,
                        overflowing_tokens: tokenized_input.overflowing_tokens,
                        num_truncated_tokens: tokenized_input.num_truncated_tokens,
                    })
                    .collect::<Vec<PyTokenizedInput>>())
            }
            Err(e) => Err(exceptions::ValueError::py_err(e))
        }
    }
}


#[pyclass(module = "rust_tokenizers")]
struct PyBertTokenizer {
    tokenizer: BertTokenizer,
}

impl PyTokenizer<BertTokenizer, BertVocab> for PyBertTokenizer {
    fn tokenizer(&self) -> &BertTokenizer {
        &self.tokenizer
    }
}

impl PyMultiThreadTokenizer<BertTokenizer, BertVocab> for PyBertTokenizer {}

#[pymethods]
impl PyBertTokenizer {
    #[new]
    fn new(obj: &PyRawObject, path: String, do_lower_case: bool) {
        obj.init(PyBertTokenizer {
            tokenizer: BertTokenizer::from_file(path.as_str(), do_lower_case).unwrap(),
        });
    }

    fn tokenize(&self, text: &str) -> PyResult<Vec<String>> {
        <Self as PyTokenizer<BertTokenizer, BertVocab>>::tokenize(&self, text)
    }

    fn tokenize_list(&self, text_list: Vec<&str>) -> PyResult<Vec<Vec<String>>> {
        <Self as PyMultiThreadTokenizer<BertTokenizer, BertVocab>>::tokenize_list(&self, text_list)
    }

    fn encode(&self, text: &str, max_len: usize, truncation_strategy: &str, stride: usize) -> PyResult<PyTokenizedInput> {
        <Self as PyTokenizer<BertTokenizer, BertVocab>>::encode(&self, text, max_len, truncation_strategy, stride)
    }

    fn encode_pair(&self, text_a: &str, text_b: &str, max_len: usize, truncation_strategy: &str, stride: usize) -> PyResult<PyTokenizedInput> {
        <Self as PyTokenizer<BertTokenizer, BertVocab>>::encode_pair(&self, text_a, text_b, max_len, truncation_strategy, stride)
    }

    fn encode_list(&self, text_list: Vec<&str>, max_len: usize, truncation_strategy: &str, stride: usize) -> PyResult<Vec<PyTokenizedInput>> {
        <Self as PyMultiThreadTokenizer<BertTokenizer, BertVocab>>::encode_list(&self, text_list, max_len, truncation_strategy, stride)
    }

    fn encode_pair_list(&self, text_list: Vec<(&str, &str)>, max_len: usize, truncation_strategy: &str, stride: usize) -> PyResult<Vec<PyTokenizedInput>> {
        <Self as PyMultiThreadTokenizer<BertTokenizer, BertVocab>>::encode_pair_list(&self, text_list, max_len, truncation_strategy, stride)
    }
}

#[pyclass(module = "rust_tokenizers")]
struct PyCtrlTokenizer {
    tokenizer: CtrlTokenizer,
}

impl PyTokenizer<CtrlTokenizer, OpenAiGptVocab> for PyCtrlTokenizer {
    fn tokenizer(&self) -> &CtrlTokenizer {
        &self.tokenizer
    }
}

#[pymethods]
impl PyCtrlTokenizer {
    #[new]
    fn new(obj: &PyRawObject, vocab_path: String, merges_path: String, do_lower_case: bool) {
        obj.init(PyCtrlTokenizer {
            tokenizer: CtrlTokenizer::from_file(vocab_path.as_str(), merges_path.as_str(), do_lower_case).unwrap(),
        });
    }

    fn tokenize(&self, text: &str) -> PyResult<Vec<String>> {
        <Self as PyTokenizer<CtrlTokenizer, OpenAiGptVocab>>::tokenize(&self, text)
    }

    fn tokenize_list(&self, text_list: Vec<&str>) -> PyResult<Vec<Vec<String>>> {
        <Self as PyTokenizer<CtrlTokenizer, OpenAiGptVocab>>::tokenize_list(&self, text_list)
    }

    fn encode(&self, text: &str, max_len: usize, truncation_strategy: &str, stride: usize) -> PyResult<PyTokenizedInput> {
        <Self as PyTokenizer<CtrlTokenizer, OpenAiGptVocab>>::encode(&self, text, max_len, truncation_strategy, stride)
    }

    fn encode_pair(&self, text_a: &str, text_b: &str, max_len: usize, truncation_strategy: &str, stride: usize) -> PyResult<PyTokenizedInput> {
        <Self as PyTokenizer<CtrlTokenizer, OpenAiGptVocab>>::encode_pair(&self, text_a, text_b, max_len, truncation_strategy, stride)
    }

    fn encode_list(&self, text_list: Vec<&str>, max_len: usize, truncation_strategy: &str, stride: usize) -> PyResult<Vec<PyTokenizedInput>> {
        <Self as PyTokenizer<CtrlTokenizer, OpenAiGptVocab>>::encode_list(&self, text_list, max_len, truncation_strategy, stride)
    }

    fn encode_pair_list(&self, text_list: Vec<(&str, &str)>, max_len: usize, truncation_strategy: &str, stride: usize) -> PyResult<Vec<PyTokenizedInput>> {
        <Self as PyTokenizer<CtrlTokenizer, OpenAiGptVocab>>::encode_pair_list(&self, text_list, max_len, truncation_strategy, stride)
    }
}


#[pyclass(module = "rust_tokenizers")]
struct PyGpt2Tokenizer {
    tokenizer: Gpt2Tokenizer,
}

impl PyTokenizer<Gpt2Tokenizer, Gpt2Vocab> for PyGpt2Tokenizer {
    fn tokenizer(&self) -> &Gpt2Tokenizer {
        &self.tokenizer
    }
}

#[pymethods]
impl PyGpt2Tokenizer {
    #[new]
    fn new(obj: &PyRawObject, vocab_path: String, merges_path: String, do_lower_case: bool) {
        obj.init(PyGpt2Tokenizer {
            tokenizer: Gpt2Tokenizer::from_file(vocab_path.as_str(), &merges_path.as_str(), do_lower_case).unwrap(),
        });
    }

    fn tokenize(&self, text: &str) -> PyResult<Vec<String>> {
        <Self as PyTokenizer<Gpt2Tokenizer, Gpt2Vocab>>::tokenize(&self, text)
    }

    fn tokenize_list(&self, text_list: Vec<&str>) -> PyResult<Vec<Vec<String>>> {
        <Self as PyTokenizer<Gpt2Tokenizer, Gpt2Vocab>>::tokenize_list(&self, text_list)
    }

    fn encode(&self, text: &str, max_len: usize, truncation_strategy: &str, stride: usize) -> PyResult<PyTokenizedInput> {
        <Self as PyTokenizer<Gpt2Tokenizer, Gpt2Vocab>>::encode(&self, text, max_len, truncation_strategy, stride)
    }

    fn encode_pair(&self, text_a: &str, text_b: &str, max_len: usize, truncation_strategy: &str, stride: usize) -> PyResult<PyTokenizedInput> {
        <Self as PyTokenizer<Gpt2Tokenizer, Gpt2Vocab>>::encode_pair(&self, text_a, text_b, max_len, truncation_strategy, stride)
    }

    fn encode_list(&self, text_list: Vec<&str>, max_len: usize, truncation_strategy: &str, stride: usize) -> PyResult<Vec<PyTokenizedInput>> {
        <Self as PyTokenizer<Gpt2Tokenizer, Gpt2Vocab>>::encode_list(&self, text_list, max_len, truncation_strategy, stride)
    }

    fn encode_pair_list(&self, text_list: Vec<(&str, &str)>, max_len: usize, truncation_strategy: &str, stride: usize) -> PyResult<Vec<PyTokenizedInput>> {
        <Self as PyTokenizer<Gpt2Tokenizer, Gpt2Vocab>>::encode_pair_list(&self, text_list, max_len, truncation_strategy, stride)
    }
}

#[pyclass(module = "rust_tokenizers")]
struct PyRobertaTokenizer {
    tokenizer: RobertaTokenizer,
}

impl PyTokenizer<RobertaTokenizer, RobertaVocab> for PyRobertaTokenizer {
    fn tokenizer(&self) -> &RobertaTokenizer {
        &self.tokenizer
    }
}

#[pymethods]
impl PyRobertaTokenizer {
    #[new]
    fn new(obj: &PyRawObject, vocab_path: String, merges_path: String, do_lower_case: bool) {
        obj.init(PyRobertaTokenizer {
            tokenizer: RobertaTokenizer::from_file(vocab_path.as_str(), &merges_path.as_str(), do_lower_case).unwrap(),
        });
    }

    fn tokenize(&self, text: &str) -> PyResult<Vec<String>> {
        <Self as PyTokenizer<RobertaTokenizer, RobertaVocab>>::tokenize(&self, text)
    }

    fn tokenize_list(&self, text_list: Vec<&str>) -> PyResult<Vec<Vec<String>>> {
        <Self as PyTokenizer<RobertaTokenizer, RobertaVocab>>::tokenize_list(&self, text_list)
    }

    fn encode(&self, text: &str, max_len: usize, truncation_strategy: &str, stride: usize) -> PyResult<PyTokenizedInput> {
        <Self as PyTokenizer<RobertaTokenizer, RobertaVocab>>::encode(&self, text, max_len, truncation_strategy, stride)
    }

    fn encode_pair(&self, text_a: &str, text_b: &str, max_len: usize, truncation_strategy: &str, stride: usize) -> PyResult<PyTokenizedInput> {
        <Self as PyTokenizer<RobertaTokenizer, RobertaVocab>>::encode_pair(&self, text_a, text_b, max_len, truncation_strategy, stride)
    }

    fn encode_list(&self, text_list: Vec<&str>, max_len: usize, truncation_strategy: &str, stride: usize) -> PyResult<Vec<PyTokenizedInput>> {
        <Self as PyTokenizer<RobertaTokenizer, RobertaVocab>>::encode_list(&self, text_list, max_len, truncation_strategy, stride)
    }

    fn encode_pair_list(&self, text_list: Vec<(&str, &str)>, max_len: usize, truncation_strategy: &str, stride: usize) -> PyResult<Vec<PyTokenizedInput>> {
        <Self as PyTokenizer<RobertaTokenizer, RobertaVocab>>::encode_pair_list(&self, text_list, max_len, truncation_strategy, stride)
    }
}

#[pyclass(module = "rust_tokenizers")]
struct PyOpenAiGptTokenizer {
    tokenizer: OpenAiGptTokenizer,
}

impl PyTokenizer<OpenAiGptTokenizer, OpenAiGptVocab> for PyOpenAiGptTokenizer {
    fn tokenizer(&self) -> &OpenAiGptTokenizer {
        &self.tokenizer
    }
}

#[pymethods]
impl PyOpenAiGptTokenizer {
    #[new]
    fn new(obj: &PyRawObject, vocab_path: String, merges_path: String, do_lower_case: bool) {
        obj.init(PyOpenAiGptTokenizer {
            tokenizer: OpenAiGptTokenizer::from_file(vocab_path.as_str(), merges_path.as_str(), do_lower_case).unwrap(),
        });
    }

    fn tokenize(&self, text: &str) -> PyResult<Vec<String>> {
        <Self as PyTokenizer<OpenAiGptTokenizer, OpenAiGptVocab>>::tokenize(&self, text)
    }

    fn tokenize_list(&self, text_list: Vec<&str>) -> PyResult<Vec<Vec<String>>> {
        <Self as PyTokenizer<OpenAiGptTokenizer, OpenAiGptVocab>>::tokenize_list(&self, text_list)
    }

    fn encode(&self, text: &str, max_len: usize, truncation_strategy: &str, stride: usize) -> PyResult<PyTokenizedInput> {
        <Self as PyTokenizer<OpenAiGptTokenizer, OpenAiGptVocab>>::encode(&self, text, max_len, truncation_strategy, stride)
    }

    fn encode_pair(&self, text_a: &str, text_b: &str, max_len: usize, truncation_strategy: &str, stride: usize) -> PyResult<PyTokenizedInput> {
        <Self as PyTokenizer<OpenAiGptTokenizer, OpenAiGptVocab>>::encode_pair(&self, text_a, text_b, max_len, truncation_strategy, stride)
    }

    fn encode_list(&self, text_list: Vec<&str>, max_len: usize, truncation_strategy: &str, stride: usize) -> PyResult<Vec<PyTokenizedInput>> {
        <Self as PyTokenizer<OpenAiGptTokenizer, OpenAiGptVocab>>::encode_list(&self, text_list, max_len, truncation_strategy, stride)
    }

    fn encode_pair_list(&self, text_list: Vec<(&str, &str)>, max_len: usize, truncation_strategy: &str, stride: usize) -> PyResult<Vec<PyTokenizedInput>> {
        <Self as PyTokenizer<OpenAiGptTokenizer, OpenAiGptVocab>>::encode_pair_list(&self, text_list, max_len, truncation_strategy, stride)
    }
}

#[pyclass(module = "rust_tokenizers")]
struct PySentencePieceTokenizer {
    tokenizer: SentencePieceTokenizer,
}

impl PyTokenizer<SentencePieceTokenizer, SentencePieceVocab> for PySentencePieceTokenizer {
    fn tokenizer(&self) -> &SentencePieceTokenizer {
        &self.tokenizer
    }
}

impl PyMultiThreadTokenizer<SentencePieceTokenizer, SentencePieceVocab> for PySentencePieceTokenizer {}

#[pymethods]
impl PySentencePieceTokenizer {
    #[new]
    fn new(obj: &PyRawObject, path: String, do_lower_case: bool) {
        obj.init(PySentencePieceTokenizer {
            tokenizer: SentencePieceTokenizer::from_file(path.as_str(), do_lower_case).unwrap(),
        });
    }

    fn tokenize(&self, text: &str) -> PyResult<Vec<String>> {
        <Self as PyTokenizer<SentencePieceTokenizer, SentencePieceVocab>>::tokenize(&self, text)
    }

    fn tokenize_list(&self, text_list: Vec<&str>) -> PyResult<Vec<Vec<String>>> {
        <Self as PyMultiThreadTokenizer<SentencePieceTokenizer, SentencePieceVocab>>::tokenize_list(&self, text_list)
    }

    fn encode(&self, text: &str, max_len: usize, truncation_strategy: &str, stride: usize) -> PyResult<PyTokenizedInput> {
        <Self as PyTokenizer<SentencePieceTokenizer, SentencePieceVocab>>::encode(&self, text, max_len, truncation_strategy, stride)
    }

    fn encode_pair(&self, text_a: &str, text_b: &str, max_len: usize, truncation_strategy: &str, stride: usize) -> PyResult<PyTokenizedInput> {
        <Self as PyTokenizer<SentencePieceTokenizer, SentencePieceVocab>>::encode_pair(&self, text_a, text_b, max_len, truncation_strategy, stride)
    }

    fn encode_list(&self, text_list: Vec<&str>, max_len: usize, truncation_strategy: &str, stride: usize) -> PyResult<Vec<PyTokenizedInput>> {
        <Self as PyMultiThreadTokenizer<SentencePieceTokenizer, SentencePieceVocab>>::encode_list(&self, text_list, max_len, truncation_strategy, stride)
    }

    fn encode_pair_list(&self, text_list: Vec<(&str, &str)>, max_len: usize, truncation_strategy: &str, stride: usize) -> PyResult<Vec<PyTokenizedInput>> {
        <Self as PyMultiThreadTokenizer<SentencePieceTokenizer, SentencePieceVocab>>::encode_pair_list(&self, text_list, max_len, truncation_strategy, stride)
    }
}

#[pyclass(module = "rust_tokenizers")]
struct PyAlbertTokenizer {
    tokenizer: AlbertTokenizer,
}

impl PyTokenizer<AlbertTokenizer, AlbertVocab> for PyAlbertTokenizer {
    fn tokenizer(&self) -> &AlbertTokenizer {
        &self.tokenizer
    }
}

impl PyMultiThreadTokenizer<AlbertTokenizer, AlbertVocab> for PyAlbertTokenizer {}

#[pymethods]
impl PyAlbertTokenizer {
    #[new]
    fn new(obj: &PyRawObject, path: String, do_lower_case: bool, keep_accents: bool) {
        obj.init(PyAlbertTokenizer {
            tokenizer: AlbertTokenizer::from_file(path.as_str(), do_lower_case, keep_accents).unwrap(),
        });
    }

    fn tokenize(&self, text: &str) -> PyResult<Vec<String>> {
        <Self as PyTokenizer<AlbertTokenizer, AlbertVocab>>::tokenize(&self, text)
    }

    fn tokenize_list(&self, text_list: Vec<&str>) -> PyResult<Vec<Vec<String>>> {
        <Self as PyMultiThreadTokenizer<AlbertTokenizer, AlbertVocab>>::tokenize_list(&self, text_list)
    }

    fn encode(&self, text: &str, max_len: usize, truncation_strategy: &str, stride: usize) -> PyResult<PyTokenizedInput> {
        <Self as PyTokenizer<AlbertTokenizer, AlbertVocab>>::encode(&self, text, max_len, truncation_strategy, stride)
    }

    fn encode_pair(&self, text_a: &str, text_b: &str, max_len: usize, truncation_strategy: &str, stride: usize) -> PyResult<PyTokenizedInput> {
        <Self as PyTokenizer<AlbertTokenizer, AlbertVocab>>::encode_pair(&self, text_a, text_b, max_len, truncation_strategy, stride)
    }

    fn encode_list(&self, text_list: Vec<&str>, max_len: usize, truncation_strategy: &str, stride: usize) -> PyResult<Vec<PyTokenizedInput>> {
        <Self as PyMultiThreadTokenizer<AlbertTokenizer, AlbertVocab>>::encode_list(&self, text_list, max_len, truncation_strategy, stride)
    }

    fn encode_pair_list(&self, text_list: Vec<(&str, &str)>, max_len: usize, truncation_strategy: &str, stride: usize) -> PyResult<Vec<PyTokenizedInput>> {
        <Self as PyMultiThreadTokenizer<AlbertTokenizer, AlbertVocab>>::encode_pair_list(&self, text_list, max_len, truncation_strategy, stride)
    }
}

#[pyclass(module = "rust_tokenizers")]
struct PyT5Tokenizer {
    tokenizer: T5Tokenizer,
}

impl PyTokenizer<T5Tokenizer, T5Vocab> for PyT5Tokenizer {
    fn tokenizer(&self) -> &T5Tokenizer {
        &self.tokenizer
    }
}

impl PyMultiThreadTokenizer<T5Tokenizer, T5Vocab> for PyT5Tokenizer {}

#[pymethods]
impl PyT5Tokenizer {
    #[new]
    fn new(obj: &PyRawObject, path: String, do_lower_case: bool) {
        obj.init(PyT5Tokenizer {
            tokenizer: T5Tokenizer::from_file(path.as_str(), do_lower_case).unwrap(),
        });
    }

    fn tokenize(&self, text: &str) -> PyResult<Vec<String>> {
        <Self as PyTokenizer<T5Tokenizer, T5Vocab>>::tokenize(&self, text)
    }

    fn tokenize_list(&self, text_list: Vec<&str>) -> PyResult<Vec<Vec<String>>> {
        <Self as PyMultiThreadTokenizer<T5Tokenizer, T5Vocab>>::tokenize_list(&self, text_list)
    }

    fn encode(&self, text: &str, max_len: usize, truncation_strategy: &str, stride: usize) -> PyResult<PyTokenizedInput> {
        <Self as PyTokenizer<T5Tokenizer, T5Vocab>>::encode(&self, text, max_len, truncation_strategy, stride)
    }

    fn encode_pair(&self, text_a: &str, text_b: &str, max_len: usize, truncation_strategy: &str, stride: usize) -> PyResult<PyTokenizedInput> {
        <Self as PyTokenizer<T5Tokenizer, T5Vocab>>::encode_pair(&self, text_a, text_b, max_len, truncation_strategy, stride)
    }

    fn encode_list(&self, text_list: Vec<&str>, max_len: usize, truncation_strategy: &str, stride: usize) -> PyResult<Vec<PyTokenizedInput>> {
        <Self as PyMultiThreadTokenizer<T5Tokenizer, T5Vocab>>::encode_list(&self, text_list, max_len, truncation_strategy, stride)
    }

    fn encode_pair_list(&self, text_list: Vec<(&str, &str)>, max_len: usize, truncation_strategy: &str, stride: usize) -> PyResult<Vec<PyTokenizedInput>> {
        <Self as PyMultiThreadTokenizer<T5Tokenizer, T5Vocab>>::encode_pair_list(&self, text_list, max_len, truncation_strategy, stride)
    }
}

#[pyclass(module = "rust_tokenizers")]
struct PyXLMRobertaTokenizer {
    tokenizer: XLMRobertaTokenizer,
}

impl PyTokenizer<XLMRobertaTokenizer, XLMRobertaVocab> for PyXLMRobertaTokenizer {
    fn tokenizer(&self) -> &XLMRobertaTokenizer {
        &self.tokenizer
    }
}

impl PyMultiThreadTokenizer<XLMRobertaTokenizer, XLMRobertaVocab> for PyXLMRobertaTokenizer {}

#[pymethods]
impl PyXLMRobertaTokenizer {
    #[new]
    fn new(obj: &PyRawObject, path: String, do_lower_case: bool) {
        obj.init(PyXLMRobertaTokenizer {
            tokenizer: XLMRobertaTokenizer::from_file(path.as_str(), do_lower_case).unwrap(),
        });
    }

    fn tokenize(&self, text: &str) -> PyResult<Vec<String>> {
        <Self as PyTokenizer<XLMRobertaTokenizer, XLMRobertaVocab>>::tokenize(&self, text)
    }

    fn tokenize_list(&self, text_list: Vec<&str>) -> PyResult<Vec<Vec<String>>> {
        <Self as PyMultiThreadTokenizer<XLMRobertaTokenizer, XLMRobertaVocab>>::tokenize_list(&self, text_list)
    }

    fn encode(&self, text: &str, max_len: usize, truncation_strategy: &str, stride: usize) -> PyResult<PyTokenizedInput> {
        <Self as PyTokenizer<XLMRobertaTokenizer, XLMRobertaVocab>>::encode(&self, text, max_len, truncation_strategy, stride)
    }

    fn encode_pair(&self, text_a: &str, text_b: &str, max_len: usize, truncation_strategy: &str, stride: usize) -> PyResult<PyTokenizedInput> {
        <Self as PyTokenizer<XLMRobertaTokenizer, XLMRobertaVocab>>::encode_pair(&self, text_a, text_b, max_len, truncation_strategy, stride)
    }

    fn encode_list(&self, text_list: Vec<&str>, max_len: usize, truncation_strategy: &str, stride: usize) -> PyResult<Vec<PyTokenizedInput>> {
        <Self as PyMultiThreadTokenizer<XLMRobertaTokenizer, XLMRobertaVocab>>::encode_list(&self, text_list, max_len, truncation_strategy, stride)
    }

    fn encode_pair_list(&self, text_list: Vec<(&str, &str)>, max_len: usize, truncation_strategy: &str, stride: usize) -> PyResult<Vec<PyTokenizedInput>> {
        <Self as PyMultiThreadTokenizer<XLMRobertaTokenizer, XLMRobertaVocab>>::encode_pair_list(&self, text_list, max_len, truncation_strategy, stride)
    }
}


#[pymodule]
fn rust_tokenizers(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyBertTokenizer>()?;
    m.add_class::<PyCtrlTokenizer>()?;
    m.add_class::<PyGpt2Tokenizer>()?;
    m.add_class::<PyRobertaTokenizer>()?;
    m.add_class::<PyOpenAiGptTokenizer>()?;
    m.add_class::<PySentencePieceTokenizer>()?;
    m.add_class::<PyAlbertTokenizer>()?;
    m.add_class::<PyT5Tokenizer>()?;
    m.add_class::<PyXLMRobertaTokenizer>()?;
    Ok(())
}