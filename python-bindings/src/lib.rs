use pyo3::exceptions;
use pyo3::prelude::*;
use rust_tokenizers::tokenizer::{AlbertTokenizer, BertTokenizer, CtrlTokenizer, DeBERTaTokenizer, DeBERTaV2Tokenizer, FNetTokenizer, Gpt2Tokenizer, M2M100Tokenizer, MBart50Tokenizer, MultiThreadedTokenizer, NLLBTokenizer, OpenAiGptTokenizer, PegasusTokenizer, ProphetNetTokenizer, ReformerTokenizer, RobertaTokenizer, SentencePieceBpeTokenizer, SentencePieceTokenizer, T5Tokenizer, Tokenizer, TruncationStrategy, XLMRobertaTokenizer, XLNetTokenizer};
use rust_tokenizers::vocab::{AlbertVocab, BertVocab, DeBERTaV2Vocab, DeBERTaVocab, FNetVocab, Gpt2Vocab, M2M100Vocab, MBart50Vocab, NLLBVocab, OpenAiGptVocab, PegasusVocab, ProphetNetVocab, ReformerVocab, RobertaVocab, SentencePieceVocab, T5Vocab, Vocab, XLMRobertaVocab, XLNetVocab};

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
        Ok(self.tokenizer().tokenize(text))
    }

    fn tokenize_list(&self, text_list: Vec<&str>) -> PyResult<Vec<Vec<String>>> {
        Ok(self.tokenizer().tokenize_list(text_list.as_slice()))
    }

    fn encode(
        &self,
        text: &str,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<PyTokenizedInput> {
        let truncation_strategy = match truncation_strategy {
            "longest_first" => Ok(TruncationStrategy::LongestFirst),
            "only_first" => Ok(TruncationStrategy::OnlyFirst),
            "only_second" => Ok(TruncationStrategy::OnlySecond),
            "do_not_truncate" => Ok(TruncationStrategy::DoNotTruncate),
            _ => Err("Invalid truncation strategy provided. Must be one of `longest_first`, `only_first`, `only_second` or `do_not_truncate`")
        };
        match truncation_strategy {
            Ok(truncation_strategy) => {
                let tokenized_input =
                    self.tokenizer()
                        .encode(text, None, max_len, &truncation_strategy, stride);
                Ok(PyTokenizedInput {
                    token_ids: tokenized_input.token_ids,
                    segment_ids: tokenized_input.segment_ids,
                    special_tokens_mask: tokenized_input.special_tokens_mask,
                    overflowing_tokens: tokenized_input.overflowing_tokens,
                    num_truncated_tokens: tokenized_input.num_truncated_tokens,
                })
            }
            Err(e) => Err(exceptions::PyValueError::new_err(e)),
        }
    }

    fn encode_pair(
        &self,
        text_a: &str,
        text_b: &str,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<PyTokenizedInput> {
        let truncation_strategy = match truncation_strategy {
            "longest_first" => Ok(TruncationStrategy::LongestFirst),
            "only_first" => Ok(TruncationStrategy::OnlyFirst),
            "only_second" => Ok(TruncationStrategy::OnlySecond),
            "do_not_truncate" => Ok(TruncationStrategy::DoNotTruncate),
            _ => Err("Invalid truncation strategy provided. Must be one of `longest_first`, `only_first`, `only_second` or `do_not_truncate`")
        };
        match truncation_strategy {
            Ok(truncation_strategy) => {
                let tokenized_input = self.tokenizer().encode(
                    text_a,
                    Some(text_b),
                    max_len,
                    &truncation_strategy,
                    stride,
                );
                Ok(PyTokenizedInput {
                    token_ids: tokenized_input.token_ids,
                    segment_ids: tokenized_input.segment_ids,
                    special_tokens_mask: tokenized_input.special_tokens_mask,
                    overflowing_tokens: tokenized_input.overflowing_tokens,
                    num_truncated_tokens: tokenized_input.num_truncated_tokens,
                })
            }
            Err(e) => Err(exceptions::PyValueError::new_err(e)),
        }
    }

    fn encode_list(
        &self,
        text_list: Vec<&str>,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<Vec<PyTokenizedInput>> {
        let truncation_strategy = match truncation_strategy {
            "longest_first" => Ok(TruncationStrategy::LongestFirst),
            "only_first" => Ok(TruncationStrategy::OnlyFirst),
            "only_second" => Ok(TruncationStrategy::OnlySecond),
            "do_not_truncate" => Ok(TruncationStrategy::DoNotTruncate),
            _ => Err("Invalid truncation strategy provided. Must be one of `longest_first`, `only_first`, `only_second` or `do_not_truncate`")
        };
        match truncation_strategy {
            Ok(truncation_strategy) => {
                let tokenized_inputs = self.tokenizer().encode_list(
                    text_list.as_slice(),
                    max_len,
                    &truncation_strategy,
                    stride,
                );
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
            Err(e) => Err(exceptions::PyValueError::new_err(e)),
        }
    }

    fn encode_pair_list(
        &self,
        text_list: Vec<(&str, &str)>,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<Vec<PyTokenizedInput>> {
        let truncation_strategy = match truncation_strategy {
            "longest_first" => Ok(TruncationStrategy::LongestFirst),
            "only_first" => Ok(TruncationStrategy::OnlyFirst),
            "only_second" => Ok(TruncationStrategy::OnlySecond),
            "do_not_truncate" => Ok(TruncationStrategy::DoNotTruncate),
            _ => Err("Invalid truncation strategy provided. Must be one of `longest_first`, `only_first`, `only_second` or `do_not_truncate`")
        };
        match truncation_strategy {
            Ok(truncation_strategy) => {
                let tokenized_inputs = self.tokenizer().encode_pair_list(
                    text_list.as_slice(),
                    max_len,
                    &truncation_strategy,
                    stride,
                );
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
            Err(e) => Err(exceptions::PyValueError::new_err(e)),
        }
    }
}

trait PyMultiThreadTokenizer<T: MultiThreadedTokenizer<U>, U: Vocab>
where
    Self: PyTokenizer<T, U>,
{
    fn tokenize_list(&self, text_list: Vec<&str>) -> PyResult<Vec<Vec<String>>> {
        Ok(MultiThreadedTokenizer::tokenize_list(
            self.tokenizer(),
            text_list.as_slice(),
        ))
    }

    fn encode_list(
        &self,
        text_list: Vec<&str>,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<Vec<PyTokenizedInput>> {
        let truncation_strategy = match truncation_strategy {
            "longest_first" => Ok(TruncationStrategy::LongestFirst),
            "only_first" => Ok(TruncationStrategy::OnlyFirst),
            "only_second" => Ok(TruncationStrategy::OnlySecond),
            "do_not_truncate" => Ok(TruncationStrategy::DoNotTruncate),
            _ => Err("Invalid truncation strategy provided. Must be one of `longest_first`, `only_first`, `only_second` or `do_not_truncate`")
        };
        match truncation_strategy {
            Ok(truncation_strategy) => {
                let tokenized_inputs = MultiThreadedTokenizer::encode_list(
                    self.tokenizer(),
                    &text_list,
                    max_len,
                    &truncation_strategy,
                    stride,
                );
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
            Err(e) => Err(exceptions::PyValueError::new_err(e)),
        }
    }

    fn encode_pair_list(
        &self,
        text_list: Vec<(&str, &str)>,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<Vec<PyTokenizedInput>> {
        let truncation_strategy = match truncation_strategy {
            "longest_first" => Ok(TruncationStrategy::LongestFirst),
            "only_first" => Ok(TruncationStrategy::OnlyFirst),
            "only_second" => Ok(TruncationStrategy::OnlySecond),
            "do_not_truncate" => Ok(TruncationStrategy::DoNotTruncate),
            _ => Err("Invalid truncation strategy provided. Must be one of `longest_first`, `only_first`, `only_second` or `do_not_truncate`")
        };
        match truncation_strategy {
            Ok(truncation_strategy) => {
                let tokenized_inputs = MultiThreadedTokenizer::encode_pair_list(
                    self.tokenizer(),
                    &text_list,
                    max_len,
                    &truncation_strategy,
                    stride,
                );
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
            Err(e) => Err(exceptions::PyValueError::new_err(e)),
        }
    }
}

#[pyclass(dict, module = "rust_tokenizers")]
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
    fn new(path: String, do_lower_case: bool, strip_accents: bool) -> Self {
        PyBertTokenizer {
            tokenizer: BertTokenizer::from_file(path.as_str(), do_lower_case, strip_accents)
                .unwrap(),
        }
    }

    fn tokenize(&self, text: &str) -> PyResult<Vec<String>> {
        <Self as PyTokenizer<BertTokenizer, BertVocab>>::tokenize(self, text)
    }

    fn tokenize_list(&self, text_list: Vec<&str>) -> PyResult<Vec<Vec<String>>> {
        <Self as PyMultiThreadTokenizer<BertTokenizer, BertVocab>>::tokenize_list(self, text_list)
    }

    fn encode(
        &self,
        text: &str,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<PyTokenizedInput> {
        <Self as PyTokenizer<BertTokenizer, BertVocab>>::encode(
            self,
            text,
            max_len,
            truncation_strategy,
            stride,
        )
    }

    fn encode_pair(
        &self,
        text_a: &str,
        text_b: &str,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<PyTokenizedInput> {
        <Self as PyTokenizer<BertTokenizer, BertVocab>>::encode_pair(
            self,
            text_a,
            text_b,
            max_len,
            truncation_strategy,
            stride,
        )
    }

    fn encode_list(
        &self,
        text_list: Vec<&str>,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<Vec<PyTokenizedInput>> {
        <Self as PyMultiThreadTokenizer<BertTokenizer, BertVocab>>::encode_list(
            self,
            text_list,
            max_len,
            truncation_strategy,
            stride,
        )
    }

    fn encode_pair_list(
        &self,
        text_list: Vec<(&str, &str)>,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<Vec<PyTokenizedInput>> {
        <Self as PyMultiThreadTokenizer<BertTokenizer, BertVocab>>::encode_pair_list(
            self,
            text_list,
            max_len,
            truncation_strategy,
            stride,
        )
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

impl PyMultiThreadTokenizer<CtrlTokenizer, OpenAiGptVocab> for PyCtrlTokenizer {}

#[pymethods]
impl PyCtrlTokenizer {
    #[new]
    fn new(vocab_path: String, merges_path: String, do_lower_case: bool) -> Self {
        PyCtrlTokenizer {
            tokenizer: CtrlTokenizer::from_file(
                vocab_path.as_str(),
                merges_path.as_str(),
                do_lower_case,
            )
            .unwrap(),
        }
    }

    fn tokenize(&self, text: &str) -> PyResult<Vec<String>> {
        <Self as PyTokenizer<CtrlTokenizer, OpenAiGptVocab>>::tokenize(self, text)
    }

    fn tokenize_list(&self, text_list: Vec<&str>) -> PyResult<Vec<Vec<String>>> {
        <Self as PyMultiThreadTokenizer<CtrlTokenizer, OpenAiGptVocab>>::tokenize_list(
            self, text_list,
        )
    }

    fn encode(
        &self,
        text: &str,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<PyTokenizedInput> {
        <Self as PyTokenizer<CtrlTokenizer, OpenAiGptVocab>>::encode(
            self,
            text,
            max_len,
            truncation_strategy,
            stride,
        )
    }

    fn encode_pair(
        &self,
        text_a: &str,
        text_b: &str,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<PyTokenizedInput> {
        <Self as PyTokenizer<CtrlTokenizer, OpenAiGptVocab>>::encode_pair(
            self,
            text_a,
            text_b,
            max_len,
            truncation_strategy,
            stride,
        )
    }

    fn encode_list(
        &self,
        text_list: Vec<&str>,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<Vec<PyTokenizedInput>> {
        <Self as PyMultiThreadTokenizer<CtrlTokenizer, OpenAiGptVocab>>::encode_list(
            self,
            text_list,
            max_len,
            truncation_strategy,
            stride,
        )
    }

    fn encode_pair_list(
        &self,
        text_list: Vec<(&str, &str)>,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<Vec<PyTokenizedInput>> {
        <Self as PyMultiThreadTokenizer<CtrlTokenizer, OpenAiGptVocab>>::encode_pair_list(
            self,
            text_list,
            max_len,
            truncation_strategy,
            stride,
        )
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

impl PyMultiThreadTokenizer<Gpt2Tokenizer, Gpt2Vocab> for PyGpt2Tokenizer {}

#[pymethods]
impl PyGpt2Tokenizer {
    #[new]
    fn new(vocab_path: String, merges_path: String, do_lower_case: bool) -> Self {
        PyGpt2Tokenizer {
            tokenizer: Gpt2Tokenizer::from_file(
                vocab_path.as_str(),
                merges_path.as_str(),
                do_lower_case,
            )
            .unwrap(),
        }
    }

    fn tokenize(&self, text: &str) -> PyResult<Vec<String>> {
        <Self as PyTokenizer<Gpt2Tokenizer, Gpt2Vocab>>::tokenize(self, text)
    }

    fn tokenize_list(&self, text_list: Vec<&str>) -> PyResult<Vec<Vec<String>>> {
        <Self as PyMultiThreadTokenizer<Gpt2Tokenizer, Gpt2Vocab>>::tokenize_list(self, text_list)
    }

    fn encode(
        &self,
        text: &str,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<PyTokenizedInput> {
        <Self as PyTokenizer<Gpt2Tokenizer, Gpt2Vocab>>::encode(
            self,
            text,
            max_len,
            truncation_strategy,
            stride,
        )
    }

    fn encode_pair(
        &self,
        text_a: &str,
        text_b: &str,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<PyTokenizedInput> {
        <Self as PyTokenizer<Gpt2Tokenizer, Gpt2Vocab>>::encode_pair(
            self,
            text_a,
            text_b,
            max_len,
            truncation_strategy,
            stride,
        )
    }

    fn encode_list(
        &self,
        text_list: Vec<&str>,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<Vec<PyTokenizedInput>> {
        <Self as PyMultiThreadTokenizer<Gpt2Tokenizer, Gpt2Vocab>>::encode_list(
            self,
            text_list,
            max_len,
            truncation_strategy,
            stride,
        )
    }

    fn encode_pair_list(
        &self,
        text_list: Vec<(&str, &str)>,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<Vec<PyTokenizedInput>> {
        <Self as PyMultiThreadTokenizer<Gpt2Tokenizer, Gpt2Vocab>>::encode_pair_list(
            self,
            text_list,
            max_len,
            truncation_strategy,
            stride,
        )
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

impl PyMultiThreadTokenizer<RobertaTokenizer, RobertaVocab> for PyRobertaTokenizer {}

#[pymethods]
impl PyRobertaTokenizer {
    #[new]
    fn new(
        vocab_path: String,
        merges_path: String,
        do_lower_case: bool,
        add_prefix_space: bool,
    ) -> Self {
        PyRobertaTokenizer {
            tokenizer: RobertaTokenizer::from_file(
                vocab_path.as_str(),
                merges_path.as_str(),
                do_lower_case,
                add_prefix_space,
            )
            .unwrap(),
        }
    }

    fn tokenize(&self, text: &str) -> PyResult<Vec<String>> {
        <Self as PyTokenizer<RobertaTokenizer, RobertaVocab>>::tokenize(self, text)
    }

    fn tokenize_list(&self, text_list: Vec<&str>) -> PyResult<Vec<Vec<String>>> {
        <Self as PyMultiThreadTokenizer<RobertaTokenizer, RobertaVocab>>::tokenize_list(
            self, text_list,
        )
    }

    fn encode(
        &self,
        text: &str,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<PyTokenizedInput> {
        <Self as PyTokenizer<RobertaTokenizer, RobertaVocab>>::encode(
            self,
            text,
            max_len,
            truncation_strategy,
            stride,
        )
    }

    fn encode_pair(
        &self,
        text_a: &str,
        text_b: &str,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<PyTokenizedInput> {
        <Self as PyTokenizer<RobertaTokenizer, RobertaVocab>>::encode_pair(
            self,
            text_a,
            text_b,
            max_len,
            truncation_strategy,
            stride,
        )
    }

    fn encode_list(
        &self,
        text_list: Vec<&str>,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<Vec<PyTokenizedInput>> {
        <Self as PyMultiThreadTokenizer<RobertaTokenizer, RobertaVocab>>::encode_list(
            self,
            text_list,
            max_len,
            truncation_strategy,
            stride,
        )
    }

    fn encode_pair_list(
        &self,
        text_list: Vec<(&str, &str)>,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<Vec<PyTokenizedInput>> {
        <Self as PyMultiThreadTokenizer<RobertaTokenizer, RobertaVocab>>::encode_pair_list(
            self,
            text_list,
            max_len,
            truncation_strategy,
            stride,
        )
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

impl PyMultiThreadTokenizer<OpenAiGptTokenizer, OpenAiGptVocab> for PyOpenAiGptTokenizer {}

#[pymethods]
impl PyOpenAiGptTokenizer {
    #[new]
    fn new(vocab_path: String, merges_path: String, do_lower_case: bool) -> Self {
        PyOpenAiGptTokenizer {
            tokenizer: OpenAiGptTokenizer::from_file(
                vocab_path.as_str(),
                merges_path.as_str(),
                do_lower_case,
            )
            .unwrap(),
        }
    }

    fn tokenize(&self, text: &str) -> PyResult<Vec<String>> {
        <Self as PyTokenizer<OpenAiGptTokenizer, OpenAiGptVocab>>::tokenize(self, text)
    }

    fn tokenize_list(&self, text_list: Vec<&str>) -> PyResult<Vec<Vec<String>>> {
        <Self as PyMultiThreadTokenizer<OpenAiGptTokenizer, OpenAiGptVocab>>::tokenize_list(
            self, text_list,
        )
    }

    fn encode(
        &self,
        text: &str,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<PyTokenizedInput> {
        <Self as PyTokenizer<OpenAiGptTokenizer, OpenAiGptVocab>>::encode(
            self,
            text,
            max_len,
            truncation_strategy,
            stride,
        )
    }

    fn encode_pair(
        &self,
        text_a: &str,
        text_b: &str,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<PyTokenizedInput> {
        <Self as PyTokenizer<OpenAiGptTokenizer, OpenAiGptVocab>>::encode_pair(
            self,
            text_a,
            text_b,
            max_len,
            truncation_strategy,
            stride,
        )
    }

    fn encode_list(
        &self,
        text_list: Vec<&str>,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<Vec<PyTokenizedInput>> {
        <Self as PyMultiThreadTokenizer<OpenAiGptTokenizer, OpenAiGptVocab>>::encode_list(
            self,
            text_list,
            max_len,
            truncation_strategy,
            stride,
        )
    }

    fn encode_pair_list(
        &self,
        text_list: Vec<(&str, &str)>,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<Vec<PyTokenizedInput>> {
        <Self as PyMultiThreadTokenizer<OpenAiGptTokenizer, OpenAiGptVocab>>::encode_pair_list(
            self,
            text_list,
            max_len,
            truncation_strategy,
            stride,
        )
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

impl PyMultiThreadTokenizer<SentencePieceTokenizer, SentencePieceVocab>
    for PySentencePieceTokenizer
{
}

#[pymethods]
impl PySentencePieceTokenizer {
    #[new]
    fn new(path: String, do_lower_case: bool) -> Self {
        PySentencePieceTokenizer {
            tokenizer: SentencePieceTokenizer::from_file(path.as_str(), do_lower_case).unwrap(),
        }
    }

    fn tokenize(&self, text: &str) -> PyResult<Vec<String>> {
        <Self as PyTokenizer<SentencePieceTokenizer, SentencePieceVocab>>::tokenize(self, text)
    }

    fn tokenize_list(&self, text_list: Vec<&str>) -> PyResult<Vec<Vec<String>>> {
        <Self as PyMultiThreadTokenizer<SentencePieceTokenizer, SentencePieceVocab>>::tokenize_list(
            self, text_list,
        )
    }

    fn encode(
        &self,
        text: &str,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<PyTokenizedInput> {
        <Self as PyTokenizer<SentencePieceTokenizer, SentencePieceVocab>>::encode(
            self,
            text,
            max_len,
            truncation_strategy,
            stride,
        )
    }

    fn encode_pair(
        &self,
        text_a: &str,
        text_b: &str,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<PyTokenizedInput> {
        <Self as PyTokenizer<SentencePieceTokenizer, SentencePieceVocab>>::encode_pair(
            self,
            text_a,
            text_b,
            max_len,
            truncation_strategy,
            stride,
        )
    }

    fn encode_list(
        &self,
        text_list: Vec<&str>,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<Vec<PyTokenizedInput>> {
        <Self as PyMultiThreadTokenizer<SentencePieceTokenizer, SentencePieceVocab>>::encode_list(
            self,
            text_list,
            max_len,
            truncation_strategy,
            stride,
        )
    }

    fn encode_pair_list(
        &self,
        text_list: Vec<(&str, &str)>,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<Vec<PyTokenizedInput>> {
        <Self as PyMultiThreadTokenizer<SentencePieceTokenizer, SentencePieceVocab>>::encode_pair_list(self, text_list, max_len, truncation_strategy, stride)
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
    fn new(path: String, do_lower_case: bool, strip_accents: bool) -> Self {
        PyAlbertTokenizer {
            tokenizer: AlbertTokenizer::from_file(path.as_str(), do_lower_case, strip_accents)
                .unwrap(),
        }
    }

    fn tokenize(&self, text: &str) -> PyResult<Vec<String>> {
        <Self as PyTokenizer<AlbertTokenizer, AlbertVocab>>::tokenize(self, text)
    }

    fn tokenize_list(&self, text_list: Vec<&str>) -> PyResult<Vec<Vec<String>>> {
        <Self as PyMultiThreadTokenizer<AlbertTokenizer, AlbertVocab>>::tokenize_list(
            self, text_list,
        )
    }

    fn encode(
        &self,
        text: &str,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<PyTokenizedInput> {
        <Self as PyTokenizer<AlbertTokenizer, AlbertVocab>>::encode(
            self,
            text,
            max_len,
            truncation_strategy,
            stride,
        )
    }

    fn encode_pair(
        &self,
        text_a: &str,
        text_b: &str,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<PyTokenizedInput> {
        <Self as PyTokenizer<AlbertTokenizer, AlbertVocab>>::encode_pair(
            self,
            text_a,
            text_b,
            max_len,
            truncation_strategy,
            stride,
        )
    }

    fn encode_list(
        &self,
        text_list: Vec<&str>,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<Vec<PyTokenizedInput>> {
        <Self as PyMultiThreadTokenizer<AlbertTokenizer, AlbertVocab>>::encode_list(
            self,
            text_list,
            max_len,
            truncation_strategy,
            stride,
        )
    }

    fn encode_pair_list(
        &self,
        text_list: Vec<(&str, &str)>,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<Vec<PyTokenizedInput>> {
        <Self as PyMultiThreadTokenizer<AlbertTokenizer, AlbertVocab>>::encode_pair_list(
            self,
            text_list,
            max_len,
            truncation_strategy,
            stride,
        )
    }
}

#[pyclass(module = "rust_tokenizers")]
struct PyXLNetTokenizer {
    tokenizer: XLNetTokenizer,
}

impl PyTokenizer<XLNetTokenizer, XLNetVocab> for PyXLNetTokenizer {
    fn tokenizer(&self) -> &XLNetTokenizer {
        &self.tokenizer
    }
}

impl PyMultiThreadTokenizer<XLNetTokenizer, XLNetVocab> for PyXLNetTokenizer {}

#[pymethods]
impl PyXLNetTokenizer {
    #[new]
    fn new(path: String, do_lower_case: bool, strip_accents: bool) -> Self {
        PyXLNetTokenizer {
            tokenizer: XLNetTokenizer::from_file(path.as_str(), do_lower_case, strip_accents)
                .unwrap(),
        }
    }

    fn tokenize(&self, text: &str) -> PyResult<Vec<String>> {
        <Self as PyTokenizer<XLNetTokenizer, XLNetVocab>>::tokenize(self, text)
    }

    fn tokenize_list(&self, text_list: Vec<&str>) -> PyResult<Vec<Vec<String>>> {
        <Self as PyMultiThreadTokenizer<XLNetTokenizer, XLNetVocab>>::tokenize_list(self, text_list)
    }

    fn encode(
        &self,
        text: &str,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<PyTokenizedInput> {
        <Self as PyTokenizer<XLNetTokenizer, XLNetVocab>>::encode(
            self,
            text,
            max_len,
            truncation_strategy,
            stride,
        )
    }

    fn encode_pair(
        &self,
        text_a: &str,
        text_b: &str,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<PyTokenizedInput> {
        <Self as PyTokenizer<XLNetTokenizer, XLNetVocab>>::encode_pair(
            self,
            text_a,
            text_b,
            max_len,
            truncation_strategy,
            stride,
        )
    }

    fn encode_list(
        &self,
        text_list: Vec<&str>,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<Vec<PyTokenizedInput>> {
        <Self as PyMultiThreadTokenizer<XLNetTokenizer, XLNetVocab>>::encode_list(
            self,
            text_list,
            max_len,
            truncation_strategy,
            stride,
        )
    }

    fn encode_pair_list(
        &self,
        text_list: Vec<(&str, &str)>,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<Vec<PyTokenizedInput>> {
        <Self as PyMultiThreadTokenizer<XLNetTokenizer, XLNetVocab>>::encode_pair_list(
            self,
            text_list,
            max_len,
            truncation_strategy,
            stride,
        )
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
    fn new(path: String, do_lower_case: bool) -> Self {
        PyT5Tokenizer {
            tokenizer: T5Tokenizer::from_file(path.as_str(), do_lower_case).unwrap(),
        }
    }

    fn tokenize(&self, text: &str) -> PyResult<Vec<String>> {
        <Self as PyTokenizer<T5Tokenizer, T5Vocab>>::tokenize(self, text)
    }

    fn tokenize_list(&self, text_list: Vec<&str>) -> PyResult<Vec<Vec<String>>> {
        <Self as PyMultiThreadTokenizer<T5Tokenizer, T5Vocab>>::tokenize_list(self, text_list)
    }

    fn encode(
        &self,
        text: &str,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<PyTokenizedInput> {
        <Self as PyTokenizer<T5Tokenizer, T5Vocab>>::encode(
            self,
            text,
            max_len,
            truncation_strategy,
            stride,
        )
    }

    fn encode_pair(
        &self,
        text_a: &str,
        text_b: &str,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<PyTokenizedInput> {
        <Self as PyTokenizer<T5Tokenizer, T5Vocab>>::encode_pair(
            self,
            text_a,
            text_b,
            max_len,
            truncation_strategy,
            stride,
        )
    }

    fn encode_list(
        &self,
        text_list: Vec<&str>,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<Vec<PyTokenizedInput>> {
        <Self as PyMultiThreadTokenizer<T5Tokenizer, T5Vocab>>::encode_list(
            self,
            text_list,
            max_len,
            truncation_strategy,
            stride,
        )
    }

    fn encode_pair_list(
        &self,
        text_list: Vec<(&str, &str)>,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<Vec<PyTokenizedInput>> {
        <Self as PyMultiThreadTokenizer<T5Tokenizer, T5Vocab>>::encode_pair_list(
            self,
            text_list,
            max_len,
            truncation_strategy,
            stride,
        )
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
    fn new(path: String, do_lower_case: bool) -> Self {
        PyXLMRobertaTokenizer {
            tokenizer: XLMRobertaTokenizer::from_file(path.as_str(), do_lower_case).unwrap(),
        }
    }

    fn tokenize(&self, text: &str) -> PyResult<Vec<String>> {
        <Self as PyTokenizer<XLMRobertaTokenizer, XLMRobertaVocab>>::tokenize(self, text)
    }

    fn tokenize_list(&self, text_list: Vec<&str>) -> PyResult<Vec<Vec<String>>> {
        <Self as PyMultiThreadTokenizer<XLMRobertaTokenizer, XLMRobertaVocab>>::tokenize_list(
            self, text_list,
        )
    }

    fn encode(
        &self,
        text: &str,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<PyTokenizedInput> {
        <Self as PyTokenizer<XLMRobertaTokenizer, XLMRobertaVocab>>::encode(
            self,
            text,
            max_len,
            truncation_strategy,
            stride,
        )
    }

    fn encode_pair(
        &self,
        text_a: &str,
        text_b: &str,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<PyTokenizedInput> {
        <Self as PyTokenizer<XLMRobertaTokenizer, XLMRobertaVocab>>::encode_pair(
            self,
            text_a,
            text_b,
            max_len,
            truncation_strategy,
            stride,
        )
    }

    fn encode_list(
        &self,
        text_list: Vec<&str>,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<Vec<PyTokenizedInput>> {
        <Self as PyMultiThreadTokenizer<XLMRobertaTokenizer, XLMRobertaVocab>>::encode_list(
            self,
            text_list,
            max_len,
            truncation_strategy,
            stride,
        )
    }

    fn encode_pair_list(
        &self,
        text_list: Vec<(&str, &str)>,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<Vec<PyTokenizedInput>> {
        <Self as PyMultiThreadTokenizer<XLMRobertaTokenizer, XLMRobertaVocab>>::encode_pair_list(
            self,
            text_list,
            max_len,
            truncation_strategy,
            stride,
        )
    }
}

#[pyclass(module = "rust_tokenizers")]
struct PyReformerTokenizer {
    tokenizer: ReformerTokenizer,
}

impl PyTokenizer<ReformerTokenizer, ReformerVocab> for PyReformerTokenizer {
    fn tokenizer(&self) -> &ReformerTokenizer {
        &self.tokenizer
    }
}

impl PyMultiThreadTokenizer<ReformerTokenizer, ReformerVocab> for PyReformerTokenizer {}

#[pymethods]
impl PyReformerTokenizer {
    #[new]
    fn new(path: String, do_lower_case: bool) -> Self {
        PyReformerTokenizer {
            tokenizer: ReformerTokenizer::from_file(path.as_str(), do_lower_case).unwrap(),
        }
    }

    fn tokenize(&self, text: &str) -> PyResult<Vec<String>> {
        <Self as PyTokenizer<ReformerTokenizer, ReformerVocab>>::tokenize(self, text)
    }

    fn tokenize_list(&self, text_list: Vec<&str>) -> PyResult<Vec<Vec<String>>> {
        <Self as PyMultiThreadTokenizer<ReformerTokenizer, ReformerVocab>>::tokenize_list(
            self, text_list,
        )
    }

    fn encode(
        &self,
        text: &str,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<PyTokenizedInput> {
        <Self as PyTokenizer<ReformerTokenizer, ReformerVocab>>::encode(
            self,
            text,
            max_len,
            truncation_strategy,
            stride,
        )
    }

    fn encode_pair(
        &self,
        text_a: &str,
        text_b: &str,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<PyTokenizedInput> {
        <Self as PyTokenizer<ReformerTokenizer, ReformerVocab>>::encode_pair(
            self,
            text_a,
            text_b,
            max_len,
            truncation_strategy,
            stride,
        )
    }

    fn encode_list(
        &self,
        text_list: Vec<&str>,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<Vec<PyTokenizedInput>> {
        <Self as PyMultiThreadTokenizer<ReformerTokenizer, ReformerVocab>>::encode_list(
            self,
            text_list,
            max_len,
            truncation_strategy,
            stride,
        )
    }

    fn encode_pair_list(
        &self,
        text_list: Vec<(&str, &str)>,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<Vec<PyTokenizedInput>> {
        <Self as PyMultiThreadTokenizer<ReformerTokenizer, ReformerVocab>>::encode_pair_list(
            self,
            text_list,
            max_len,
            truncation_strategy,
            stride,
        )
    }
}

#[pyclass(module = "rust_tokenizers")]
struct PyProphetNetTokenizer {
    tokenizer: ProphetNetTokenizer,
}

impl PyTokenizer<ProphetNetTokenizer, ProphetNetVocab> for PyProphetNetTokenizer {
    fn tokenizer(&self) -> &ProphetNetTokenizer {
        &self.tokenizer
    }
}

impl PyMultiThreadTokenizer<ProphetNetTokenizer, ProphetNetVocab> for PyProphetNetTokenizer {}

#[pymethods]
impl PyProphetNetTokenizer {
    #[new]
    fn new(path: String, do_lower_case: bool, strip_accents: bool) -> Self {
        PyProphetNetTokenizer {
            tokenizer: ProphetNetTokenizer::from_file(path.as_str(), do_lower_case, strip_accents)
                .unwrap(),
        }
    }

    fn tokenize(&self, text: &str) -> PyResult<Vec<String>> {
        <Self as PyTokenizer<ProphetNetTokenizer, ProphetNetVocab>>::tokenize(self, text)
    }

    fn tokenize_list(&self, text_list: Vec<&str>) -> PyResult<Vec<Vec<String>>> {
        <Self as PyMultiThreadTokenizer<ProphetNetTokenizer, ProphetNetVocab>>::tokenize_list(
            self, text_list,
        )
    }

    fn encode(
        &self,
        text: &str,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<PyTokenizedInput> {
        <Self as PyTokenizer<ProphetNetTokenizer, ProphetNetVocab>>::encode(
            self,
            text,
            max_len,
            truncation_strategy,
            stride,
        )
    }

    fn encode_pair(
        &self,
        text_a: &str,
        text_b: &str,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<PyTokenizedInput> {
        <Self as PyTokenizer<ProphetNetTokenizer, ProphetNetVocab>>::encode_pair(
            self,
            text_a,
            text_b,
            max_len,
            truncation_strategy,
            stride,
        )
    }

    fn encode_list(
        &self,
        text_list: Vec<&str>,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<Vec<PyTokenizedInput>> {
        <Self as PyMultiThreadTokenizer<ProphetNetTokenizer, ProphetNetVocab>>::encode_list(
            self,
            text_list,
            max_len,
            truncation_strategy,
            stride,
        )
    }

    fn encode_pair_list(
        &self,
        text_list: Vec<(&str, &str)>,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<Vec<PyTokenizedInput>> {
        <Self as PyMultiThreadTokenizer<ProphetNetTokenizer, ProphetNetVocab>>::encode_pair_list(
            self,
            text_list,
            max_len,
            truncation_strategy,
            stride,
        )
    }
}

#[pyclass(module = "rust_tokenizers")]
struct PyPegasusTokenizer {
    tokenizer: PegasusTokenizer,
}

impl PyTokenizer<PegasusTokenizer, PegasusVocab> for PyPegasusTokenizer {
    fn tokenizer(&self) -> &PegasusTokenizer {
        &self.tokenizer
    }
}

impl PyMultiThreadTokenizer<PegasusTokenizer, PegasusVocab> for PyPegasusTokenizer {}

#[pymethods]
impl PyPegasusTokenizer {
    #[new]
    fn new(path: String, do_lower_case: bool) -> Self {
        PyPegasusTokenizer {
            tokenizer: PegasusTokenizer::from_file(path.as_str(), do_lower_case).unwrap(),
        }
    }

    fn tokenize(&self, text: &str) -> PyResult<Vec<String>> {
        <Self as PyTokenizer<PegasusTokenizer, PegasusVocab>>::tokenize(self, text)
    }

    fn tokenize_list(&self, text_list: Vec<&str>) -> PyResult<Vec<Vec<String>>> {
        <Self as PyMultiThreadTokenizer<PegasusTokenizer, PegasusVocab>>::tokenize_list(
            self, text_list,
        )
    }

    fn encode(
        &self,
        text: &str,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<PyTokenizedInput> {
        <Self as PyTokenizer<PegasusTokenizer, PegasusVocab>>::encode(
            self,
            text,
            max_len,
            truncation_strategy,
            stride,
        )
    }

    fn encode_pair(
        &self,
        text_a: &str,
        text_b: &str,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<PyTokenizedInput> {
        <Self as PyTokenizer<PegasusTokenizer, PegasusVocab>>::encode_pair(
            self,
            text_a,
            text_b,
            max_len,
            truncation_strategy,
            stride,
        )
    }

    fn encode_list(
        &self,
        text_list: Vec<&str>,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<Vec<PyTokenizedInput>> {
        <Self as PyMultiThreadTokenizer<PegasusTokenizer, PegasusVocab>>::encode_list(
            self,
            text_list,
            max_len,
            truncation_strategy,
            stride,
        )
    }

    fn encode_pair_list(
        &self,
        text_list: Vec<(&str, &str)>,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<Vec<PyTokenizedInput>> {
        <Self as PyMultiThreadTokenizer<PegasusTokenizer, PegasusVocab>>::encode_pair_list(
            self,
            text_list,
            max_len,
            truncation_strategy,
            stride,
        )
    }
}

#[pyclass(module = "rust_tokenizers")]
struct PyMBart50Tokenizer {
    tokenizer: MBart50Tokenizer,
}

impl PyTokenizer<MBart50Tokenizer, MBart50Vocab> for PyMBart50Tokenizer {
    fn tokenizer(&self) -> &MBart50Tokenizer {
        &self.tokenizer
    }
}

impl PyMultiThreadTokenizer<MBart50Tokenizer, MBart50Vocab> for PyMBart50Tokenizer {}

#[pymethods]
impl PyMBart50Tokenizer {
    #[new]
    fn new(path: String, do_lower_case: bool) -> Self {
        PyMBart50Tokenizer {
            tokenizer: MBart50Tokenizer::from_file(path.as_str(), do_lower_case).unwrap(),
        }
    }

    fn tokenize(&self, text: &str) -> PyResult<Vec<String>> {
        <Self as PyTokenizer<MBart50Tokenizer, MBart50Vocab>>::tokenize(self, text)
    }

    fn tokenize_list(&self, text_list: Vec<&str>) -> PyResult<Vec<Vec<String>>> {
        <Self as PyMultiThreadTokenizer<MBart50Tokenizer, MBart50Vocab>>::tokenize_list(
            self, text_list,
        )
    }

    fn encode(
        &self,
        text: &str,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<PyTokenizedInput> {
        <Self as PyTokenizer<MBart50Tokenizer, MBart50Vocab>>::encode(
            self,
            text,
            max_len,
            truncation_strategy,
            stride,
        )
    }

    fn encode_pair(
        &self,
        text_a: &str,
        text_b: &str,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<PyTokenizedInput> {
        <Self as PyTokenizer<MBart50Tokenizer, MBart50Vocab>>::encode_pair(
            self,
            text_a,
            text_b,
            max_len,
            truncation_strategy,
            stride,
        )
    }

    fn encode_list(
        &self,
        text_list: Vec<&str>,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<Vec<PyTokenizedInput>> {
        <Self as PyMultiThreadTokenizer<MBart50Tokenizer, MBart50Vocab>>::encode_list(
            self,
            text_list,
            max_len,
            truncation_strategy,
            stride,
        )
    }

    fn encode_pair_list(
        &self,
        text_list: Vec<(&str, &str)>,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<Vec<PyTokenizedInput>> {
        <Self as PyMultiThreadTokenizer<MBart50Tokenizer, MBart50Vocab>>::encode_pair_list(
            self,
            text_list,
            max_len,
            truncation_strategy,
            stride,
        )
    }
}

#[pyclass(module = "rust_tokenizers")]
struct PySentencePieceBpeTokenizer {
    tokenizer: SentencePieceBpeTokenizer,
}

impl PyTokenizer<SentencePieceBpeTokenizer, SentencePieceVocab> for PySentencePieceBpeTokenizer {
    fn tokenizer(&self) -> &SentencePieceBpeTokenizer {
        &self.tokenizer
    }
}

impl PyMultiThreadTokenizer<SentencePieceBpeTokenizer, SentencePieceVocab>
    for PySentencePieceBpeTokenizer
{
}

#[pymethods]
impl PySentencePieceBpeTokenizer {
    #[new]
    fn new(path: String, do_lower_case: bool) -> Self {
        PySentencePieceBpeTokenizer {
            tokenizer: SentencePieceBpeTokenizer::from_file(path.as_str(), do_lower_case).unwrap(),
        }
    }

    fn tokenize(&self, text: &str) -> PyResult<Vec<String>> {
        <Self as PyTokenizer<SentencePieceBpeTokenizer, SentencePieceVocab>>::tokenize(self, text)
    }

    fn tokenize_list(&self, text_list: Vec<&str>) -> PyResult<Vec<Vec<String>>> {
        <Self as PyMultiThreadTokenizer<SentencePieceBpeTokenizer, SentencePieceVocab>>::tokenize_list(
            self, text_list,
        )
    }

    fn encode(
        &self,
        text: &str,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<PyTokenizedInput> {
        <Self as PyTokenizer<SentencePieceBpeTokenizer, SentencePieceVocab>>::encode(
            self,
            text,
            max_len,
            truncation_strategy,
            stride,
        )
    }

    fn encode_pair(
        &self,
        text_a: &str,
        text_b: &str,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<PyTokenizedInput> {
        <Self as PyTokenizer<SentencePieceBpeTokenizer, SentencePieceVocab>>::encode_pair(
            self,
            text_a,
            text_b,
            max_len,
            truncation_strategy,
            stride,
        )
    }

    fn encode_list(
        &self,
        text_list: Vec<&str>,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<Vec<PyTokenizedInput>> {
        <Self as PyMultiThreadTokenizer<SentencePieceBpeTokenizer, SentencePieceVocab>>::encode_list(
            self,
            text_list,
            max_len,
            truncation_strategy,
            stride,
        )
    }

    fn encode_pair_list(
        &self,
        text_list: Vec<(&str, &str)>,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<Vec<PyTokenizedInput>> {
        <Self as PyMultiThreadTokenizer<SentencePieceBpeTokenizer, SentencePieceVocab>>::encode_pair_list(
            self,
            text_list,
            max_len,
            truncation_strategy,
            stride,
        )
    }
}

#[pyclass(module = "rust_tokenizers")]
struct PyM2M100Tokenizer {
    tokenizer: M2M100Tokenizer,
}

impl PyTokenizer<M2M100Tokenizer, M2M100Vocab> for PyM2M100Tokenizer {
    fn tokenizer(&self) -> &M2M100Tokenizer {
        &self.tokenizer
    }
}

impl PyMultiThreadTokenizer<M2M100Tokenizer, M2M100Vocab> for PyM2M100Tokenizer {}

#[pymethods]
impl PyM2M100Tokenizer {
    #[new]
    fn new(vocab_path: String, merges_path: String, do_lower_case: bool) -> Self {
        PyM2M100Tokenizer {
            tokenizer: M2M100Tokenizer::from_files(
                vocab_path.as_str(),
                merges_path.as_str(),
                do_lower_case,
            )
            .unwrap(),
        }
    }

    fn tokenize(&self, text: &str) -> PyResult<Vec<String>> {
        <Self as PyTokenizer<M2M100Tokenizer, M2M100Vocab>>::tokenize(self, text)
    }

    fn tokenize_list(&self, text_list: Vec<&str>) -> PyResult<Vec<Vec<String>>> {
        <Self as PyMultiThreadTokenizer<M2M100Tokenizer, M2M100Vocab>>::tokenize_list(
            self, text_list,
        )
    }

    fn encode(
        &self,
        text: &str,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<PyTokenizedInput> {
        <Self as PyTokenizer<M2M100Tokenizer, M2M100Vocab>>::encode(
            self,
            text,
            max_len,
            truncation_strategy,
            stride,
        )
    }

    fn encode_pair(
        &self,
        text_a: &str,
        text_b: &str,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<PyTokenizedInput> {
        <Self as PyTokenizer<M2M100Tokenizer, M2M100Vocab>>::encode_pair(
            self,
            text_a,
            text_b,
            max_len,
            truncation_strategy,
            stride,
        )
    }

    fn encode_list(
        &self,
        text_list: Vec<&str>,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<Vec<PyTokenizedInput>> {
        <Self as PyMultiThreadTokenizer<M2M100Tokenizer, M2M100Vocab>>::encode_list(
            self,
            text_list,
            max_len,
            truncation_strategy,
            stride,
        )
    }

    fn encode_pair_list(
        &self,
        text_list: Vec<(&str, &str)>,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<Vec<PyTokenizedInput>> {
        <Self as PyMultiThreadTokenizer<M2M100Tokenizer, M2M100Vocab>>::encode_pair_list(
            self,
            text_list,
            max_len,
            truncation_strategy,
            stride,
        )
    }
}

#[pyclass(module = "rust_tokenizers")]
struct PyFNetTokenizer {
    tokenizer: FNetTokenizer,
}

impl PyTokenizer<FNetTokenizer, FNetVocab> for PyFNetTokenizer {
    fn tokenizer(&self) -> &FNetTokenizer {
        &self.tokenizer
    }
}

impl PyMultiThreadTokenizer<FNetTokenizer, FNetVocab> for PyFNetTokenizer {}

#[pymethods]
impl PyFNetTokenizer {
    #[new]
    fn new(vocab_path: String, do_lower_case: bool, strip_accents: bool) -> Self {
        PyFNetTokenizer {
            tokenizer: FNetTokenizer::from_file(vocab_path.as_str(), do_lower_case, strip_accents)
                .unwrap(),
        }
    }

    fn tokenize(&self, text: &str) -> PyResult<Vec<String>> {
        <Self as PyTokenizer<FNetTokenizer, FNetVocab>>::tokenize(self, text)
    }

    fn tokenize_list(&self, text_list: Vec<&str>) -> PyResult<Vec<Vec<String>>> {
        <Self as PyMultiThreadTokenizer<FNetTokenizer, FNetVocab>>::tokenize_list(self, text_list)
    }

    fn encode(
        &self,
        text: &str,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<PyTokenizedInput> {
        <Self as PyTokenizer<FNetTokenizer, FNetVocab>>::encode(
            self,
            text,
            max_len,
            truncation_strategy,
            stride,
        )
    }

    fn encode_pair(
        &self,
        text_a: &str,
        text_b: &str,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<PyTokenizedInput> {
        <Self as PyTokenizer<FNetTokenizer, FNetVocab>>::encode_pair(
            self,
            text_a,
            text_b,
            max_len,
            truncation_strategy,
            stride,
        )
    }

    fn encode_list(
        &self,
        text_list: Vec<&str>,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<Vec<PyTokenizedInput>> {
        <Self as PyMultiThreadTokenizer<FNetTokenizer, FNetVocab>>::encode_list(
            self,
            text_list,
            max_len,
            truncation_strategy,
            stride,
        )
    }

    fn encode_pair_list(
        &self,
        text_list: Vec<(&str, &str)>,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<Vec<PyTokenizedInput>> {
        <Self as PyMultiThreadTokenizer<FNetTokenizer, FNetVocab>>::encode_pair_list(
            self,
            text_list,
            max_len,
            truncation_strategy,
            stride,
        )
    }
}

#[pyclass(module = "rust_tokenizers")]
struct PyDeBertaTokenizer {
    tokenizer: DeBERTaTokenizer,
}

impl PyTokenizer<DeBERTaTokenizer, DeBERTaVocab> for PyDeBertaTokenizer {
    fn tokenizer(&self) -> &DeBERTaTokenizer {
        &self.tokenizer
    }
}

impl PyMultiThreadTokenizer<DeBERTaTokenizer, DeBERTaVocab> for PyDeBertaTokenizer {}

#[pymethods]
impl PyDeBertaTokenizer {
    #[new]
    fn new(vocab_path: String, merges_path: String, do_lower_case: bool) -> Self {
        PyDeBertaTokenizer {
            tokenizer: DeBERTaTokenizer::from_file(
                vocab_path.as_str(),
                merges_path.as_str(),
                do_lower_case,
            )
            .unwrap(),
        }
    }

    fn tokenize(&self, text: &str) -> PyResult<Vec<String>> {
        <Self as PyTokenizer<DeBERTaTokenizer, DeBERTaVocab>>::tokenize(self, text)
    }

    fn tokenize_list(&self, text_list: Vec<&str>) -> PyResult<Vec<Vec<String>>> {
        <Self as PyMultiThreadTokenizer<DeBERTaTokenizer, DeBERTaVocab>>::tokenize_list(
            self, text_list,
        )
    }

    fn encode(
        &self,
        text: &str,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<PyTokenizedInput> {
        <Self as PyTokenizer<DeBERTaTokenizer, DeBERTaVocab>>::encode(
            self,
            text,
            max_len,
            truncation_strategy,
            stride,
        )
    }

    fn encode_pair(
        &self,
        text_a: &str,
        text_b: &str,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<PyTokenizedInput> {
        <Self as PyTokenizer<DeBERTaTokenizer, DeBERTaVocab>>::encode_pair(
            self,
            text_a,
            text_b,
            max_len,
            truncation_strategy,
            stride,
        )
    }

    fn encode_list(
        &self,
        text_list: Vec<&str>,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<Vec<PyTokenizedInput>> {
        <Self as PyMultiThreadTokenizer<DeBERTaTokenizer, DeBERTaVocab>>::encode_list(
            self,
            text_list,
            max_len,
            truncation_strategy,
            stride,
        )
    }

    fn encode_pair_list(
        &self,
        text_list: Vec<(&str, &str)>,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<Vec<PyTokenizedInput>> {
        <Self as PyMultiThreadTokenizer<DeBERTaTokenizer, DeBERTaVocab>>::encode_pair_list(
            self,
            text_list,
            max_len,
            truncation_strategy,
            stride,
        )
    }
}

#[pyclass(module = "rust_tokenizers")]
struct PyDeBertaV2Tokenizer {
    tokenizer: DeBERTaV2Tokenizer,
}

impl PyTokenizer<DeBERTaV2Tokenizer, DeBERTaV2Vocab> for PyDeBertaV2Tokenizer {
    fn tokenizer(&self) -> &DeBERTaV2Tokenizer {
        &self.tokenizer
    }
}

impl PyMultiThreadTokenizer<DeBERTaV2Tokenizer, DeBERTaV2Vocab> for PyDeBertaV2Tokenizer {}

#[pymethods]
impl PyDeBertaV2Tokenizer {
    #[new]
    fn new(
        vocab_path: String,
        do_lower_case: bool,
        strip_accents: bool,
        add_prefix_space: bool,
    ) -> Self {
        PyDeBertaV2Tokenizer {
            tokenizer: DeBERTaV2Tokenizer::from_file(
                vocab_path.as_str(),
                do_lower_case,
                strip_accents,
                add_prefix_space,
            )
            .unwrap(),
        }
    }

    fn tokenize(&self, text: &str) -> PyResult<Vec<String>> {
        <Self as PyTokenizer<DeBERTaV2Tokenizer, DeBERTaV2Vocab>>::tokenize(self, text)
    }

    fn tokenize_list(&self, text_list: Vec<&str>) -> PyResult<Vec<Vec<String>>> {
        <Self as PyMultiThreadTokenizer<DeBERTaV2Tokenizer, DeBERTaV2Vocab>>::tokenize_list(
            self, text_list,
        )
    }

    fn encode(
        &self,
        text: &str,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<PyTokenizedInput> {
        <Self as PyTokenizer<DeBERTaV2Tokenizer, DeBERTaV2Vocab>>::encode(
            self,
            text,
            max_len,
            truncation_strategy,
            stride,
        )
    }

    fn encode_pair(
        &self,
        text_a: &str,
        text_b: &str,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<PyTokenizedInput> {
        <Self as PyTokenizer<DeBERTaV2Tokenizer, DeBERTaV2Vocab>>::encode_pair(
            self,
            text_a,
            text_b,
            max_len,
            truncation_strategy,
            stride,
        )
    }

    fn encode_list(
        &self,
        text_list: Vec<&str>,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<Vec<PyTokenizedInput>> {
        <Self as PyMultiThreadTokenizer<DeBERTaV2Tokenizer, DeBERTaV2Vocab>>::encode_list(
            self,
            text_list,
            max_len,
            truncation_strategy,
            stride,
        )
    }

    fn encode_pair_list(
        &self,
        text_list: Vec<(&str, &str)>,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<Vec<PyTokenizedInput>> {
        <Self as PyMultiThreadTokenizer<DeBERTaV2Tokenizer, DeBERTaV2Vocab>>::encode_pair_list(
            self,
            text_list,
            max_len,
            truncation_strategy,
            stride,
        )
    }
}

#[pyclass(module = "rust_tokenizers")]
struct PyNLLBTokenizer {
    tokenizer: NLLBTokenizer,
}

impl PyTokenizer<NLLBTokenizer, NLLBVocab> for PyNLLBTokenizer {
    fn tokenizer(&self) -> &NLLBTokenizer {
        &self.tokenizer
    }
}

impl PyMultiThreadTokenizer<NLLBTokenizer, NLLBVocab> for PyNLLBTokenizer {}

#[pymethods]
impl PyNLLBTokenizer {
    #[new]
    fn new(vocab_path: String, merges_path: String, special_token_map: String) -> Self {
        PyNLLBTokenizer {
            tokenizer: NLLBTokenizer::from_files(
                vocab_path.as_str(),
                merges_path.as_str(),
                special_token_map.as_str(),
            )
                .unwrap(),
        }
    }

    fn tokenize(&self, text: &str) -> PyResult<Vec<String>> {
        <Self as PyTokenizer<NLLBTokenizer, NLLBVocab>>::tokenize(self, text)
    }

    fn tokenize_list(&self, text_list: Vec<&str>) -> PyResult<Vec<Vec<String>>> {
        <Self as PyMultiThreadTokenizer<NLLBTokenizer, NLLBVocab>>::tokenize_list(
            self, text_list,
        )
    }

    fn encode(
        &self,
        text: &str,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<PyTokenizedInput> {
        <Self as PyTokenizer<NLLBTokenizer, NLLBVocab>>::encode(
            self,
            text,
            max_len,
            truncation_strategy,
            stride,
        )
    }

    fn encode_pair(
        &self,
        text_a: &str,
        text_b: &str,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<PyTokenizedInput> {
        <Self as PyTokenizer<NLLBTokenizer, NLLBVocab>>::encode_pair(
            self,
            text_a,
            text_b,
            max_len,
            truncation_strategy,
            stride,
        )
    }

    fn encode_list(
        &self,
        text_list: Vec<&str>,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<Vec<PyTokenizedInput>> {
        <Self as PyMultiThreadTokenizer<NLLBTokenizer, NLLBVocab>>::encode_list(
            self,
            text_list,
            max_len,
            truncation_strategy,
            stride,
        )
    }

    fn encode_pair_list(
        &self,
        text_list: Vec<(&str, &str)>,
        max_len: usize,
        truncation_strategy: &str,
        stride: usize,
    ) -> PyResult<Vec<PyTokenizedInput>> {
        <Self as PyMultiThreadTokenizer<NLLBTokenizer, NLLBVocab>>::encode_pair_list(
            self,
            text_list,
            max_len,
            truncation_strategy,
            stride,
        )
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
    m.add_class::<PySentencePieceBpeTokenizer>()?;
    m.add_class::<PyAlbertTokenizer>()?;
    m.add_class::<PyT5Tokenizer>()?;
    m.add_class::<PyXLMRobertaTokenizer>()?;
    m.add_class::<PyXLNetTokenizer>()?;
    m.add_class::<PyReformerTokenizer>()?;
    m.add_class::<PyProphetNetTokenizer>()?;
    m.add_class::<PyPegasusTokenizer>()?;
    m.add_class::<PyMBart50Tokenizer>()?;
    m.add_class::<PyM2M100Tokenizer>()?;
    m.add_class::<PyFNetTokenizer>()?;
    m.add_class::<PyDeBertaTokenizer>()?;
    m.add_class::<PyDeBertaV2Tokenizer>()?;
    m.add_class::<PyNLLBTokenizer>()?;
    Ok(())
}
