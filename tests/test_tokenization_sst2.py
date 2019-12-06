import tempfile
from pathlib import Path

import pytest
from transformers.data.processors.glue import Sst2Processor
from transformers.file_utils import get_from_cache
from transformers.tokenization_bert import BertTokenizer
from transformers.tokenization_distilbert import DistilBertTokenizer
from rust_transformers import PyBertTokenizer
import os


@pytest.mark.slow
class TestTokenizationSST2:
    def setup_class(self):
        self.processor = Sst2Processor()
        # Note: these tests do not download automatically test datasets. Please download them manually and update your
        # environment variables accordingly
        self.examples = self.processor.get_train_examples(os.environ["SST2_PATH"])
        self.test_dir = Path(tempfile.mkdtemp())

    def test_tokenization_bert(self):
        # Given
        self.base_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True,
                                                            cache_dir=self.test_dir)
        self.rust_tokenizer = PyBertTokenizer(
            get_from_cache(self.base_tokenizer.pretrained_vocab_files_map['vocab_file']['bert-base-uncased']))
        output_baseline = []
        for example in self.examples:
            output_baseline.append(self.base_tokenizer.encode_plus(example.text_a,
                                                                   add_special_tokens=True,
                                                                   return_overflowing_tokens=True,
                                                                   return_special_tokens_mask=True,
                                                                   max_length=128))

        # When
        output_rust = self.rust_tokenizer.encode_list([example.text_a for example in self.examples],
                                                      max_len=128,
                                                      truncation_strategy='longest_first',
                                                      stride=0)

        # Then
        for rust, baseline in zip(output_rust, output_baseline):
            assert (rust.token_ids == baseline['input_ids'])
            assert (rust.segment_ids == baseline['token_type_ids'])
            assert (rust.special_tokens_mask == baseline['special_tokens_mask'])

    def test_tokenization_distilbert(self):
        # Given
        self.base_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True,
                                                                  cache_dir=self.test_dir)
        self.rust_tokenizer = PyBertTokenizer(
            get_from_cache(self.base_tokenizer.pretrained_vocab_files_map['vocab_file']['distilbert-base-uncased']))
        output_baseline = []
        for example in self.examples:
            output_baseline.append(self.base_tokenizer.encode_plus(example.text_a,
                                                                   add_special_tokens=True,
                                                                   return_overflowing_tokens=True,
                                                                   return_special_tokens_mask=True,
                                                                   max_length=128))

        # When
        output_rust = self.rust_tokenizer.encode_list([example.text_a for example in self.examples],
                                                      max_len=128,
                                                      truncation_strategy='longest_first',
                                                      stride=0)

        # Then
        for rust, baseline in zip(output_rust, output_baseline):
            assert (rust.token_ids == baseline['input_ids'])
            assert (rust.segment_ids == baseline['token_type_ids'])
            assert (rust.special_tokens_mask == baseline['special_tokens_mask'])
