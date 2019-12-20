# Copyright 2018 The HuggingFace Inc. team.
# Copyright 2019 Guillaume Becquin
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tempfile
from pathlib import Path
import pytest
from transformers.data.processors.glue import Sst2Processor
from transformers.file_utils import get_from_cache
from transformers.tokenization_bert import BertTokenizer
from transformers.tokenization_distilbert import DistilBertTokenizer
from transformers.tokenization_ctrl import CTRLTokenizer
from transformers.tokenization_gpt2 import GPT2Tokenizer
from transformers.tokenization_roberta import RobertaTokenizer
from rust_transformers import PyBertTokenizer, PyCtrlTokenizer, PyGpt2Tokenizer, PyRobertaTokenizer
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

    def test_tokenization_ctrl(self):
        # Given
        self.base_tokenizer = CTRLTokenizer.from_pretrained('ctrl', do_lower_case=True,
                                                            cache_dir=self.test_dir)
        self.rust_tokenizer = PyCtrlTokenizer(
            get_from_cache(self.base_tokenizer.pretrained_vocab_files_map['vocab_file']['ctrl']),
            get_from_cache(self.base_tokenizer.pretrained_vocab_files_map['merges_file']['ctrl'])
        )
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

    def test_tokenization_gpt2(self):
        # Given
        self.base_tokenizer = GPT2Tokenizer.from_pretrained('gpt2', do_lower_case=True,
                                                            cache_dir=self.test_dir)
        self.rust_tokenizer = PyGpt2Tokenizer(
            get_from_cache(self.base_tokenizer.pretrained_vocab_files_map['vocab_file']['gpt2']),
            get_from_cache(self.base_tokenizer.pretrained_vocab_files_map['merges_file']['gpt2'])
        )
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

    def test_tokenization_roberta(self):
        # Given
        self.base_tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True,
                                                               cache_dir=self.test_dir)
        self.rust_tokenizer = PyRobertaTokenizer(
            get_from_cache(self.base_tokenizer.pretrained_vocab_files_map['vocab_file']['roberta-base']),
            get_from_cache(self.base_tokenizer.pretrained_vocab_files_map['merges_file']['roberta-base'])
        )
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
