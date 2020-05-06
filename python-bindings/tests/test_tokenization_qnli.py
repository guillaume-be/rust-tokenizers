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
from zipfile import ZipFile

import pytest
import requests
from transformers.data.processors.glue import QnliProcessor
from transformers.file_utils import get_from_cache
from transformers.tokenization_bert import BertTokenizer
from transformers.tokenization_distilbert import DistilBertTokenizer
from rust_tokenizers import PyBertTokenizer


@pytest.mark.slow
class TestTokenizationQNLI:
    def setup_class(self):
        self.processor = QnliProcessor()
        self.test_dir = Path(tempfile.mkdtemp())
        qnli_url = 'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FQNLIv2.zip?alt=media&token=6fdcf570-0fc5-4631-8456-9505272d1601'
        contents = requests.get(qnli_url)
        (self.test_dir / 'QNLI.zip').open('wb').write(contents.content)
        with ZipFile(self.test_dir / 'QNLI.zip', 'r') as zipObj:
            zipObj.extractall(self.test_dir)
        self.examples = self.processor.get_train_examples(self.test_dir / 'QNLI')
        self.test_dir = Path(tempfile.mkdtemp())

    def test_tokenization_bert(self):
        # Given
        self.base_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True,
                                                            cache_dir=self.test_dir)
        self.rust_tokenizer = PyBertTokenizer(
            get_from_cache(self.base_tokenizer.pretrained_vocab_files_map['vocab_file']['bert-base-uncased']),
            do_lower_case=True)
        output_baseline = []
        for example in self.examples:
            output_baseline.append(self.base_tokenizer.encode_plus(example.text_a,
                                                                   text_pair=example.text_b,
                                                                   add_special_tokens=True,
                                                                   return_overflowing_tokens=True,
                                                                   return_special_tokens_mask=True,
                                                                   max_length=128))

        # When
        output_rust = self.rust_tokenizer.encode_pair_list(
            [(example.text_a, example.text_b) for example in self.examples],
            max_len=128,
            truncation_strategy='longest_first',
            stride=0)

        # Then
        for rust, baseline in zip(output_rust, output_baseline):
            assert rust.token_ids == baseline['input_ids'], f'Difference in tokenization for {self.rust_tokenizer.__class__}: \n ' \
                                                            f'Rust: {rust.token_ids} \n' \
                                                            f' Python {baseline["input_ids"]}'
            assert (rust.segment_ids == baseline['token_type_ids'])
            assert (rust.special_tokens_mask == baseline['special_tokens_mask'])

    def test_tokenization_distilbert(self):
        # Given
        self.base_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased', do_lower_case=False,
                                                                  cache_dir=self.test_dir)
        self.rust_tokenizer = PyBertTokenizer(
            get_from_cache(self.base_tokenizer.pretrained_vocab_files_map['vocab_file']['distilbert-base-cased']),
            do_lower_case=False)
        output_baseline = []
        for example in self.examples:
            output_baseline.append(self.base_tokenizer.encode_plus(example.text_a,
                                                                   text_pair=example.text_b,
                                                                   add_special_tokens=True,
                                                                   return_overflowing_tokens=True,
                                                                   return_special_tokens_mask=True,
                                                                   max_length=128))

        # When
        output_rust = self.rust_tokenizer.encode_pair_list(
            [(example.text_a, example.text_b) for example in self.examples],
            max_len=128,
            truncation_strategy='longest_first',
            stride=0)

        # Then
        for rust, baseline in zip(output_rust, output_baseline):
            assert rust.token_ids == baseline['input_ids'], f'Difference in tokenization for {self.rust_tokenizer.__class__}: \n ' \
                                                            f'Rust: {rust.token_ids} \n' \
                                                            f' Python {baseline["input_ids"]}'
