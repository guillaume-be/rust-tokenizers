# Copyright 2018 The Open AI Team Authors, The Google AI Language Team Authors
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

import requests
from transformers.data.processors.glue import Sst2Processor
from transformers.file_utils import get_from_cache
from transformers import CTRLTokenizer
from rust_tokenizers import PyCtrlTokenizer


class TestBenchmarkCtrl:
    def setup_class(self):
        self.processor = Sst2Processor()
        self.test_dir = Path(tempfile.mkdtemp())
        sst2_url = 'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FSST-2.zip?alt=media&token=aabc5f6b-e466-44a2-b9b4-cf6337f84ac8'
        contents = requests.get(sst2_url)
        (self.test_dir / 'SST-2.zip').open('wb').write(contents.content)
        with ZipFile(self.test_dir / 'SST-2.zip', 'r') as zipObj:
            zipObj.extractall(self.test_dir)
        self.examples = self.processor.get_train_examples(self.test_dir / 'SST-2')
        self.base_tokenizer = CTRLTokenizer.from_pretrained('ctrl',
                                                            do_lower_case=False,
                                                            cache_dir=self.test_dir)
        self.rust_tokenizer = PyCtrlTokenizer(
            get_from_cache(self.base_tokenizer.pretrained_vocab_files_map['vocab_file']['ctrl']),
            get_from_cache(self.base_tokenizer.pretrained_vocab_files_map['merges_file']['ctrl']),
            do_lower_case=False)

    def setup_python_tokenizer(self):
        self.base_tokenizer = CTRLTokenizer.from_pretrained('ctrl',
                                                            do_lower_case=False,
                                                            cache_dir=self.test_dir)

    def setup_rust_tokenizer(self):
        self.rust_tokenizer = PyCtrlTokenizer(
            get_from_cache(self.base_tokenizer.pretrained_vocab_files_map['vocab_file']['ctrl']),
            get_from_cache(self.base_tokenizer.pretrained_vocab_files_map['merges_file']['ctrl']),
            do_lower_case=False)

    def python_ctrl_tokenizer(self):
        output_baseline = []
        for example in self.examples:
            output_baseline.append(self.base_tokenizer.encode_plus(example.text_a,
                                                                   add_special_tokens=True,
                                                                   return_overflowing_tokens=True,
                                                                   return_special_tokens_mask=True,
                                                                   max_length=128))

    def rust_ctrl_tokenizer_single_threaded(self):
        output_baseline = []
        for example in self.examples:
            output_baseline.append(self.rust_tokenizer.encode(example.text_a,
                                                              max_len=128,
                                                              truncation_strategy='longest_first',
                                                              stride=0))

    def rust_ctrl_tokenizer_multi_threaded(self):
        self.rust_tokenizer.encode_list([example.text_a for example in self.examples],
                                        max_len=128,
                                        truncation_strategy='longest_first',
                                        stride=0)

    def test_python_ctrl_tokenizer_single_threaded(self, benchmark):
        benchmark.pedantic(self.python_ctrl_tokenizer, setup=self.setup_python_tokenizer, iterations=1, rounds=3)

    def test_rust_ctrl_tokenizer_single_threaded(self, benchmark):
        benchmark.pedantic(self.rust_ctrl_tokenizer_single_threaded, setup=self.setup_rust_tokenizer, iterations=1,
                           rounds=3)

    def test_rust_ctrl_tokenizer_multi_threaded(self, benchmark):
        benchmark.pedantic(self.rust_ctrl_tokenizer_multi_threaded, setup=self.setup_rust_tokenizer, iterations=1,
                           rounds=3)
