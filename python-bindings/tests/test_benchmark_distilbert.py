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
import math
import tempfile
from pathlib import Path
import gc
from transformers.file_utils import get_from_cache
from transformers import DistilBertTokenizer
from rust_tokenizers import PyBertTokenizer
from transformers import DistilBertForSequenceClassification
import torch
from timeit import default_timer as timer


class TestBenchmarkDistilBert:
    def setup_class(self):
        self.use_gpu = torch.cuda.is_available()
        self.test_dir = Path(tempfile.mkdtemp())

        self.base_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True,
                                                                  cache_dir=self.test_dir)
        self.rust_tokenizer = PyBertTokenizer(
            get_from_cache(self.base_tokenizer.pretrained_vocab_files_map['vocab_file']['distilbert-base-uncased']),
            do_lower_case=True,
            strip_accents=True)
        self.model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased',
                                                                         output_attentions=False).eval()
        if self.use_gpu:
            self.model.cuda()
        self.sentence_list = ['For instance, on the planet Earth, man had always assumed that he was more intelligent '
                              'than dolphins because he had achieved so much—the wheel, New York, wars and so on—whilst'
                              ' all the dolphins had ever done was muck about in the water having a good time. But '
                              'conversely, the dolphins had always believed that they were far more intelligent than '
                              'man—for precisely the same reasons.'] * 64

        # Pre-allocate GPU memory
        tokens_list = [self.base_tokenizer.tokenize(sentence) for sentence in self.sentence_list]
        features = [self.base_tokenizer.convert_tokens_to_ids(tokens) for tokens in tokens_list]
        features = [self.base_tokenizer.prepare_for_model(input, None, add_special_tokens=True, max_length=128) for
                    input
                    in features]
        all_input_ids = torch.tensor([f['input_ids'] for f in features], dtype=torch.long)

        if self.use_gpu:
            all_input_ids = all_input_ids.cuda()

        with torch.no_grad():
            _ = self.model(all_input_ids)[0].cpu().numpy()

    def setup_base_tokenizer(self):
        self.base_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True,
                                                                  cache_dir=self.test_dir)

    def setup_rust_tokenizer(self):
        self.rust_tokenizer = PyBertTokenizer(
            get_from_cache(self.base_tokenizer.pretrained_vocab_files_map['vocab_file']['distilbert-base-uncased']),
            do_lower_case=True,
            strip_accents=True
        )

    def baseline_batch(self):
        tokens_list = [self.base_tokenizer.tokenize(sentence) for sentence in self.sentence_list]
        features = [self.base_tokenizer.convert_tokens_to_ids(tokens) for tokens in tokens_list]
        features = [self.base_tokenizer.prepare_for_model(input,
                                                          None,
                                                          add_special_tokens=True,
                                                          max_length=128) for input in features]
        all_input_ids = torch.tensor([f['input_ids'] for f in features], dtype=torch.long)
        if self.use_gpu:
            all_input_ids = all_input_ids.cuda()
        with torch.no_grad():
            output = self.model(all_input_ids)[0].cpu().numpy()
        return output

    def rust_batch_single_threaded(self):
        features = [self.rust_tokenizer.encode(sentence,
                                               max_len=128,
                                               truncation_strategy='longest_first',
                                               stride=0) for sentence in self.sentence_list]
        all_input_ids = torch.tensor([f.token_ids for f in features], dtype=torch.long)
        if self.use_gpu:
            all_input_ids = all_input_ids.cuda()
        with torch.no_grad():
            output = self.model(all_input_ids)[0].cpu().numpy()
        return output

    def rust_batch_multi_threaded(self):
        features = self.rust_tokenizer.encode_list(self.sentence_list,
                                                   max_len=128,
                                                   truncation_strategy='longest_first',
                                                   stride=0)
        all_input_ids = torch.tensor([f.token_ids for f in features], dtype=torch.long)
        if self.use_gpu:
            all_input_ids = all_input_ids.cuda()
        with torch.no_grad():
            output = self.model(all_input_ids)[0].cpu().numpy()
        return output

    def test_distilbert_baseline(self):
        values = []
        for i in range(10):
            self.setup_base_tokenizer()
            t0 = timer()
            self.baseline_batch()
            t1 = timer()
            values.append((t1 - t0) * 1000)
        mean = sum(values) / len(values)
        std_dev = math.sqrt(sum([(value - mean) ** 2 for value in values])) / (len(values) - 1)
        print(f'baseline - mean: {mean:.2f}, std. dev: {std_dev:.2f}')

    def test_distilbert_rust_single_threaded(self):
        values = []
        for i in range(10):
            self.setup_rust_tokenizer()
            t0 = timer()
            self.rust_batch_single_threaded()
            t1 = timer()
            values.append((t1 - t0) * 1000)
        mean = sum(values) / len(values)
        std_dev = math.sqrt(sum([(value - mean) ** 2 for value in values])) / (len(values) - 1)
        print(f'rust single thread - mean: {mean:.2f}, std. dev: {std_dev:.2f}')

    def test_distilbert_rust_multi_threaded(self):
        values = []
        for i in range(10):
            self.setup_rust_tokenizer()
            t0 = timer()
            self.rust_batch_multi_threaded()
            t1 = timer()
            values.append((t1 - t0) * 1000)
        mean = sum(values) / len(values)
        std_dev = math.sqrt(sum([(value - mean) ** 2 for value in values])) / (len(values) - 1)
        print(f'rust multi threaded - mean: {mean:.2f}, std. dev: {std_dev:.2f}')

    def teardown_class(self):
        self.model = None
        self.base_tokenizer = None
        self.rust_tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()
