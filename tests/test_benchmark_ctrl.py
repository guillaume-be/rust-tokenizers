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
import gc
from transformers.file_utils import get_from_cache
from transformers.tokenization_ctrl import CTRLTokenizer
from rust_transformers import PyCtrlTokenizer
from transformers.modeling_ctrl import CTRLModel
import torch


class TestBenchmarkCTRL:
    def setup_class(self):
        self.use_gpu = torch.cuda.is_available()
        self.test_dir = Path(tempfile.mkdtemp())

        self.base_tokenizer = CTRLTokenizer.from_pretrained('ctrl', do_lower_case=True,
                                                            cache_dir=self.test_dir)
        self.rust_tokenizer = PyCtrlTokenizer(
            get_from_cache(self.base_tokenizer.pretrained_vocab_files_map['vocab_file']['ctrl']),
            get_from_cache(self.base_tokenizer.pretrained_vocab_files_map['merges_file']['ctrl'])
        )
        self.model = CTRLModel.from_pretrained('ctrl',
                                               output_attentions=False).eval()
        if self.use_gpu:
            self.model.cuda()
        self.sentence_list = ['For instance, on the planet Earth, man had always assumed that he was more intelligent '
                              'than dolphins because he had achieved so much—the wheel, New York, wars and so on—whilst'
                              ' all the dolphins had ever done was muck about in the water having a good time. But '
                              'conversely, the dolphins had always believed that they were far more intelligent than '
                              'man—for precisely the same reasons.'] * 1

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

    def test_ctrl_baseline(self, benchmark):
        benchmark(self.baseline_batch)

    def test_ctrl_rust_single_threaded(self, benchmark):
        benchmark(self.rust_batch_single_threaded)

    def teardown_class(self):
        self.model = None
        self.base_tokenizer = None
        self.rust_tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()
