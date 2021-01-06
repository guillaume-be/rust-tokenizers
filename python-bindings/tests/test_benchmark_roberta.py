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
from transformers import RobertaTokenizer
from rust_tokenizers import PyRobertaTokenizer
from transformers import RobertaModel
import torch
from timeit import default_timer as timer


class TestBenchmarkRoberta:
    def setup_class(self):
        self.use_gpu = torch.cuda.is_available()
        self.test_dir = Path(tempfile.mkdtemp())

        self.base_tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True,
                                                               cache_dir=self.test_dir)
        self.rust_tokenizer = PyRobertaTokenizer(
            get_from_cache(self.base_tokenizer.pretrained_vocab_files_map['vocab_file']['roberta-base']),
            get_from_cache(self.base_tokenizer.pretrained_vocab_files_map['merges_file']['roberta-base']),
            do_lower_case=True,
            add_prefix_space=True
        )
        self.model = RobertaModel.from_pretrained('roberta-base',
                                                  output_attentions=False).eval()
        if self.use_gpu:
            self.model.cuda()
        #     Extracted from https://en.wikipedia.org/wiki/Deep_learning
        self.sentence_list = [
            'Deep learning (also known as deep structured learning or hierarchical learning) is part of a broader family of machine learning methods based on artificial neural networks.Learning can be supervised, semi-supervised or unsupervised.',
            'Deep learning is a class of machine learning algorithms that[11](pp199–200) uses multiple layers to progressively extract higher level features from the raw input.',
            'For example, in image processing, lower layers may identify edges, while higher layers may identify the concepts relevant to a human such as digits or letters or faces.',
            'Most modern deep learning models are based on artificial neural networks, specifically, Convolutional Neural Networks (CNN)s, although they can also include propositional formulas organized layer-wise in deep generative models.',
            'In deep learning, each level learns to transform its input data into a slightly more abstract and composite representation.',
            'In an image recognition application, the raw input may be a matrix of pixels; the first representational layer may abstract the pixels and encode edges; the second layer may compose and encode arrangements of edges;',
            'he third layer may encode a nose and eyes; and the fourth layer may recognize that the image contains a face. Importantly, a deep learning process can learn which features to optimally place in which level on its own.',
            '(Of course, this does not completely eliminate the need for hand-tuning; for example, varying numbers of layers and layer sizes can provide different degrees of abstraction.)[',
            'The word "deep" in "deep learning" refers to the number of layers through which the data is transformed. More precisely, deep learning systems have a substantial credit assignment path (CAP) depth. The CAP is the chain of transformations from input to output.',
            'CAPs describe potentially causal connections between input and output. For a feedforward neural network, the depth of the CAPs is that of the network and is the number of hidden layers plus one (as the output layer is also parameterized).',
            'For recurrent neural networks, in which a signal may propagate through a layer more than once, the CAP depth is potentially unlimited.[2] No universally agreed upon threshold of depth divides shallow learning from deep learning.',
            'CAP of depth 2 has been shown to be a universal approximator in the sense that it can emulate any function.[14] Beyond that, more layers do not add to the function approximator ability of the network.',
            'Deep models (CAP > 2) are able to extract better features than shallow models and hence, extra layers help in learning the features effectively. Deep learning architectures can be constructed with a greedy layer-by-layer method.',
            'Deep learning helps to disentangle these abstractions and pick out which features improve performance.[1]. For supervised learning tasks, deep learning methods eliminate feature engineering, by translating the data into compact intermediate representations',
            'Deep learning algorithms can be applied to unsupervised learning tasks. This is an important benefit because unlabeled data are more abundant than the labeled data. Examples of deep structures that can be trained in an unsupervised manner are neural history compressors and deep belief networks.',
            'Deep neural networks are generally interpreted in terms of the universal approximation theorem or probabilistic inference. The classic universal approximation theorem concerns the capacity of feedforward neural networks with a single hidden layer of finite size to approximate continuous functions.',
            'In 1989, the first proof was published by George Cybenko for sigmoid activation functions and was generalised to feed-forward multi-layer architectures in 1991 by Kurt Hornik.Recent work also showed that universal approximation also holds for non-bounded activation functions such as the rectified linear unit.',
            'he universal approximation theorem for deep neural networks concerns the capacity of networks with bounded width but the depth is allowed to grow. Lu et al. proved that if the width of a deep neural network with ReLU activation is strictly larger than the input dimension, then the network can approximate any Lebesgue integrable function',
            'The probabilistic interpretation[24] derives from the field of machine learning. It features inference, as well as the optimization concepts of training and testing, related to fitting and generalization, respectively',
            'More specifically, the probabilistic interpretation considers the activation nonlinearity as a cumulative distribution function. The probabilistic interpretation led to the introduction of dropout as regularizer in neural networks.',
            'The probabilistic interpretation was introduced by researchers including Hopfield, Widrow and Narendra and popularized in surveys such as the one by Bishop. The term Deep Learning was introduced to the machine learning community by Rina Dechter in 1986',
            'The first general, working learning algorithm for supervised, deep, feedforward, multilayer perceptrons was published by Alexey Ivakhnenko and Lapa in 1965.[32] A 1971 paper described already a deep network with 8 layers trained by the group method of data handling algorithm.',
            'Other deep learning working architectures, specifically those built for computer vision, began with the Neocognitron introduced by Kunihiko Fukushima in 1980.[34] In 1989, Yann LeCun et al. applied the standard backpropagation algorithm',
            'By 1991 such systems were used for recognizing isolated 2-D hand-written digits, while recognizing 3-D objects was done by matching 2-D images with a handcrafted 3-D object model. Weng et al. suggested that a human brain does not use a monolithic 3-D object model and in 1992 they published Cresceptron',
            'Because it directly used natural images, Cresceptron started the beginning of general-purpose visual learning for natural 3D worlds. Cresceptron is a cascade of layers similar to Neocognitron. But while Neocognitron required a human programmer to hand-merge features, Cresceptron learned an open number of features in each layer without supervision',
            'Cresceptron segmented each learned object from a cluttered scene through back-analysis through the network. Max pooling, now often adopted by deep neural networks (e.g. ImageNet tests), was first used in Cresceptron to reduce the position resolution by a factor of (2x2) to 1 through the cascade for better generalization',
            'In 1994, André de Carvalho, together with Mike Fairhurst and David Bisset, published experimental results of a multi-layer boolean neural network, also known as a weightless neural network, composed of a 3-layers self-organising feature extraction neural network module (SOFT) followed by a multi-layer classification neural network module (GSN)',
            'n 1995, Brendan Frey demonstrated that it was possible to train a network containing six fully connected layers and several hundred hidden units using the wake-sleep algorithm, co-developed with Peter Dayan and Hinton. Many factors contribute to the slow speed, including the vanishing gradient problem analyzed in 1991 by Sepp Hochreiter',
            'Simpler models that use task-specific handcrafted features such as Gabor filters and support vector machines (SVMs) were a popular choice in the 1990s and 2000s, because of artificial neural network\'s (ANN) computational cost and a lack of understanding of how the brain wires its biological networks.',
            'Both shallow and deep learning (e.g., recurrent nets) of ANNs have been explored for many years.[47][48][49] These methods never outperformed non-uniform internal-handcrafting Gaussian mixture model/Hidden Markov model (GMM-HMM) technology based on generative models of speech trained discriminatively.',
            'Key difficulties have been analyzed, including gradient diminishing[45] and weak temporal correlation structure in neural predictive models.[51][52] Additional difficulties were the lack of training data and limited computing power. Most speech recognition researchers moved away from neural nets to pursue generative modeling.',
            'An exception was at SRI International in the late 1990s. Funded by the US government\'s NSA and DARPA, SRI studied deep neural networks in speech and speaker recognition. The speaker recognition team led by Larry Heck achieved the first significant success with deep neural networks.',
            'While SRI experienced success with deep neural networks in speaker recognition, they were unsuccessful in demonstrating similar success in speech recognition. The principle of elevating "raw" features over hand-crafted optimization was first explored successfully in the architecture of deep autoencoder on the "raw" spectrogram'
        ]

        # Pre-allocate GPU memory
        tokens_list = [self.base_tokenizer.tokenize(sentence) for sentence in self.sentence_list]
        features = [self.base_tokenizer.convert_tokens_to_ids(tokens) for tokens in tokens_list]
        features = [self.base_tokenizer.prepare_for_model(input, None, add_special_tokens=True, max_length=128) for
                    input in features]
        max_len = max([len(f['input_ids']) for f in features])
        features = [f['input_ids'] + [0] * (max_len - len(f['input_ids'])) for f in features]
        all_input_ids = torch.tensor(features, dtype=torch.long)

        if self.use_gpu:
            all_input_ids = all_input_ids.cuda()

        with torch.no_grad():
            _ = self.model(all_input_ids)[0].cpu().numpy()

    def setup_base_tokenizer(self):
        self.base_tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True,
                                                               cache_dir=self.test_dir)

    def setup_rust_tokenizer(self):
        self.rust_tokenizer = PyRobertaTokenizer(
            get_from_cache(self.base_tokenizer.pretrained_vocab_files_map['vocab_file']['roberta-base']),
            get_from_cache(self.base_tokenizer.pretrained_vocab_files_map['merges_file']['roberta-base']),
            do_lower_case=True,
            add_prefix_space=True
        )

    def baseline_batch(self):
        tokens_list = [self.base_tokenizer.tokenize(sentence) for sentence in self.sentence_list]
        features = [self.base_tokenizer.convert_tokens_to_ids(tokens) for tokens in tokens_list]
        features = [self.base_tokenizer.prepare_for_model(input,
                                                          None,
                                                          add_special_tokens=True,
                                                          max_length=128) for input in features]
        max_len = max([len(f['input_ids']) for f in features])
        features = [f['input_ids'] + [0] * (max_len - len(f['input_ids'])) for f in features]
        all_input_ids = torch.tensor(features, dtype=torch.long)
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
        max_len = max([len(f.token_ids) for f in features])
        features = [f.token_ids + [0] * (max_len - len(f.token_ids)) for f in features]
        all_input_ids = torch.tensor(features, dtype=torch.long)
        if self.use_gpu:
            all_input_ids = all_input_ids.cuda()
        with torch.no_grad():
            output = self.model(all_input_ids)[0].cpu().numpy()
        return output

    def test_roberta_baseline(self):
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

    def test_roberta_rust_single_threaded(self):
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

    def teardown_class(self):
        self.model = None
        self.base_tokenizer = None
        self.rust_tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()
