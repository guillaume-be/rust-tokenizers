# rust-tokenizers

[![Build Status](https://github.com/guillaume-be/rust-tokenizers/workflows/Build/badge.svg?event=push)](https://github.com/guillaume-be/rust-tokenizers/actions)
[![Latest version](https://img.shields.io/crates/v/rust_tokenizers.svg)](https://crates.io/crates/rust_tokenizers)
![License](https://img.shields.io/crates/l/rust_tokenizers.svg)

Rust-tokenizer is a drop-in replacement for the tokenization methods from the [Transformers library](https://github.com/huggingface/transformers)
These tokenizers are used in the [rust-bert](https://github.com/guillaume-be/rust-bert) crate.
A broad range of tokenizers for state-of-the-art transformers architectures is included, including:
- Sentence Piece (unigram model)
- BERT
- ALBERT
- DistilBERT
- RoBERTa
- GPT
- GPT2
- CTRL

The wordpiece based tokenizers include both single-threaded and multi-threaded processing. The Byte-Pair-Encoding tokenizers favor the use of a shared cache and are only available as single-threaded tokenizers
Using the tokenizers requires downloading manually the tokenizers required files (vocabulary or merge files). These can be found in the [Transformers library](https://github.com/huggingface/transformers).

The sentence piece model loads the same `.model` proto files as the [C++ library](https://github.com/google/sentencepiece)

# Usage example (Rust)

```rust
let vocab = Arc::new(rust_tokenizers::BertVocab::from_file(&vocab_path));

let test_sentence = Example::new_from_string("This is a sample sentence to be tokenized");
let bert_tokenizer: BertTokenizer = BertTokenizer::from_existing_vocab(vocab.clone());

println!("{:?}", bert_tokenizer.encode(&test_sentence.sentence_1,
                                       None,
                                       128,
                                       &TruncationStrategy::LongestFirst,
                                       0));
```


# Python bindings set-up

Rust-tokenizer requires a rust nightly build in order to use the Python API. Building from source involves the following steps:

1. Install Rust and use the nightly tool chain
2. run `python setup.py install` in the `/python-bindings` repository. This will compile the Rust library and install the python API
3. Example use are available in the `/tests` folder, including benchmark and integration tests

The library is fully unit tested at the Rust level

# Usage example (Python)

```python
from rust_transformers import PyBertTokenizer
from transformers.modeling_bert import BertForSequenceClassification

rust_tokenizer = PyBertTokenizer('bert-base-uncased-vocab.txt')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', output_attentions=False).cuda()
model = model.eval()

sentence = '''For instance, on the planet Earth, man had always assumed that he was more intelligent than dolphins because 
              he had achieved so much—the wheel, New York, wars and so on—whilst all the dolphins had ever done was muck 
              about in the water having a good time. But conversely, the dolphins had always believed that they were far 
              more intelligent than man—for precisely the same reasons.'''

features = rust_tokenizer.encode(sentence, max_len=128, truncation_strategy='only_first', stride=0)
input_ids = torch.tensor([f.token_ids for f in features], dtype=torch.long).cuda()

with torch.no_grad():
    output = model(all_input_ids)[0].cpu().numpy()
```
