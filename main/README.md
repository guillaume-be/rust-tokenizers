# rust-tokenizers

Rust-tokenizer is a drop-in replacement for the tokenization methods from the [Transformers library](https://github.com/huggingface/transformers)
It includes a broad range of tokenizers for state-of-the-art transformers architectures, including:
- Sentence Piece (unigram model)
- BERT
- ALBERT
- DistilBERT
- RoBERTa
- GPT
- GPT2
- ProphetNet
- CTRL

The wordpiece based tokenizers include both single-threaded and multi-threaded processing. The Byte-Pair-Encoding tokenizers favor the use of a shared cache and are only available as single-threaded tokenizers
Using the tokenizers requires downloading manually the tokenizers required files (vocabulary or merge files). These can be found in the [Transformers library](https://github.com/huggingface/transformers).

# Usage example

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
