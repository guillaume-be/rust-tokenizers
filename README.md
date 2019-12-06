# rust-tokenizers

Rust-tokenizer is a drop-in replacement for the tokenization methods from the [Transformers library](https://github.com/huggingface/transformers)

### Benchmark results

|Name (time in ms)                      |       Min       |        Max       |    Mean        |     StdDev     | 
|---------------------------------------|-----------------|------------------|----------------|----------------|
|test_distilbert_rust_multi_threaded    |  101.16 (1.0)   |  105.58 (1.0)    |  103.04 (1.0)  |   1.49 (5.95)  |
|test_distilbert_rust_single_threaded   |  106.98 (1.06)  |  113.92 (1.08)   | 109.25 (1.06)  |   2.29 (9.12)  |
|test_distilbert_baseline               |  170.26 (1.68)  |  176.84 (1.67)   | 173.49 (1.68)  |   2.13 (8.51)  |
|test_bert_rust_multi_threaded          |   199.57 (1.97) |  205.45 (1.95)   |  202.35 (1.96) |   2.23 (8.89)  |
|test_bert_rust_single_threaded         |  204.37 (2.02)  |  205.08 (1.94)   |  204.72 (1.99) |   0.25 (1.0)   |
|test_bert_baseline                     |  271.12 (2.68)  |   273.69 (2.59)  |  272.30 (2.64) |   1.25 (4.98)  |
