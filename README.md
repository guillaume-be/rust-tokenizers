# rust-tokenizers

Rust-tokenizer is a drop-in replacement for the tokenization methods from the [Transformers library](https://github.com/huggingface/transformers)

# Set-up

Rust-tokenizer requires a rust nightly build in order to use the Python API. Building from source involes the following steps:

1. Install Rust and use the nightly tool chain
2. run `python setup.py install` in the repository. This will compile the Rust library and install the python API
3. Example use are available in the `/tests` folder, including benchmark and integration tests

The library is fully unit tested at the Rust level
