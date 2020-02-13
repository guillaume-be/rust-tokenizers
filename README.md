# rust-tokenizers

Rust-tokenizer is a drop-in replacement for the tokenization methods from the [Transformers library](https://github.com/huggingface/transformers)

# Set-up

Rust-tokenizer requires a rust nightly build in order to use the Python API. Building from source involes the following steps:

1. Install Rust and use the nightly tool chain
2. run `python setup.py install` in the repository. This will compile the Rust library and install the python API
3. Example use are available in the `/tests` folder, including benchmark and integration tests

The library is fully unit tested at the Rust level

# Usage example

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
