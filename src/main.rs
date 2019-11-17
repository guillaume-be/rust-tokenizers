use transformers;
use transformers::preprocessing::vocab::base_vocab::Vocab;
use std::process;
use transformers::preprocessing::adapters::Example;
use transformers::base_tokenizer;

fn main() {

    let _test_string = String::from("For instance, on the planet Earth, man had always assumed that he was more intelligent than dolphins because he had achieved so much—the wheel, New York, wars and so on—whilst all the dolphins had ever done was muck about in the water having a good time . But conversely, the dolphins had always believed that they were far more intelligent than man—for precisely the same reasons.");

    let bert_vocab = transformers::BertVocab::from_file("E:/Coding/rust-transformers/resources/vocab/bert-base-uncased-vocab.txt");
    println!("{:?}", bert_vocab.values.len());
    println!("{:?}", bert_vocab.token_to_id("hello"));
    println!("{:?}", bert_vocab.token_to_id("[UNK]"));
    println!("{:?}", bert_vocab.token_to_id("[PAD]"));
    println!("{:?}", bert_vocab.token_to_id("❀"));
    println!("{:?}", bert_vocab.special_values);

    let _data = match transformers::preprocessing::adapters::read_sst2(
        "E:/Coding/rust-transformers/resources/data/SST-2/dev.tsv",
        b'\t') {
        Ok(examples) => {
            examples
        }
        Err(err) => {
            println!("{}", err);
            process::exit(1);
        }
    };

    println!("{:?}", _data);

    let _test_sentence = Example::new_from_string("[MASK]it \'s a charming [SEP] [SEP] and often [MASK] affecting journey. [MASK]");
    println!("{:?}", _test_sentence);

    let tokenized_output = base_tokenizer::split_on_special_tokens(&_test_sentence.sentence_1, &bert_vocab);
    println!("{:?}", tokenized_output);

    let output = &base_tokenizer::tokenize_cjk_chars(&_test_sentence.sentence_1);
    println!("{:?}", output);
}
