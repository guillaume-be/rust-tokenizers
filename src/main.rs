use transformers;
use transformers::preprocessing::vocab::base_vocab::Vocab;
use std::process;
use transformers::preprocessing::adapters::Example;

fn main() {

    let _test_string = String::from("For instance, on the planet Earth, man had always assumed that he was more intelligent than dolphins because he had achieved so much—the wheel, New York, wars and so on—whilst all the dolphins had ever done was muck about in the water having a good time . But conversely, the dolphins had always believed that they were far more intelligent than man—for precisely the same reasons.");

    let _test = transformers::BertVocab::from_file("E:/Coding/rust-transformers/resources/vocab/bert-base-uncased-vocab.txt");
    println!("{:?}", _test.values.len());
    println!("{:?}", _test.token_to_id("hello"));
    println!("{:?}", _test.token_to_id("[UNK]"));
    println!("{:?}", _test.token_to_id("[PAD]"));
    println!("{:?}", _test.token_to_id("❀"));
    println!("{:?}", _test.special_values);

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

    let _test_sentence = Example::new_from_string("it \'s a charming and often affecting journey. ");
    println!("{:?}", _test_sentence);

}
