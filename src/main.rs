use transformers;
use transformers::preprocessing::vocab::base_vocab::Vocab;
use std::process;
use transformers::preprocessing::adapters::Example;
use transformers::preprocessing::tokenizer::bert_tokenizer::tokenize_bert;

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

//    println!("{:?}", _data);

//    let _test_sentence = Example::new_from_string("[MASK]it \'s a charming [SEP] [SEP] and often [MASK] affecting journey. [MASK]");
    let _test_sentence = Example::new_from_string("[MASK]Reprise au tout début des années [SEP]1960[SEP] par le commissariat à l'énergie atomique (CEA), cette structure reste, au xxie siècle, l'un des principaux employeurs de main d'œuvre de la commune.");
//    let _test_sentence = Example::new_from_string("[CLS]吉村洋文辭職參選日本大阪府知事後，與其同屬大阪維新會的前大阪府知事松井一郎在哪一次選舉成功接替吉村洋文的位置？");
    println!("{:?}", _test_sentence);

//    for example in _data{
//        tokenize_bert(&example.sentence_1, &bert_vocab);
//    }

    let tokenized_output = tokenize_bert(&_test_sentence.sentence_1, &bert_vocab);
    println!("{:?}", tokenized_output);

}
