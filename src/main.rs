use rust_transformers;
use rust_transformers::preprocessing::vocab::base_vocab::Vocab;
use std::process;
use rust_transformers::preprocessing::adapters::Example;
use rust_transformers::preprocessing::tokenizer::bert_tokenizer::BertTokenizer;
use std::time::Instant;
use rust_transformers::preprocessing::tokenizer::base_tokenizer::Tokenizer;
use std::sync::Arc;

fn main() {
    let vocab_path = "E:/Coding/rust-transformers/resources/vocab/bert-base-uncased-vocab.txt";
    let bert_vocab = Arc::new(rust_transformers::BertVocab::from_file(vocab_path));

    let _data = match rust_transformers::preprocessing::adapters::read_sst2(
        "E:/Coding/rust-transformers/resources/data/SST-2/train.tsv",
        b'\t') {
        Ok(examples) => {
            examples
        }
        Err(err) => {
            println!("{}", err);
            process::exit(1);
        }
    };

    let _test_sentence = Example::new_from_string("[MASK]Reprise �au tout début des années [SEP]1960[SEP] par le commissariat à l'énergie atomique (CEA), cette structure reste, au xxie siècle, l'un des principaux employeurs de main d'œuvre de la commune.");
//    let _test_sentence = Example::new_from_string("[CLS]吉村洋文辭職參選日本大阪府知事後，與其同屬大阪維新會的前大阪府知事松井一郎在哪一次選舉成功接替吉村洋文的位置？");

    let bert_tokenizer: BertTokenizer = BertTokenizer::from_existing_vocab(bert_vocab.clone());
    let tokenized_text = bert_tokenizer.tokenize(&_test_sentence.sentence_1);
    let encoded_text = bert_tokenizer.encode(&_test_sentence.sentence_1, None, 10);
    println!("{:?}", tokenized_text);
    println!("{:?}", encoded_text);


    let _text_list: Vec<&str> = _data.iter().map(|v| v.sentence_1.as_ref()).collect();
    let _before = Instant::now();
//    let _results = bert_tokenizer.tokenize_list(_text_list);
//    for text in _text_list{
//        bert_tokenizer.tokenize(text);
//    }
    let _results = bert_tokenizer.encode_list(_text_list, 128);
//    for result in _text_list {
//        bert_tokenizer.encode(&result);
//    }
    println!("Elapsed time: {:.2?}", _before.elapsed());
}
