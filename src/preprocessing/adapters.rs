use std::fs::File;
use std::error::Error;

#[derive(Debug)]
pub enum Label {
    Positive,
    Negative,
}

#[derive(Debug)]
pub struct Example {
    sentence_1: String,
    sentence_2: String,
    label: Label,
}

pub fn read_sst2(path: &str, sep: u8) -> Result<Vec<Example>, Box<dyn Error>> {
    let mut examples: Vec<Example> = Vec::new();
    let f = File::open(path).expect("Could not open source file.");

    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .delimiter(sep)
        .flexible(false)
        .from_reader(f);

    for result in rdr.records() {
        let record = result?;
        let example = Example {
            sentence_1: String::from(&record[0]),
            sentence_2: String::from(""),
            label: match &record[1] {
                "0" => Ok(Label::Negative),
                "1" => Ok(Label::Positive),
                _ => Err("invalid label encountered")
            }?,
        };
        examples.push(example);
    };
    Ok(examples)
}