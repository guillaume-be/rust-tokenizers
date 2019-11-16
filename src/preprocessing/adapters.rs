use std::fs::File;
use std::error::Error;

#[derive(Debug)]
pub enum Label {
    Positive,
    Negative,
    Unassigned,
}

#[derive(Debug)]
pub struct Example {
    pub sentence_1: String,
    pub sentence_2: String,
    pub label: Label,
}

impl Example {
    fn new(sentence_1: &str, sentence_2: &str, label: &str) -> Result<Self, Box<dyn Error>> {
        Ok(Example {
            sentence_1: String::from(sentence_1),
            sentence_2: String::from(sentence_2),
            label: match label {
                "0" => Ok(Label::Negative),
                "1" => Ok(Label::Positive),
                _ => Err("invalid label class (must be 0 or 1)")
            }?,
        })
    }

    pub fn new_from_string(sentence: &str) -> Self {
        Example {
            sentence_1: String::from(sentence),
            sentence_2: String::from(""),
            label: Label::Unassigned,
        }
    }

    pub fn new_from_strings(sentence_1: &str, sentence_2: &str) -> Self {
        Example {
            sentence_1: String::from(sentence_1),
            sentence_2: String::from(sentence_2),
            label: Label::Unassigned,
        }
    }
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
        let example = Example::new(&record[0], "", &record[1])?;
        examples.push(example);
    };
    Ok(examples)
}
