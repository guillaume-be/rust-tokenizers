// Copyright 2019 Guillaume Becquin
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! # Adapter helpers to load datasets
//! While this crate does not aim at providing built-in support for loading dataset, it exposes
//! a few adapters for testing and benchmarking purposes (e.g. for SST2 sentence classification)

use crate::error::*;
use snafu::ResultExt;
use std::fs::File;

/// # Sentiment analysis label
/// Enum to represent a binary sentiment (positive or negative). An additional variant is available for
/// enums which have not yet been assigned.
#[derive(Debug)]
pub enum Label {
    Positive,
    Negative,
    Unassigned,
}

/// # SST2 sample
/// Contains a placeholder for up to 2 sentences (the SST2 dataset only contains one per example) and
/// a label
#[derive(Debug)]
pub struct Example {
    pub sentence_1: String,
    pub sentence_2: String,
    pub label: Label,
}

impl Example {
    fn new(sentence_1: &str, sentence_2: &str, label: &str) -> Result<Self, TokenizerError> {
        Ok(Example {
            sentence_1: String::from(sentence_1),
            sentence_2: String::from(sentence_2),
            label: match label {
                "0" => Ok(Label::Negative),
                "1" => Ok(Label::Positive),
                _ => ValueSnafu {
                    message: "invalid label class (must be 0 or 1)",
                }
                .fail(),
            }?,
        })
    }

    /// Creates a new `Example` from an unlabbeled string.
    ///
    ///  # Arguments
    /// - sentence (`&str`): sentence string
    ///
    ///  # Returns
    /// - `Example` containing the example with an unassigned label
    pub fn new_from_string(sentence: &str) -> Self {
        Example {
            sentence_1: String::from(sentence),
            sentence_2: String::from(""),
            label: Label::Unassigned,
        }
    }

    /// Creates a new `Example` from a sentence pair and a label `&str`.
    ///
    ///  # Arguments
    /// - sentence_1 (`&str`): first sentence string
    /// - sentence_2 (`&str`): second sentence string
    ///
    ///  # Returns
    /// - `Example` containing the example with two sentences and an unassigned label
    pub fn new_from_strings(sentence_1: &str, sentence_2: &str) -> Self {
        Example {
            sentence_1: String::from(sentence_1),
            sentence_2: String::from(sentence_2),
            label: Label::Unassigned,
        }
    }
}

/// Reads a SST2 dataset file and returns a vector of SST2 examples
///
///  # Arguments
/// - path (`&str`): path to the SST2 file
/// - sep (`u8`): separator for CSV parsing (default is a `\t` for SST2 dataset files)
///
///  # Returns
/// - `Result<Vec<Example>, TokenizerError>` containing the examples with their corresponding label
pub fn read_sst2(path: &str, sep: u8) -> Result<Vec<Example>, TokenizerError> {
    let mut examples: Vec<Example> = Vec::new();
    let f = File::open(path).expect("Could not open source file.");

    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .delimiter(sep)
        .flexible(false)
        .from_reader(f);

    for result in rdr.records() {
        let record = result.context(CSVDeserializeSnafu)?;
        let example = Example::new(&record[0], "", &record[1])?;
        examples.push(example);
    }
    Ok(examples)
}
