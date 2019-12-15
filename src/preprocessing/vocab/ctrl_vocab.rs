// Copyright 2018 Salesforce
// Copyright 2018 The HuggingFace Inc. team.
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

use std::collections::HashMap;
use crate::preprocessing::vocab::base_vocab::Vocab;
use std::{process, ptr};
use std::fs::File;
use std::io::{BufReader, BufRead};
use std::mem::ManuallyDrop;

pub struct CtrlVocab {
    pub values: HashMap<String, i64>,
    pub unknown_value: &'static str,
    pub special_values: HashMap<String, i64>,
}

impl Vocab for CtrlVocab {
    fn unknown_value() -> &'static str { "<unk>" }

    fn values(&self) -> &HashMap<String, i64> {
        &self.values
    }

    fn special_values(&self) -> &HashMap<String, i64> {
        &self.special_values
    }

    fn from_file(path: &str) -> CtrlVocab {
        let f = File::open(path).expect("Could not open vocabulary file.");
        let br = BufReader::new(f);
        let values: HashMap<String, i64> = serde_json::from_reader(br).expect("could not parse vocabulary");
        let mut special_values = HashMap::new();
        let unknown_value = CtrlVocab::unknown_value();
        CtrlVocab::_register_as_special_value(unknown_value, &values, &mut special_values);

        CtrlVocab { values, unknown_value, special_values }
    }

    fn token_to_id(&self, token: &str) -> i64 {
        match self._token_to_id(token, &self.values, &self.special_values, &self.unknown_value) {
            Ok(index) => index,
            Err(err) => {
                println!("{}", err);
                process::exit(1);
            }
        }
    }
}

#[derive(Eq, PartialEq, Hash, Clone, Debug)]
pub struct BpePairRef<'a> {
    pub byte_1: &'a String,
    pub byte_2: &'a String,
}

pub struct BpePairVocab {
    pub values: HashMap<(String, String), i64>
}

impl BpePairVocab {
    pub fn from_file(path: &str) -> BpePairVocab {
        let f = File::open(path).expect("Could not open vocabulary file.");
        let br = BufReader::new(f);
        let mut data = HashMap::new();
        let mut index = 0;
        for line in br.lines().skip(1) {
            let tuple: Vec<String> = line.unwrap().trim().split(' ').map(|v| v.to_owned()).collect();
            if tuple.len() > 1 {
                data.insert((tuple[0].clone(), tuple[1].clone()), index);
                index += 1;
            }
        };

        BpePairVocab { values: data }
    }

    pub fn byte_pair_to_id(&self, byte_pair: &BpePairRef) -> Option<&i64> {
        unsafe {
            let byte_1 = byte_pair.byte_1;
            let byte_2 = byte_pair.byte_2;
            let k = (ptr::read(byte_1), ptr::read(byte_2));
            let k = ManuallyDrop::new(k);
            let v = self.values.get(&k);
            v
        }
    }
}

