// Copyright 2018 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
// Copyright 2019-2020 Guillaume Becquin
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


use crate::Vocab;
use std::collections::HashMap;
use crate::preprocessing::vocab::base_vocab::swap_key_values;
use std::process;
use std::fs::File;
use std::io::Read;
use protobuf::parse_from_bytes;
use crate::preprocessing::vocab::sentencepiece_proto::sentencepiece_model::ModelProto;


pub struct T5Vocab {
    pub values: HashMap<String, i64>,
    pub indices: HashMap<i64, String>,
    pub unknown_value: &'static str,
    pub special_values: HashMap<String, i64>,
    pub special_indices: HashMap<i64, String>,
}

impl T5Vocab {
    pub fn eos_value() -> &'static str { "</s>" }
    pub fn pad_value() -> &'static str { "<pad>" }
}

impl Vocab for T5Vocab {
    fn unknown_value() -> &'static str { "<unk>" }

    fn get_unknown_value(&self) -> &'static str { "<unk>" }

    fn values(&self) -> &HashMap<String, i64> {
        &self.values
    }

    fn indices(&self) -> &HashMap<i64, String> { &self.indices }

    fn special_values(&self) -> &HashMap<String, i64> { &self.special_values }

    fn special_indices(&self) -> &HashMap<i64, String> { &self.special_indices }

    fn from_file(path: &str) -> T5Vocab {
        let mut f = File::open(path).unwrap();
        let mut contents = Vec::new();
        f.read_to_end(&mut contents).unwrap();

        let proto = parse_from_bytes::<ModelProto>(contents.as_slice()).unwrap();

        let mut values = HashMap::new();
        for (idx, piece) in proto.get_pieces().iter().enumerate() {
            values.insert(piece.get_piece().to_owned(), idx as i64);
        }

        let mut special_values = HashMap::new();
        let unknown_value = T5Vocab::unknown_value();
        T5Vocab::_register_as_special_value(unknown_value, &values, &mut special_values);

        let eos_value = T5Vocab::eos_value();
        T5Vocab::_register_as_special_value(eos_value, &values, &mut special_values);


        let pad_value = T5Vocab::pad_value();
        T5Vocab::_register_as_special_value(pad_value, &values, &mut special_values);

        let indices = swap_key_values(&values);
        let special_indices = swap_key_values(&special_values);

        T5Vocab { values, indices, unknown_value, special_values, special_indices }
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

    fn id_to_token(&self, id: &i64) -> String {
        match self._id_to_token(&id, &self.indices, &self.special_indices, &self.unknown_value) {
            Ok(token) => token,
            Err(err) => {
                println!("{}", err);
                process::exit(1);
            }
        }
    }
}