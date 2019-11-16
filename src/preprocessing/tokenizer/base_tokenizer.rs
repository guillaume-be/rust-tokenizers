use crate::preprocessing::vocab::base_vocab::Vocab;

pub fn split_on_special_tokens<'a>(text: &'a str, vocab: &'a impl Vocab) -> Vec<&'a str> {
    let mut text_list: Vec<&str> = vec!(text);
    let mut temp_list: Vec<&str>;

    for special_value in vocab.special_values() {
        temp_list = vec!();
        for subtext in &text_list {
            let new_items = split_with_separator(subtext, special_value.0);
            temp_list.extend(new_items);
        }
        text_list = temp_list;
    }
    text_list
}

fn split_with_separator<'a>(text: &'a str, separator: &'a str) -> Vec<&'a str> {
    let split_text: Vec<&str> = text.split(separator).collect();
    let mut result: Vec<&str> = vec!();
    for (i, subtext) in split_text.iter().enumerate() {
        let trimmed_subtext = subtext.trim();
        if (i == 0) & trimmed_subtext.is_empty() {
            result.push(separator);
        } else if i == split_text.len() - 1 {
            if !trimmed_subtext.is_empty() {
                result.push(trimmed_subtext);
            }
        } else {
            if !trimmed_subtext.is_empty() {
                result.push(trimmed_subtext);
            }
            result.push(separator);
        }
    }
    result
}