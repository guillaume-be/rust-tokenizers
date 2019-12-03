use crate::preprocessing::vocab::base_vocab::Vocab;
use crate::BertVocab;
use unicode_normalization::is_nfd;
use unicode_normalization::char::{decompose_canonical, is_combining_mark};
use std::char;
use std::char::REPLACEMENT_CHARACTER;
use crate::preprocessing::tokenizer::constants::{WHITESPACE_CHARS, ADDITIONAL_WHITESPACE_CHARS,
                                                 PUNCTUATION_CHARS, CONTROL_CHARS};

pub fn clean_text(text: &str) -> String {
    let mut output = String::new();
    for character in text.chars() {
        if is_control(&character) || character == '\x00' || character == REPLACEMENT_CHARACTER {
            continue;
        }
        if is_whitespace(&character) {
            output.push(' ');
        } else {
            output.push(character);
        }
    }
    output
}

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
    if text.is_empty() {
        result.push(text);
        return result;
    }
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

pub fn tokenize_cjk_chars(text: &str) -> String {
    let mut output = String::new();
    for character in text.chars() {
        if is_cjk_char(&character) {
            output.push(' ');
            output.push(character);
            output.push(' ');
        } else {
            output.push(character);
        }
    }
    output
}

fn is_cjk_char(character: &char) -> bool {
    let u32_char = *character as u32;
    ((u32_char >= 0x4E00) & (u32_char <= 0x9FFF)) |
        ((u32_char >= 0x3400) & (u32_char <= 0x4DBF)) |
        ((u32_char >= 0x20000) & (u32_char <= 0x2A6DF)) |
        ((u32_char >= 0x2A700) & (u32_char <= 0x2B73F)) |
        ((u32_char >= 0x2B740) & (u32_char <= 0x2B81F)) |
        ((u32_char >= 0x2B820) & (u32_char <= 0x2CEAF)) |
        ((u32_char >= 0xF900) & (u32_char <= 0xFAFF)) |
        ((u32_char >= 0x2F800) & (u32_char <= 0x2FA1F))
}


fn is_whitespace(character: &char) -> bool {
    WHITESPACE_CHARS.contains(&(*character as u32))
}

fn is_control(character: &char) -> bool {
//    This is a custom method to check if a character is a control character. The BERT tokenizer is
// taking any character whose unicode category starts with `C` as a control character, which includes
// the traditional control `Cc` category, but also the format `Cc`, private use `Co` and surrogate `Cs`.
// The unassigned unicode category `Cn` has been skipped in order to avoid unnecessary checks.
    if ADDITIONAL_WHITESPACE_CHARS.contains(character) {
        false
    } else {
        let u32_char = *character as u32;
        if (u32_char <= 0x001F) |
            ((u32_char >= 0x0080) & (u32_char <= 0x009F)) |
            ((u32_char >= 0xE0020) & (u32_char <= 0xE007F)) |
            ((u32_char >= 0xE000) & (u32_char <= 0xF8FF)) |
            ((u32_char >= 0xF0000) & (u32_char <= 0xFFFFD)) |
            ((u32_char >= 0x100000) & (u32_char <= 0x10FFFD)) |
            ((u32_char >= 0xD800) & (u32_char <= 0xDB7F)) |
            ((u32_char >= 0xDB80) & (u32_char <= 0xDBFF)) |
            ((u32_char >= 0xDC00) & (u32_char <= 0xDFFF)) |
            CONTROL_CHARS.contains(&u32_char)
        {
            true
        } else {
            false
        }
    }
}

fn is_punctuation(character: &char) -> bool {
    let u32_char = *character as u32;
    if ((u32_char >= 33) & (u32_char <= 47)) |
        ((u32_char >= 58) & (u32_char <= 64)) |
        ((u32_char >= 91) & (u32_char <= 96)) |
        ((u32_char >= 123) & (u32_char <= 126)) {
        return true;
    } else {
        PUNCTUATION_CHARS.contains(&u32_char)
    }

//    character.is_ascii_punctuation()
}

pub fn whitespace_tokenize(text: &str) -> Vec<&str> {
    text.split_whitespace().collect()
}

pub fn strip_accents(text: String) -> String {
    if !is_nfd(&text) {
        let mut decomposed_string: String = String::with_capacity(text.capacity());
        for character in text.chars() {
            decompose_canonical(character, |c| if !is_combining_mark(c) { decomposed_string.push(c) });
        }
        decomposed_string
    } else {
        text
    }
}

pub fn split_on_punct(text: String, vocab: &impl Vocab) -> Vec<String> {
    let mut output: Vec<String> = Vec::new();
    let mut start_new_word: bool = true;
    let mut temp_string = String::new();
    if vocab.special_values().contains_key(&text) {
        output.push(text);
        output
    } else {
        for character in text.chars() {
            if is_punctuation(&character) {
                if !&temp_string.is_empty() {
                    output.push(temp_string.clone());
                    temp_string = String::new();
                }
                output.push(character.to_string());
                start_new_word = true
            } else {
                if start_new_word {
                    temp_string = String::new();
                }
                start_new_word = false;
                temp_string.push(character);
            }
        }
        if !start_new_word & !&temp_string.is_empty() {
            output.push(temp_string.clone());
        }
        output
    }
}

pub fn tokenize_wordpiece(token: String, vocab: &impl Vocab, max_word_len: usize) -> Vec<String> {
    let mut tokenized_text: Vec<String> = Vec::new();
    if token.chars().count() > max_word_len {
        tokenized_text.push(BertVocab::unknown_value().to_owned());
    } else {
        let char_indices: Vec<usize> = token.char_indices().map(|v| v.0).collect();
        let max_end: usize = char_indices.last().unwrap() + token.chars().last().unwrap().len_utf8();
        let mut start: usize = 0;
        let mut pos_end;
        let mut end;
        while start < max_end {
            end = max_end;
            pos_end = char_indices.len() - 1;
            while start < end {
                let mut substr = token[start..end].to_owned();
                if start > 0 {
                    substr = format!("##{}", substr);
                }
                if vocab.values().contains_key(&substr) {
                    tokenized_text.push(substr);
                    break;
                }
                if pos_end == start {
                    let mut tokenized_text: Vec<String> = Vec::new();
                    tokenized_text.push(BertVocab::unknown_value().to_owned());
                    return tokenized_text;
                }
                pos_end = pos_end - 1;
                end = char_indices[pos_end + 1];
            }
            start = end;
        }
    }
    tokenized_text
}

//==============================
// Unit tests
//==============================
#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn generate_test_vocab() -> BertVocab {
        let values: HashMap<String, i64> = [
            ("hello".to_owned(), 0),
            ("world".to_owned(), 1),
            ("[UNK]".to_owned(), 2),
            ("!".to_owned(), 3),
            ("[CLS]".to_owned(), 4),
            ("[SEP]".to_owned(), 5),
            ("[MASK]".to_owned(), 6),
            ("[中]".to_owned(), 7),
            ("华".to_owned(), 8),
            ("人]".to_owned(), 9),
            ("[PAD]".to_owned(), 10)
        ].iter().cloned().collect();

        let special_values: HashMap<String, i64> = [
            ("[UNK]".to_owned(), 2),
            ("[CLS]".to_owned(), 4),
            ("[SEP]".to_owned(), 5),
            ("[MASK]".to_owned(), 6),
            ("[PAD]".to_owned(), 10)
        ].iter().cloned().collect();

        BertVocab { values, unknown_value: "[UNK]", special_values }
    }

    #[test]
    fn test_clean_text() {
//        Given

//        When & Then
        for (source_text, expected_result) in [
            (
                "Sentence with no special character.",
                "Sentence with no special character."
            ),
            (
                "Sentence with \n some \r\n line breaks.",
                "Sentence with   some    line breaks."
            ),
            (
                "Sentence with �replacement character.",
                "Sentence with replacement character."
            ),
            (
                "Sentence with \t \t tabs.",
                "Sentence with     tabs."
            ),
            (
                "Sentence with \x00null character.",
                "Sentence with null character."
            ),
        ].iter() {
            assert_eq!(clean_text(*source_text), *expected_result);
        }
    }

    #[test]
    fn test_split_on_special_tokens() {
//        Given
        let vocab = generate_test_vocab();

//        When & Then
        for (source_text, expected_result) in [
            (
                "Sentence with [MASK] token.",
                vec!("Sentence with", "[MASK]", "token.")
            ),
            (
                "[CLS]Sentence with [MASK] token.",
                vec!("[CLS]", "Sentence with", "[MASK]", "token.")
            ),
            (
                "[CLS]",
                vec!("[CLS]")
            ),
            (
                "[CLS] [PAD]",
                vec!("[CLS]", "[PAD]")
            ),
            (
                "[CLS]       [PAD]",
                vec!("[CLS]", "[PAD]")
            ),
            (
                "asdf[CLS]",
                vec!("asdf", "[CLS]")
            ),
            (
                "No special token in sentence",
                vec!("No special token in sentence")
            ),
            (
                "",
                vec!("")
            ),
            (
                "[UNK]中华人民共和国 [PAD] asdf",
                vec!("[UNK]", "中华人民共和国", "[PAD]", "asdf")
            ),
        ].iter() {
            assert_eq!(split_on_special_tokens(*source_text, &vocab), *expected_result);
        }
    }

    #[test]
    fn test_tokenize_cjk_chars() {
//        Given

//        When & Then
        for (source_text, expected_result) in [
            (
                "[UNK]中华人民共和国 [PAD] asdf",
                String::from("[UNK] 中  华  人  民  共  和  国  [PAD] asdf")
            ),
            (
                "中",
                String::from(" 中 ")
            ),
            (
                "中b华",
                String::from(" 中 b 华 ")
            ),
            (
                "中[PAD]华",
                String::from(" 中 [PAD] 华 ")
            ),
        ].iter() {
            assert_eq!(tokenize_cjk_chars(source_text), *expected_result);
        }
    }

    #[test]
    fn test_is_cjk_char() {
//        Given
        let chinese_chars = "的是不我一有大在人了中到資要可以這個你會好為上來就學交也用能如文時沒說他看提那問生過下請天們所多麼小想得之還電出工對都機自後子而訊站去心\
        只家知國台很信成章何同道地發法無然但嗎當於本現年前真最和新因果定意情點題其事方清科樣些吧三此位理行作經者什謝名日正華話開實再城愛與二動比高面又車力或種像應女教分手打已次\
        長太明己路起相主關鳳間呢覺該十外凰友才民系進使她著各少全兩回加將感第性球式把被老公龍程論及別給聽水重體做校裡常東風您灣啦見解等部原月美先管區錯音否啊找網樂讓通入期選較四\
        場由書它快從歡數表怎至立內合目望認幾社告更版度考喜頭難光買今身許弟若算記代統處完號接言政玩師字並男計誰山張黨每且結改非星連哈建放直轉報活設變指氣研陳試西五希取神化物王戰\
        近世受義反單死任跟便空林士臺卻北隊功必聲寫平影業金檔片討色容央妳向市則員興利強白價安呵特思叫總辦保花議傳元求份件持萬未究決投哪喔笑貓組獨級走支曾標流竹兄阿室卡馬共需海口\
        門般線語命觀視朋聯參格黃錢修失兒住八腦板吃另換即象料錄拿專遠速基幫形確候裝孩備歌界除南器畫訴差講類英案帶久乎掉迷量引整似耶奇制邊型超識雖怪飛始品運賽費夢故班權破驗眼滿念\
        造軍精務留服六圖收舍半讀願李底約雄課答令深票達演早賣棒夠黑院假曲火準百談勝碟術推存治離易往況晚示證段導傷調團七永剛哥甚德殺怕包列概照夜排客絕軟商根九切條集千落竟越待忘盡\
        據雙供稱座值消產紅跑嘛園附硬雲遊展執聞唱育斯某技唉息苦質油救效須介首助職例熱畢節害擊亂態嗯寶倒注停古輸規福親查復步舉魚斷終輕環練印隨依趣限響省局續司角簡極幹篇羅佛克陽武\
        疑送拉習源免志鳥煩足館仍低廣土呀樓壞兵顯率聖碼眾爭初誤楚責境野預具智壓係青貴順負魔適哇測慢懷懂史配嗚味亦醫迎舞戀細灌甲帝句屬靈評騎宜敗左追狂敢春狗際遇族群痛右康佳楊木病\
        戲項抓徵善官護博補石爾營歷隻按妹里編歲擇溫守血領尋田養謂居異雨止跳君爛優封拜惡啥浪核聊急狀陸激模攻忙良劇牛壘增維靜陣抱勢嚴詞亞夫簽悲密幕毒廠爽緣店吳蘭睡致江宿翻香蠻警控\
        趙冷威微坐週宗普登母絡午恐套巴雜創舊輯幸劍亮述堂酒麗牌仔腳突搞父俊暴防吉禮素招草周房餐慮充府背典仁漫景紹諸琴憶援尤缺扁罵純惜授皮松委湖誠麻置靠繼判益波姐既射欲刻堆釋含承\
        退莫劉昨旁紀趕製尚藝肉律鐵奏樹毛罪筆彩註歸彈虎衛刀皆鍵售塊險榮播施銘囉漢賞欣升葉螢載嘿弄鐘付寄鬼哦燈呆洋嘻布磁薦檢派構媽藍貼豬策紙暗巧努雷架享宣逢均擔啟濟罷呼劃偉島歉郭\
        訓穿詳沙督梅顧敵".chars();

        let japanese_chars = "一九七二人入八力十下三千上口土夕大女子小山川五天中六円手文日月木水火犬王正出本右四左玉生田白目石立百年休先名字早気竹糸耳虫村男町花見貝赤\
        足車学林空金雨青草音校森刀万丸才工弓内午少元今公分切友太引心戸方止毛父牛半市北古台兄冬外広母用矢交会合同回寺地多光当毎池米羽考肉自色行西来何作体弟図声売形汽社角言谷走近\
        里麦画東京夜直国姉妹岩店明歩知長門昼前南点室後春星海活思科秋茶計風食首夏弱原家帰時紙書記通馬高強教理細組船週野雪魚鳥黄黒場晴答絵買朝道番間雲園数新楽話遠電鳴歌算語読聞線\
        親頭曜顔丁予化区反央平申世由氷主仕他代写号去打皮皿礼両曲向州全次安守式死列羊有血住助医君坂局役投対決究豆身返表事育使命味幸始実定岸所放昔板泳注波油受物具委和者取服苦重乗\
        係品客県屋炭度待急指持拾昭相柱洋畑界発研神秒級美負送追面島勉倍真員宮庫庭旅根酒消流病息荷起速配院悪商動宿帳".chars();

//        When & Then
        for character in chinese_chars {
            assert!(is_cjk_char(&character));
        }

        for character in japanese_chars {
            assert!(is_cjk_char(&character));
        }
    }

    #[test]
    fn test_is_whitespace() {
//        Given
        let whitespace_chars: [u32; 17] = [
            0x0020, 0x00A0, 0x1680, 0x2000, 0x2001, 0x2002, 0x2003,
            0x2004, 0x2005, 0x2006, 0x2007, 0x2008, 0x2009, 0x200A,
            0x202F, 0x205F, 0x3000,
        ];

        let additional_whitespace_chars: [char; 4] = [
            ' ', '\n', '\r', '\t'
        ];

        let non_whitespace_chars: [char; 5] = ['a', '5', '♥', '_', '越'];

//        When & Then
        for character in whitespace_chars.iter() {
            assert!(is_whitespace(&char::from_u32(*character).unwrap()));
        }

        for character in additional_whitespace_chars.iter() {
            assert!(is_whitespace(character));
        }

        for character in non_whitespace_chars.iter() {
            assert!(!is_whitespace(character));
        }
    }

    #[test]
    fn test_is_control() {
//        Given
        let standard_control_chars_without_space_return_tab: [u32; 62] = [
            0x0000, 0x0001, 0x0002, 0x0003, 0x0004, 0x0005, 0x0006, 0x0007, 0x0008, 0x000B,
            0x000C, 0x000E, 0x000F, 0x0010, 0x0011, 0x0012, 0x0013, 0x0014, 0x0015, 0x0016,
            0x0017, 0x0018, 0x0019, 0x001A, 0x001B, 0x001C, 0x001D, 0x001E, 0x001F, 0x007F,
            0x0080, 0x0081, 0x0082, 0x0083, 0x0084, 0x0085, 0x0086, 0x0087, 0x0088, 0x0089,
            0x008A, 0x008B, 0x008C, 0x008D, 0x008E, 0x008F, 0x0090, 0x0091, 0x0092, 0x0093,
            0x0094, 0x0095, 0x0096, 0x0097, 0x0098, 0x0099, 0x009A, 0x009B, 0x009C, 0x009D,
            0x009E, 0x009F
        ];

        let extended_control_chars: [u32; 223] = [
            0x0000, 0x0001, 0x0002, 0x0003, 0x0004, 0x0005, 0x0006, 0x0007, 0x0008, 0x000B, 0x000C,
            0x000E, 0x000F, 0x0010, 0x0011, 0x0012, 0x0013, 0x0014, 0x0015, 0x0016, 0x0017, 0x0018,
            0x0019, 0x001A, 0x001B, 0x001C, 0x001D, 0x001E, 0x001F, 0x007F, 0x0080, 0x0081, 0x0082,
            0x0083, 0x0084, 0x0085, 0x0086, 0x0087, 0x0088, 0x0089, 0x008A,
            0x008B, 0x008C, 0x008D, 0x008E, 0x008F, 0x0090, 0x0091, 0x0092, 0x0093, 0x0094, 0x0095,
            0x0096, 0x0097, 0x0098, 0x0099, 0x009A, 0x009B, 0x009C, 0x009D, 0x009E, 0x009F, 0x00AD,
            0x0600, 0x0601, 0x0602, 0x0603, 0x0604, 0x0605, 0x061C, 0x06DD, 0x070F, 0x08E2, 0x180E,
            0x200B, 0x200C, 0x200D, 0x200E, 0x200F, 0x202A, 0x202B, 0x202C, 0x202D, 0x202E, 0x2060,
            0x2061, 0x2062, 0x2063, 0x2064, 0x2066, 0x2067, 0x2068, 0x2069, 0x206A, 0x206B, 0x206C,
            0x206D, 0x206E, 0x206F, 0xFEFF, 0xFFF9, 0xFFFA, 0xFFFB, 0x110BD, 0x110CD, 0x13430,
            0x13431, 0x13432, 0x13433, 0x13434, 0x13435, 0x13436, 0x13437, 0x13438, 0x1BCA0, 0x1BCA1,
            0x1BCA2, 0x1BCA3, 0x1D173, 0x1D174, 0x1D175, 0x1D176, 0x1D177, 0x1D178, 0x1D179, 0x1D17A,
            0xE0001, 0xE0020, 0xE0021, 0xE0022, 0xE0023, 0xE0024, 0xE0025, 0xE0026, 0xE0027, 0xE0028,
            0xE0029, 0xE002A, 0xE002B, 0xE002C, 0xE002D, 0xE002E, 0xE002F, 0xE0030, 0xE0031, 0xE0032,
            0xE0033, 0xE0034, 0xE0035, 0xE0036, 0xE0037, 0xE0038, 0xE0039, 0xE003A, 0xE003B, 0xE003C,
            0xE003D, 0xE003E, 0xE003F, 0xE0040, 0xE0041, 0xE0042, 0xE0043, 0xE0044, 0xE0045, 0xE0046,
            0xE0047, 0xE0048, 0xE0049, 0xE004A, 0xE004B, 0xE004C, 0xE004D, 0xE004E, 0xE004F, 0xE0050,
            0xE0051, 0xE0052, 0xE0053, 0xE0054, 0xE0055, 0xE0056, 0xE0057, 0xE0058, 0xE0059, 0xE005A,
            0xE005B, 0xE005C, 0xE005D, 0xE005E, 0xE005F, 0xE0060, 0xE0061, 0xE0062, 0xE0063, 0xE0064,
            0xE0065, 0xE0066, 0xE0067, 0xE0068, 0xE0069, 0xE006A, 0xE006B, 0xE006C, 0xE006D, 0xE006E,
            0xE006F, 0xE0070, 0xE0071, 0xE0072, 0xE0073, 0xE0074, 0xE0075, 0xE0076, 0xE0077, 0xE0078,
            0xE0079, 0xE007A, 0xE007B, 0xE007C, 0xE007D, 0xE007E, 0xE007F
        ];

        let additional_whitespace_chars: [char; 4] = [
            ' ', '\n', '\r', '\t'
        ];

        let non_control_chars: [char; 5] = ['a', '5', '♥', '_', '越'];

//        When & Then
        for character in standard_control_chars_without_space_return_tab.iter() {
            assert!(is_control(&char::from_u32(*character).unwrap()));
        }

        for character in extended_control_chars.iter() {
            assert!(is_control(&char::from_u32(*character).unwrap()));
        }

        for character in additional_whitespace_chars.iter() {
            assert!(!is_control(character));
        }

        for character in non_control_chars.iter() {
            assert!(!is_control(character));
        }
    }
}