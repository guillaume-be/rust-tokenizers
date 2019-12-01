use crate::preprocessing::vocab::base_vocab::Vocab;
use crate::BertVocab;
use unicode_normalization::is_nfd;
use unicode_normalization::char::{decompose_canonical, is_combining_mark};


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
        ((u32_char >= 0x2F800) & (u32_char <= 0x2FA1F)) |
        ((u32_char >= 0xAC00) & (u32_char <= 0xD7A3)) |
        ((u32_char >= 0x1100) & (u32_char <= 0x11FF)) |
        ((u32_char >= 0x3130) & (u32_char <= 0x318F)) |
        ((u32_char >= 0xA960) & (u32_char <= 0xA97F)) |
        ((u32_char >= 0xD7B0) & (u32_char <= 0xD7FF))
}

//ToDo: Add the control chars to the list of whitespaces
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
            if character.is_ascii_punctuation() {
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

        let korean_chars = "로나는그의그그했다에대한에아르와그들있다에일이이부터에의해뜨거운단어하지만무엇다소이다그당신또는했다에의에과이에우리수아웃다른했다하는할\
        자신의시간면것방법말했다이각이야기하지세트세필요공기잘또한재생작은끝넣어홈읽기손포트큰철자추가도땅여기해야큰높은이러한따라행위이유문의남자변경갔다빛종류오프필요가있다\
        사진시험우리다시동물포인트어머니세계가까운구축자기지구아버지모든새로운일일부소요도착장소만든살고있다어디에후다시작은만둥근사람년온쇼모든좋은나를제공우리의아래의이름대\
        단히를통해단지양식문장큰생각말도움낮은온라인차이회전원인많은의미이전움직임바로소년늙은너무동일그녀모든그곳에때올라사용당신의방법에대한많은다음그쓰기것같은그래서이들그\
        녀의긴확인일참조그두이봐더일수이동올한수소리없음가장사람들내이상알고물보다통화첫째사람수도아래로측면하고지금발견머리서자신의페이지해야국가발견답변학교성장연구여전히학\
        습공장덮개음식일네사이상태유지눈결코마지막하자생각도시트리교차농장단단한시작수도이야기톱까지바다그리왼쪽후반실행하지반면키를누릅니다가까이밤실제생활조금북책수행했다과\
        학식사방친구시작아이디어물고기산중지한번기본듣다말컷확실한손목시계색얼굴나무주오픈것함께다음흰색어린이시작있어도보예완화종이그룹항상음악그모두마르크자주편지까지마일강\
        자동차피트주의초충분히일반소녀보통젊은준비된위지금까지빨간색표그래도느낌이야기조류곧몸개가족직접포즈떠나노래측정문제품블랙짧은숫자클래스바람질문일완료배지역반바위주문\
        화재남쪽문제조각이야기알고통과이후최고전체왕거리인치곱아무것도물론유지휠전체힘푸른객체결정표면깊은달섬발시스템바쁜테스트기록보트공통의금가능한비행기대신건조궁금웃음천\
        전실행확인게임모양동일시뜨거운미스가져열눈타이어가져예먼입력동쪽페인트언어중단위힘마을잘어떤비행가을지도울음소리어두운기계주의대기계획그림스타상자명사필드나머지정확한\
        수파운드완료아름다움드라이브서포함앞가르쳐주최종준녹색오빨리개발대양따뜻한무료분강한특수마음뒤에명확한꼬리생산사실공간들어가장시간더사실중백오기억단계초기개최서쪽지상\
        관심범위빠른동사노래청취육나타난여행이하아침열간단한여러모음방향전쟁누워에대하여패턴느린센터사랑사람돈봉사표시도로지도비규칙적용당겨감기예고음성에너지사냥가능성침대형\
        제계란타고셀생각어쩌면선택갑자기계산광장이유길이대표예술주제지역크기다양정착이야기무게일반얼음문제원쌍포함분할음절느낌그랜드공아직파드롭마음어디로현재무거운댄스엔진위\
        치팔폭항해재료분수숲앉아레이스창상점여름기차잠증명고독한다리운동벽캐치마운트소원하늘판즐거움겨울토기록야생악기유지유리잔디소작업에지로그인방문지난소프트재미밝은가스날\
        씨개월만곰끝행복기대꽃의복을걸치다기이한사라무역멜로디여행사무실수신행입정확한기호죽다이상수고소리제".chars();

//        When & Then
        for character in chinese_chars {
            assert!(is_cjk_char(&character));
        }

        for character in japanese_chars {
            assert!(is_cjk_char(&character));
        }

        for (idx, character) in korean_chars.enumerate() {
            assert!(is_cjk_char(&character));
        }
    }
}