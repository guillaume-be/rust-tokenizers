use std::fs;
use std::fs::File;
use std::io::copy;
use std::path::PathBuf;

pub fn download_file_to_cache(src: &str, target: &str) -> Result<PathBuf, reqwest::Error> {
    let mut home = dirs::home_dir().unwrap();
    home.push(".cache");
    home.push(".rust_tokenizers");
    home.push(target);
    if !home.exists() {
        let mut response = reqwest::blocking::get(src)?;
        fs::create_dir_all(home.parent().unwrap()).unwrap();
        let mut dest = File::create(&home).unwrap();
        copy(&mut response, &mut dest).unwrap();
    }
    Ok(home)
}
