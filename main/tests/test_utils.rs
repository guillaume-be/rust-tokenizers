use cached_path::{Cache, Error, ProgressBar};
use std::path::PathBuf;

pub fn download_file_to_cache(src: &str) -> Result<PathBuf, Error> {
    let mut cache_dir = dirs::home_dir().unwrap();
    cache_dir.push(".cache");
    cache_dir.push(".rust_tokenizers");

    let cached_path = Cache::builder()
        .dir(cache_dir)
        .progress_bar(Some(ProgressBar::Light))
        .build()?
        .cached_path(src)?;
    Ok(cached_path)
}
