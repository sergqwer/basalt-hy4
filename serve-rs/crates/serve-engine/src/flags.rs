//! serve/engine/flags.py: reading engine flags from a list of command-line words.

/// The index of the last `flag` in `words` (a later one wins), or None.
pub fn flag_index(words: &[String], flag: &str) -> Option<usize> {
    (0..words.len()).rev().find(|&i| words[i] == flag)
}

/// The word after the last `flag`; None when absent or when no word follows.
pub fn flag_value(words: &[String], flag: &str) -> Option<String> {
    let i = flag_index(words, flag)?;
    words.get(i + 1).cloned()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn last_wins() {
        let w: Vec<String> = ["--a", "1", "--a", "2", "--b"].iter().map(|s| s.to_string()).collect();
        assert_eq!(flag_value(&w, "--a").as_deref(), Some("2"));
        assert_eq!(flag_value(&w, "--b"), None);
        assert_eq!(flag_index(&w, "--c"), None);
    }
}
