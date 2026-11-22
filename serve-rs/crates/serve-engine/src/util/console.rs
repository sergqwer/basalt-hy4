//! serve/util/console.py: the server's console lines in the compact key=value style: `[server] <label>  <key=value ...>`.

pub const LABEL_WIDTH: usize = 8;

pub const LEGEND: &str = "pp=prompt read  tg=generation  c=cached  acc=drafts accepted  dl=tokens per window  \
                          hit=expert cache  pcie=routed experts over PCIe";

/// A startup line: the label padded to a column, then the text.
pub fn tagged(label: &str, text: &str) -> String {
    format!("[server] {} {text}", crate::pyfmt::ljust(label, LABEL_WIDTH))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn padded() {
        assert_eq!(tagged("kv", "1.0GiB cuda0"), "[server] kv       1.0GiB cuda0");
        assert_eq!(tagged("starting", "3s"), "[server] starting 3s");
        assert_eq!(tagged("networked", "x"), "[server] networked x");
        assert!(LEGEND.starts_with("pp=prompt read  tg=generation") && LEGEND.ends_with("over PCIe"));
        assert!(LEGEND.contains("per window  hit=expert"));
    }
}
