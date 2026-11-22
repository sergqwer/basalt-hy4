//! tools/tokenizer.py (the byte-level BPE tokenizer read from a pack's `tokenizer/` sections) and util/text.py's
//! ByteTokenizer (the mock engine's). Plain spans are byte offsets into the text (Python's are character offsets
//! into the same text: the same spans).

use std::collections::{BinaryHeap, HashMap};
use std::sync::RwLock;

/// What the server needs of a tokenizer.
pub trait Tokenizer: Send + Sync {
    /// The ids of `text`; `parse_special` reads control-token text as its token; `plain` (byte spans) stays text.
    fn encode(&self, text: &str, parse_special: bool, plain: &[(usize, usize)]) -> Vec<u32>;
    /// The text of `ids` (invalid UTF-8 replaced); Err for an id outside the vocabulary.
    fn decode(&self, ids: &[u32]) -> Result<String, String>;
    /// Python's hasattr(tok, "token_bytes").
    fn has_token_bytes(&self) -> bool;
    /// One token's raw bytes (only when has_token_bytes).
    fn token_bytes(&self, id: u32) -> Result<Vec<u8>, String>;
    /// ByteTokenizer.control_tokens (None: the tokenizer has no such property).
    fn control_tokens(&self) -> Option<Vec<String>>;
    /// The special literals (types 3 and 4), in id order.
    fn special_tokens(&self) -> Vec<String>;
    fn vocab_size(&self) -> usize;
}

/// control_literals(tok): the special literals read as tokens only when asked to parse them.
pub fn control_literals(tok: &dyn Tokenizer) -> Vec<String> {
    if let Some(named) = tok.control_tokens() {
        return named;
    }
    tok.special_tokens()
        .into_iter()
        .filter(|s| !s.is_empty() && tok.encode(s, false, &[]) != tok.encode(s, true, &[]))
        .collect()
}

// ------------------------------------------------------------------------------------------------ ByteTokenizer
pub struct ByteTokenizer;

pub const BYTE_SPECIALS: [&str; 7] = [
    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",
    "<|vision_start|>",
    "<|image_pad|>",
    "<|vision_end|>",
    "<|video_pad|>",
];

impl Tokenizer for ByteTokenizer {
    fn encode(&self, text: &str, parse_special: bool, plain: &[(usize, usize)]) -> Vec<u32> {
        let mut out = Vec::with_capacity(text.len());
        let mut i = 0;
        let b = text.as_bytes();
        while i < b.len() {
            let mut hit = false;
            if parse_special && !plain.iter().any(|&(a, e)| a <= i && i < e) {
                for (k, s) in BYTE_SPECIALS.iter().enumerate() {
                    if text[i..].starts_with(s) {
                        out.push(256 + k as u32);
                        i += s.len();
                        hit = true;
                        break;
                    }
                }
            }
            if !hit {
                let ch = text[i..].chars().next().unwrap();
                let n = ch.len_utf8();
                out.extend(b[i..i + n].iter().map(|&x| x as u32));
                i += n;
            }
        }
        out
    }

    fn decode(&self, ids: &[u32]) -> Result<String, String> {
        let mut raw = Vec::new();
        for &t in ids {
            if t >= 256 {
                let s = BYTE_SPECIALS.get((t - 256) as usize).ok_or_else(|| "list index out of range".to_string())?;
                raw.extend_from_slice(s.as_bytes());
            } else {
                raw.push(t as u8);
            }
        }
        Ok(String::from_utf8_lossy(&raw).into_owned())
    }

    fn has_token_bytes(&self) -> bool {
        false
    }

    fn token_bytes(&self, id: u32) -> Result<Vec<u8>, String> {
        self.decode(&[id]).map(|s| s.into_bytes())
    }

    fn control_tokens(&self) -> Option<Vec<String>> {
        Some(BYTE_SPECIALS.iter().map(|s| s.to_string()).collect())
    }

    fn special_tokens(&self) -> Vec<String> {
        BYTE_SPECIALS.iter().map(|s| s.to_string()).collect()
    }

    fn vocab_size(&self) -> usize {
        256 + BYTE_SPECIALS.len()
    }
}

// ------------------------------------------------------------------------------------------------ BPE
fn bytes_to_unicode() -> [char; 256] {
    let mut bs: Vec<u32> = (0x21..0x7f).chain(0xa1..0xad).chain(0xae..0x100).collect();
    let mut cs = bs.clone();
    let mut n = 0;
    for b in 0..256u32 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut out = ['\0'; 256];
    for (b, c) in bs.iter().zip(cs) {
        out[*b as usize] = char::from_u32(c).unwrap();
    }
    out
}

pub const QWEN35_PATTERN: &str = concat!(
    r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])",
    r"|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+",
    r"|\p{N}",
    r"| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*",
    r"|\s*[\r\n]+",
    r"|\s+(?!\S)",
    r"|\s+"
);

pub const JOYAI_PATTERNS: [&str; 3] = [
    r"\p{N}{1,3}",
    "[\u{4e00}-\u{9fa5}\u{3040}-\u{309f}\u{30a0}-\u{30ff}]+",
    concat!(
        r##"[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+"##,
        r"| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    ),
];

const HEAP_MIN: usize = 64;
const PIECE_CACHE_MAX: usize = 200_000;

pub struct BpeTokenizer {
    tokens: Vec<String>,
    ids: HashMap<String, u32>,
    ranks: HashMap<(String, String), u32>,
    res: Vec<fancy_regex::Regex>,
    special: HashMap<String, u32>,
    special_order: Vec<String>,
    always_re: Option<regex::Regex>,
    special_re: Option<regex::Regex>,
    byte_enc: [char; 256],
    byte_dec: HashMap<char, u8>,
    cache: RwLock<HashMap<String, Vec<u32>>>,
    bytes_cache: RwLock<HashMap<u32, Vec<u8>>>,
    pub pre: String,
}

impl BpeTokenizer {
    /// Tokenizer(tokens, merges, token_types, pre).
    pub fn new(tokens: Vec<String>, merges: &[&str], token_types: Option<Vec<i64>>, pre: &str) -> Result<Self, String> {
        let mut ids = HashMap::with_capacity(tokens.len());
        for (i, t) in tokens.iter().enumerate() {
            ids.insert(t.clone(), i as u32);
        }
        if ids.len() != tokens.len() {
            return Err(format!(
                "vocabulary has duplicate tokens: {} entries, {} unique",
                tokens.len(),
                ids.len()
            ));
        }
        let mut ranks = HashMap::with_capacity(merges.len());
        for (i, m) in merges.iter().enumerate() {
            let parts: Vec<&str> = m.split(' ').collect();
            if parts.len() != 2 {
                return Err(format!("merge {i} is not a pair: {m:?}"));
            }
            if !ids.contains_key(parts[0]) || !ids.contains_key(parts[1]) {
                return Err(format!("merge {i} names a token outside the vocabulary: {m:?}"));
            }
            ranks.insert((parts[0].to_string(), parts[1].to_string()), i as u32);
        }
        let pats: Vec<&str> = match pre {
            "joyai-llm" | "hy_v4" => JOYAI_PATTERNS.to_vec(),
            _ => vec![QWEN35_PATTERN],
        };
        let res = pats
            .iter()
            .map(|p| fancy_regex::Regex::new(p).map_err(|e| e.to_string()))
            .collect::<Result<Vec<_>, _>>()?;
        let mut special = HashMap::new();
        let mut special_order = Vec::new();
        let mut always = Vec::new();
        if let Some(types) = &token_types {
            for (i, ty) in types.iter().enumerate() {
                if (*ty == 3 || *ty == 4) && i < tokens.len() {
                    // dict assignment: a repeated literal keeps its first position and its last id
                    if special.insert(tokens[i].clone(), i as u32).is_none() {
                        special_order.push(tokens[i].clone());
                    }
                }
            }
            for t in &special_order {
                if types[special[t] as usize] == 4 {
                    always.push(t.clone());
                }
            }
        }
        let always_re = alt(&always);
        let special_re = alt(&special_order);
        let byte_enc = bytes_to_unicode();
        let byte_dec = byte_enc.iter().enumerate().map(|(b, c)| (*c, b as u8)).collect();
        Ok(BpeTokenizer {
            tokens,
            ids,
            ranks,
            res,
            special,
            special_order,
            always_re,
            special_re,
            byte_enc,
            byte_dec,
            cache: RwLock::new(HashMap::new()),
            bytes_cache: RwLock::new(HashMap::new()),
            pre: pre.to_string(),
        })
    }

    /// From the pack's tokenizer files, as server.py's build_model reads them.
    pub fn from_files(
        vocab_json: &[u8],
        merges_txt: &[u8],
        token_type_json: &[u8],
        tokenizer_json: Option<&[u8]>,
    ) -> Result<Self, String> {
        let vocab: serde_json::Value = serde_json::from_slice(vocab_json).map_err(|e| e.to_string())?;
        let vocab = vocab.as_object().ok_or("vocab.json: not an object")?;
        let mut tokens = vec![String::new(); vocab.len()];
        for (t, i) in vocab {
            let i = i.as_u64().ok_or("vocab.json: a non-integer id")? as usize;
            if i >= tokens.len() {
                return Err("vocab.json: an id past the vocabulary".into());
            }
            tokens[i] = t.clone();
        }
        let merges_text = String::from_utf8_lossy(merges_txt);
        let merges: Vec<&str> = merges_text.split('\n').collect();
        let types: Vec<i64> = serde_json::from_slice(token_type_json).map_err(|e| e.to_string())?;
        let pre = match tokenizer_json {
            Some(b) => {
                let v: serde_json::Value = serde_json::from_slice(b).map_err(|e| e.to_string())?;
                v.get("pre").and_then(|p| p.as_str()).unwrap_or("qwen35").to_string()
            }
            None => "qwen35".into(),
        };
        // Python splits merges.txt on "\n" and validates every line, so a trailing newline would fail there too
        Self::new(tokens, &merges, Some(types), &pre)
    }

    fn bpe(&self, word: &[String]) -> Vec<String> {
        if word.len() > HEAP_MIN {
            return self.bpe_heap(word);
        }
        let mut parts: Vec<String> = word.to_vec();
        while parts.len() > 1 {
            let mut best: Option<(usize, u32)> = None;
            for i in 0..parts.len() - 1 {
                if let Some(&r) = self.rank(&parts[i], &parts[i + 1]) {
                    if best.map(|(_, b)| r < b).unwrap_or(true) {
                        best = Some((i, r));
                    }
                }
            }
            let Some((i, _)) = best else { break };
            let merged = format!("{}{}", parts[i], parts[i + 1]);
            parts.splice(i..i + 2, [merged]);
        }
        parts
    }

    fn rank(&self, a: &str, b: &str) -> Option<&u32> {
        // the HashMap key is a pair of Strings; a lookup builds it (cheap next to the regex split)
        self.ranks.get(&(a.to_string(), b.to_string()))
    }

    fn bpe_heap(&self, word: &[String]) -> Vec<String> {
        let mut parts: Vec<Option<String>> = word.iter().cloned().map(Some).collect();
        let n = parts.len();
        let mut nxt: Vec<isize> = (1..n as isize).chain([-1]).collect();
        let mut prv: Vec<isize> = (-1..n as isize - 1).collect();
        // min-heap of (rank, position, left, right)
        let mut heap: BinaryHeap<std::cmp::Reverse<(u32, usize, String, String)>> = BinaryHeap::new();
        for i in 0..n.saturating_sub(1) {
            let (l, r) = (parts[i].clone().unwrap(), parts[i + 1].clone().unwrap());
            if let Some(&rk) = self.rank(&l, &r) {
                heap.push(std::cmp::Reverse((rk, i, l, r)));
            }
        }
        while let Some(std::cmp::Reverse((_, i, left, right))) = heap.pop() {
            let j = nxt[i];
            if parts[i].as_deref() != Some(left.as_str()) || j < 0 || parts[j as usize].as_deref() != Some(right.as_str())
            {
                continue;
            }
            let j = j as usize;
            let merged = format!("{left}{right}");
            parts[i] = Some(merged.clone());
            parts[j] = None;
            let k = nxt[j];
            nxt[i] = k;
            if k >= 0 {
                prv[k as usize] = i as isize;
                let kr = parts[k as usize].clone().unwrap();
                if let Some(&r2) = self.rank(&merged, &kr) {
                    heap.push(std::cmp::Reverse((r2, i, merged.clone(), kr)));
                }
            }
            let p = prv[i];
            if p >= 0 {
                let pl = parts[p as usize].clone().unwrap();
                if let Some(&r2) = self.rank(&pl, &merged) {
                    heap.push(std::cmp::Reverse((r2, p as usize, pl, merged.clone())));
                }
            }
        }
        parts.into_iter().flatten().collect()
    }

    fn split<'t>(&self, text: &'t str) -> Vec<&'t str> {
        if self.res.len() == 1 {
            return self.res[0].find_iter(text).filter_map(|m| m.ok()).map(|m| m.as_str()).collect();
        }
        let mut pieces = vec![text];
        for rx in &self.res {
            let mut next = Vec::new();
            for p in pieces {
                let mut pos = 0;
                for m in rx.find_iter(p).filter_map(|m| m.ok()) {
                    if m.start() > pos {
                        next.push(&p[pos..m.start()]);
                    }
                    if m.end() > m.start() {
                        next.push(m.as_str());
                    }
                    pos = m.end();
                }
                if pos < p.len() {
                    next.push(&p[pos..]);
                }
            }
            pieces = next;
        }
        pieces
    }

    fn encode_plain(&self, text: &str, out: &mut Vec<u32>) {
        for piece in self.split(text) {
            if let Some(got) = self.cache.read().unwrap().get(piece) {
                out.extend_from_slice(got);
                continue;
            }
            let mapped: Vec<String> = piece.bytes().map(|b| self.byte_enc[b as usize].to_string()).collect();
            let got: Vec<u32> = self.bpe(&mapped).iter().filter_map(|t| self.ids.get(t).copied()).collect();
            out.extend_from_slice(&got);
            let mut c = self.cache.write().unwrap();
            if c.len() < PIECE_CACHE_MAX {
                c.insert(piece.to_string(), got);
            }
        }
    }
}

fn alt(literals: &[String]) -> Option<regex::Regex> {
    if literals.is_empty() {
        return None;
    }
    let mut sorted: Vec<&String> = literals.iter().collect();
    sorted.sort_by_key(|s| std::cmp::Reverse(s.chars().count())); // stable, as Python's sorted
    let pat = sorted.iter().map(|s| regex::escape(s)).collect::<Vec<_>>().join("|");
    regex::Regex::new(&pat).ok()
}

impl Tokenizer for BpeTokenizer {
    fn encode(&self, text: &str, parse_special: bool, plain: &[(usize, usize)]) -> Vec<u32> {
        let pat = if parse_special { &self.special_re } else { &self.always_re };
        let mut out = Vec::new();
        let Some(pat) = pat else {
            self.encode_plain(text, &mut out);
            return out;
        };
        let mut pos = 0;
        for m in pat.find_iter(text) {
            if !plain.is_empty() && plain.iter().any(|&(a, b)| a <= m.start() && m.start() < b) {
                continue;
            }
            if m.start() < pos {
                // a match inside a skipped one cannot happen with find_iter; kept for safety
                continue;
            }
            if m.start() > pos {
                self.encode_plain(&text[pos..m.start()], &mut out);
            }
            out.push(self.special[m.as_str()]);
            pos = m.end();
        }
        if pos < text.len() {
            self.encode_plain(&text[pos..], &mut out);
        }
        out
    }

    fn decode(&self, ids: &[u32]) -> Result<String, String> {
        let mut raw = Vec::new();
        for &i in ids {
            raw.extend(self.token_bytes(i)?);
        }
        Ok(String::from_utf8_lossy(&raw).into_owned())
    }

    fn has_token_bytes(&self) -> bool {
        true
    }

    fn token_bytes(&self, i: u32) -> Result<Vec<u8>, String> {
        if let Some(b) = self.bytes_cache.read().unwrap().get(&i) {
            return Ok(b.clone());
        }
        let tok = self
            .tokens
            .get(i as usize)
            .ok_or_else(|| format!("token id {i} is outside the vocabulary ({})", self.tokens.len()))?;
        let mut raw = Vec::new();
        for ch in tok.chars() {
            match self.byte_dec.get(&ch) {
                Some(&v) => raw.push(v),
                None => {
                    if self.special.get(tok) == Some(&i) {
                        raw = tok.as_bytes().to_vec();
                        break;
                    }
                    return Err(format!("token {i} contains a character outside the byte alphabet: {ch:?}"));
                }
            }
        }
        self.bytes_cache.write().unwrap().insert(i, raw.clone());
        Ok(raw)
    }

    fn control_tokens(&self) -> Option<Vec<String>> {
        None
    }

    fn special_tokens(&self) -> Vec<String> {
        self.special_order.clone()
    }

    fn vocab_size(&self) -> usize {
        self.tokens.len()
    }
}
