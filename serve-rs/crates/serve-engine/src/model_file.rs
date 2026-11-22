//! tools/model_file.py: the .basalt model file container (layout in include/basalt/core/model_format.hpp). The reader
//! (ModelFile) in full, and ModelWriter's add_bytes/close (enough to build files in tests); the tools' streaming adds,
//! append_section and undo_append are not ported.
use std::fmt;
use std::io::{Read, Seek, SeekFrom, Write};

pub const MAGIC: &[u8; 8] = b"BASALTMF";
pub const VERSION: u32 = 2;
pub const MIGRATION: &str = "tools/migrations/basalt_v1_to_v2.py";
pub const SECTION_ALIGN: u64 = 2 << 20;
pub const HEADER_BYTES: usize = 64 << 10;
pub const HEADER_SIZE: usize = 64;
pub const ENTRY_SIZE: usize = 128;
pub const MAX_SECTIONS: usize = (HEADER_BYTES - HEADER_SIZE) / ENTRY_SIZE;
pub const COPY_CHUNK: usize = 64 << 20;

/// What Python raises, by its exception class; Display is `str(e)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MfError {
    Os(String),
    Value(String),
    Key(String),
    /// struct.error (a file shorter than its header): Python does not catch it.
    Struct(String),
}

impl fmt::Display for MfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MfError::Os(s) | MfError::Value(s) | MfError::Struct(s) => f.write_str(s),
            MfError::Key(k) => f.write_str(&crate::pyfmt::repr_str(k)),
        }
    }
}

impl std::error::Error for MfError {}

/// An OSError's `str()`: `[Errno 2] No such file or directory: 'path'`.
pub fn os_error(e: &std::io::Error, path: &str) -> MfError {
    let (no, text) = match e.kind() {
        std::io::ErrorKind::NotFound => (2, "No such file or directory".to_string()),
        std::io::ErrorKind::PermissionDenied => (13, "Permission denied".to_string()),
        std::io::ErrorKind::IsADirectory => (21, "Is a directory".to_string()),
        _ => (e.raw_os_error().unwrap_or(5), e.to_string()),
    };
    MfError::Os(format!("[Errno {no}] {text}: {}", crate::pyfmt::repr_str(path)))
}

fn align(n: u64) -> u64 {
    n.div_ceil(SECTION_ALIGN) * SECTION_ALIGN
}

/// One table entry: (offset, size, sha256).
pub type Section = (u64, u64, [u8; 32]);

/// A .basalt file's section table.
#[derive(Debug, Clone)]
pub struct ModelFile {
    pub path: String,
    pub version: u32,
    pub sections: Vec<(String, Section)>,       // in table order (a repeated name keeps its first place, last value)
}

fn u32_at(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}

fn u64_at(b: &[u8], o: usize) -> u64 {
    u64::from_le_bytes(b[o..o + 8].try_into().unwrap())
}

fn short(need: usize, at: usize, have: usize) -> MfError {
    MfError::Struct(format!("unpack_from requires a buffer of at least {} bytes for unpacking {need} bytes at offset {at} \
                             (actual buffer size is {have})", at + need))
}

impl ModelFile {
    /// Version 2 only.
    pub fn open(path: &str) -> Result<ModelFile, MfError> {
        Self::open_versions(path, &[VERSION])
    }

    pub fn open_versions(path: &str, versions: &[u32]) -> Result<ModelFile, MfError> {
        let mut head = Vec::with_capacity(HEADER_BYTES);
        std::fs::File::open(path).and_then(|f| f.take(HEADER_BYTES as u64).read_to_end(&mut head))
            .map_err(|e| os_error(&e, path))?;
        if head.len() < HEADER_SIZE {
            return Err(short(HEADER_SIZE, 0, head.len()));
        }
        let (version, n, file_bytes) = (u32_at(&head, 8), u32_at(&head, 12), u64_at(&head, 16));
        if &head[..8] != MAGIC {
            return Err(MfError::Value(format!("{path}: not a .basalt model file")));
        }
        if !versions.contains(&version) {
            let hint = if version < VERSION { format!("; upgrade it with {MIGRATION}") } else { String::new() };
            return Err(MfError::Value(format!("{path}: .basalt version {version}, this build reads version {VERSION}{hint}")));
        }
        let size = std::fs::metadata(path).map_err(|e| os_error(&e, path))?.len();
        if size < file_bytes {
            return Err(MfError::Value(format!("{path}: truncated ({size} of {file_bytes} bytes)")));
        }
        let mut sections: Vec<(String, Section)> = Vec::new();
        for i in 0..n as usize {
            let o = HEADER_SIZE + i * ENTRY_SIZE;
            if o + ENTRY_SIZE > head.len() {
                return Err(short(ENTRY_SIZE, o, head.len()));
            }
            let raw = &head[o..o + 64];
            let end = raw.iter().rposition(|b| *b != 0).map(|p| p + 1).unwrap_or(0);
            let name = std::str::from_utf8(&raw[..end])
                .map_err(|e| MfError::Value(format!("'utf-8' codec can't decode bytes: {e}")))?.to_string();
            let digest: [u8; 32] = head[o + 80..o + 112].try_into().unwrap();
            let s = (u64_at(&head, o + 64), u64_at(&head, o + 72), digest);
            match sections.iter_mut().find(|(k, _)| *k == name) {
                Some(e) => e.1 = s,
                None => sections.push((name, s)),
            }
        }
        Ok(ModelFile { path: path.to_string(), version, sections })
    }

    pub fn names(&self) -> Vec<String> {
        self.sections.iter().map(|x| x.0.clone()).collect()
    }

    pub fn has(&self, name: &str) -> bool {
        self.sections.iter().any(|x| x.0 == name)
    }

    /// The table entry of `name` (KeyError when absent).
    pub fn entry(&self, name: &str) -> Result<Section, MfError> {
        self.sections.iter().find(|x| x.0 == name).map(|x| x.1).ok_or_else(|| MfError::Key(name.to_string()))
    }

    /// (offset, size) of `name`.
    pub fn section(&self, name: &str) -> Result<(u64, u64), MfError> {
        self.entry(name).map(|(o, s, _)| (o, s))
    }

    pub fn read(&self, name: &str) -> Result<Vec<u8>, MfError> {
        let (off, size) = self.section(name)?;
        let mut f = std::fs::File::open(&self.path).map_err(|e| os_error(&e, &self.path))?;
        f.seek(SeekFrom::Start(off)).map_err(|e| os_error(&e, &self.path))?;
        let mut b = Vec::new();
        f.take(size).read_to_end(&mut b).map_err(|e| os_error(&e, &self.path))?;
        Ok(b)
    }

    /// Whether the section's bytes match the hash in the table.
    pub fn verify(&self, name: &str) -> Result<bool, MfError> {
        let (off, size, digest) = self.entry(name)?;
        let mut f = std::fs::File::open(&self.path).map_err(|e| os_error(&e, &self.path))?;
        f.seek(SeekFrom::Start(off)).map_err(|e| os_error(&e, &self.path))?;
        let mut h = Sha256::new();
        let mut left = size;
        let mut buf = vec![0u8; COPY_CHUNK.min(size as usize).max(1)];
        while left > 0 {
            let want = (buf.len() as u64).min(left) as usize;
            let got = f.read(&mut buf[..want]).map_err(|e| os_error(&e, &self.path))?;
            if got == 0 {
                return Ok(false);
            }
            h.update(&buf[..got]);
            left -= got as u64;
        }
        Ok(h.finish() == digest)
    }
}

/// Sections appended in order; the header and table are written by close().
pub struct ModelWriter {
    pub path: String,
    f: std::fs::File,
    pos: u64,
    entries: Vec<(String, u64, u64, [u8; 32])>,
}

impl ModelWriter {
    pub fn create(path: &str) -> Result<ModelWriter, MfError> {
        let mut f = std::fs::File::create(path).map_err(|e| os_error(&e, path))?;
        f.write_all(&vec![0u8; HEADER_BYTES]).map_err(|e| os_error(&e, path))?;
        Ok(ModelWriter { path: path.to_string(), f, pos: HEADER_BYTES as u64, entries: Vec::new() })
    }

    fn start(&mut self, name: &str) -> Result<u64, MfError> {
        if name.len() >= 64 {
            return Err(MfError::Value(format!("section name too long: {name}")));
        }
        if self.entries.iter().any(|e| e.0 == name) {
            return Err(MfError::Value(format!("duplicate section: {name}")));
        }
        if self.entries.len() >= MAX_SECTIONS {
            return Err(MfError::Value("too many sections".into()));
        }
        let off = align(self.pos);
        self.f.write_all(&vec![0u8; (off - self.pos) as usize]).map_err(|e| os_error(&e, &self.path))?;
        self.pos = off;
        Ok(off)
    }

    pub fn add_bytes(&mut self, name: &str, data: &[u8]) -> Result<(), MfError> {
        let off = self.start(name)?;
        self.f.write_all(data).map_err(|e| os_error(&e, &self.path))?;
        self.pos += data.len() as u64;
        let mut h = Sha256::new();
        h.update(data);
        self.entries.push((name.to_string(), off, data.len() as u64, h.finish()));
        Ok(())
    }

    pub fn close(mut self) -> Result<(), MfError> {
        let mut head = Vec::with_capacity(HEADER_SIZE + self.entries.len() * ENTRY_SIZE);
        head.extend_from_slice(MAGIC);
        head.extend_from_slice(&VERSION.to_le_bytes());
        head.extend_from_slice(&(self.entries.len() as u32).to_le_bytes());
        head.extend_from_slice(&self.pos.to_le_bytes());
        head.extend_from_slice(&[0u8; 40]);
        for (name, off, n, digest) in &self.entries {
            let mut nb = [0u8; 64];
            nb[..name.len()].copy_from_slice(name.as_bytes());
            head.extend_from_slice(&nb);
            head.extend_from_slice(&off.to_le_bytes());
            head.extend_from_slice(&n.to_le_bytes());
            head.extend_from_slice(digest);
            head.extend_from_slice(&[0u8; 16]);
        }
        let p = self.path.clone();
        self.f.seek(SeekFrom::Start(0)).and_then(|_| self.f.write_all(&head)).map_err(|e| os_error(&e, &p))
    }
}

/// SHA-256 (FIPS 180-4), for the section hashes.
pub struct Sha256 {
    h: [u32; 8],
    buf: Vec<u8>,
    len: u64,
}

const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01,
    0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08,
    0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

impl Default for Sha256 {
    fn default() -> Self {
        Self::new()
    }
}

impl Sha256 {
    pub fn new() -> Self {
        Sha256 { h: [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19],
                 buf: Vec::with_capacity(64), len: 0 }
    }

    fn block(&mut self, b: &[u8]) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes(b[i * 4..i * 4 + 4].try_into().unwrap());
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }
        let mut v = self.h;
        for i in 0..64 {
            let s1 = v[4].rotate_right(6) ^ v[4].rotate_right(11) ^ v[4].rotate_right(25);
            let ch = (v[4] & v[5]) ^ (!v[4] & v[6]);
            let t1 = v[7].wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0 = v[0].rotate_right(2) ^ v[0].rotate_right(13) ^ v[0].rotate_right(22);
            let maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
            let t2 = s0.wrapping_add(maj);
            v = [t1.wrapping_add(t2), v[0], v[1], v[2], v[3].wrapping_add(t1), v[4], v[5], v[6]];
        }
        for i in 0..8 {
            self.h[i] = self.h[i].wrapping_add(v[i]);
        }
    }

    pub fn update(&mut self, mut data: &[u8]) {
        self.len += data.len() as u64;
        if !self.buf.is_empty() {
            let take = (64 - self.buf.len()).min(data.len());
            self.buf.extend_from_slice(&data[..take]);
            data = &data[take..];
            if self.buf.len() == 64 {
                let b = std::mem::take(&mut self.buf);
                self.block(&b);
            }
        }
        while data.len() >= 64 {
            self.block(&data[..64]);
            data = &data[64..];
        }
        self.buf.extend_from_slice(data);
    }

    pub fn finish(mut self) -> [u8; 32] {
        let bits = self.len.wrapping_mul(8);
        let mut pad = vec![0x80u8];
        pad.resize((55usize.wrapping_sub(self.buf.len()) % 64) + 1, 0);
        pad.extend_from_slice(&bits.to_be_bytes());
        let len = self.len;
        self.update(&pad);
        self.len = len;
        let mut out = [0u8; 32];
        for (i, h) in self.h.iter().enumerate() {
            out[i * 4..i * 4 + 4].copy_from_slice(&h.to_be_bytes());
        }
        out
    }
}

/// Lowercase hex of bytes.
pub fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_vectors() {
        let h = |b: &[u8]| {
            let mut s = Sha256::new();
            s.update(b);
            hex(&s.finish())
        };
        assert_eq!(h(b""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
        assert_eq!(h(b"abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        assert_eq!(h(&[b'a'; 1000]), "41edece42d63e8d9bf515a9ba6932e1c20cbc9f5a5d134645adb5db1b9737ea3");
    }

    #[test]
    fn round_trip_and_errors() {
        let d = std::env::temp_dir().join(format!("serve-mf-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let p = d.join("t.basalt").to_string_lossy().into_owned();
        let mut w = ModelWriter::create(&p).unwrap();
        w.add_bytes("meta", b"{}").unwrap();
        w.add_bytes("tokenizer/vocab.json", b"{\"a\": 0}").unwrap();
        assert!(w.add_bytes("meta", b"x").is_err());
        w.close().unwrap();
        let m = ModelFile::open(&p).unwrap();
        assert_eq!(m.names(), ["meta", "tokenizer/vocab.json"]);
        assert_eq!(m.section("meta").unwrap(), (2 << 20, 2));
        assert_eq!(m.section("tokenizer/vocab.json").unwrap().0, 4 << 20);
        assert_eq!(m.read("tokenizer/vocab.json").unwrap(), b"{\"a\": 0}");
        assert!(m.verify("meta").unwrap());
        assert_eq!(m.read("nope").unwrap_err().to_string(), "'nope'");
        let junk = d.join("junk").to_string_lossy().into_owned();
        std::fs::write(&junk, vec![b'x'; 5000]).unwrap();
        assert_eq!(ModelFile::open(&junk).unwrap_err(), MfError::Value(format!("{junk}: not a .basalt model file")));
        let missing = d.join("missing").to_string_lossy().into_owned();
        assert_eq!(ModelFile::open(&missing).unwrap_err().to_string(),
                   format!("[Errno 2] No such file or directory: {}", crate::pyfmt::repr_str(&missing)));
        let _ = std::fs::remove_dir_all(&d);
    }
}
