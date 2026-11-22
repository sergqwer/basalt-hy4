//! Python's http.server.BaseHTTPRequestHandler (3.12) and ThreadingHTTPServer, as serve/handler.py uses them:
//! HTTP/1.0 responses, one request per connection, the same request-line and header parsing, the same error pages.
use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

pub const PROTOCOL_VERSION: &str = "HTTP/1.0";
/// `Server:` (Python: "BaseHTTP/0.6 Python/3.x.y")
pub const SERVER_VERSION: &str = "BaseHTTP/0.6 basalt-serve-rs/0.1";
pub const MAX_LINE: usize = 65536;
pub const MAX_HEADERS: usize = 100;

const ERROR_MESSAGE_FORMAT: &str = "<!DOCTYPE HTML>\n<html lang=\"en\">\n    <head>\n        <meta charset=\"utf-8\">\n        <title>Error response</title>\n    </head>\n    <body>\n        <h1>Error response</h1>\n        <p>Error code: %(code)d</p>\n        <p>Message: %(message)s.</p>\n        <p>Error code explanation: %(code)s - %(explain)s.</p>\n    </body>\n</html>\n";

/// BaseHTTPRequestHandler.responses: (phrase, explanation)
pub fn responses(code: u16) -> (&'static str, &'static str) {
    match code {
        100 => ("Continue", "Request received, please continue"),
        200 => ("OK", "Request fulfilled, document follows"),
        201 => ("Created", "Document created, URL follows"),
        202 => ("Accepted", "Request accepted, processing continues off-line"),
        204 => ("No Content", "Request fulfilled, nothing follows"),
        400 => ("Bad Request", "Bad request syntax or unsupported method"),
        401 => ("Unauthorized", "No permission -- see authorization schemes"),
        403 => ("Forbidden", "Request forbidden -- authorization will not help"),
        404 => ("Not Found", "Nothing matches the given URI"),
        405 => ("Method Not Allowed", "Specified method is invalid for this resource"),
        408 => ("Request Timeout", "Request timed out; try again later"),
        409 => ("Conflict", "Request conflict"),
        413 => ("Request Entity Too Large", "Entity is too large"),
        414 => ("Request-URI Too Long", "URI is too long"),
        415 => ("Unsupported Media Type", "Entity body in unsupported format"),
        422 => ("Unprocessable Entity", ""),
        429 => ("Too Many Requests", "The user has sent too many requests in a given amount of time (\"rate limiting\")"),
        431 => ("Request Header Fields Too Large", "The server is unwilling to process the request because its header fields are too large"),
        500 => ("Internal Server Error", "Server got itself in trouble"),
        501 => ("Not Implemented", "Server does not support this operation"),
        502 => ("Bad Gateway", "Invalid responses from another server/proxy"),
        503 => ("Service Unavailable", "The server cannot process the request due to a high load"),
        504 => ("Gateway Timeout", "The gateway server did not receive a timely response"),
        505 => ("HTTP Version Not Supported", "Cannot fulfill request"),
        _ => ("???", "???"),
    }
}

/// The request's headers as email.message.Message holds them (names as sent, values as read).
#[derive(Debug, Clone, Default)]
pub struct Headers {
    pub items: Vec<(String, String)>,
    /// a malformed line ended the header block (`headers.defects` / `get_payload()`)
    pub defect: bool,
}

impl Headers {
    pub fn get(&self, name: &str) -> Option<&str> {
        self.items.iter().find(|(k, _)| k.eq_ignore_ascii_case(name)).map(|(_, v)| v.as_str())
    }
    pub fn get_or<'a>(&'a self, name: &str, default: &'a str) -> &'a str {
        self.get(name).unwrap_or(default)
    }
    pub fn get_all(&self, name: &str) -> Vec<&str> {
        self.items.iter().filter(|(k, _)| k.eq_ignore_ascii_case(name)).map(|(_, v)| v.as_str()).collect()
    }
}

fn latin1(b: &[u8]) -> String {
    b.iter().map(|&c| c as char).collect()
}

/// email.feedparser's header-line test: `^(From |[\041-\071\073-\176]*:|[\t ])`
fn header_line_kind(line: &str) -> Option<bool> {
    if line.starts_with("From ") {
        return Some(false); // a unix-from line: skipped as a header
    }
    if line.starts_with(' ') || line.starts_with('\t') {
        return Some(true); // continuation
    }
    for c in line.chars() {
        if c == ':' {
            return Some(false);
        }
        let n = c as u32;
        if !(0x21..=0x7e).contains(&n) {
            return None;
        }
    }
    None
}

/// email's parse of the header block lines (each with its own line end).
fn parse_header_block(lines: &[String]) -> Headers {
    let mut h = Headers::default();
    let mut cur: Option<(String, String)> = None;
    for (i, line) in lines.iter().enumerate() {
        let is_cont = line.starts_with(' ') || line.starts_with('\t');
        match header_line_kind(line) {
            Some(true) if is_cont => {
                if let Some((_, v)) = cur.as_mut() {
                    v.push_str(line);
                    continue;
                }
                // a continuation with no header before it: a defect, the line is kept as a header-less value
                h.defect = true;
                continue;
            }
            Some(_) if line.starts_with("From ") && i == 0 => continue,
            Some(_) => {
                if let Some((k, v)) = cur.take() {
                    h.items.push((k, v.trim_end_matches(['\r', '\n']).to_string()));
                }
                let (name, value) = line.split_once(':').unwrap();
                cur = Some((name.to_string(), value.trim_start_matches([' ', '\t']).to_string()));
            }
            None => {
                // MissingHeaderBodySeparatorDefect: this line and the rest are the payload
                h.defect = true;
                break;
            }
        }
    }
    if let Some((k, v)) = cur.take() {
        h.items.push((k, v.trim_end_matches(['\r', '\n']).to_string()));
    }
    h
}

/// One connection: the buffered reader (rfile) and the writer (wfile).
pub struct Conn {
    pub reader: BufReader<TcpStream>,
    pub writer: TcpStream,
    pub peer: Option<SocketAddr>,
}

impl Conn {
    /// rfile.readline(limit): up to and including "\n", at most `limit` bytes
    pub fn readline(&mut self, limit: usize) -> io::Result<Vec<u8>> {
        let mut out = Vec::new();
        while out.len() < limit {
            let buf = self.reader.fill_buf()?;
            if buf.is_empty() {
                break;
            }
            let take = buf.len().min(limit - out.len());
            if let Some(i) = buf[..take].iter().position(|&b| b == b'\n') {
                out.extend_from_slice(&buf[..=i]);
                self.reader.consume(i + 1);
                return Ok(out);
            }
            out.extend_from_slice(&buf[..take]);
            self.reader.consume(take);
        }
        Ok(out)
    }
    /// rfile.read(n): n bytes or fewer at EOF
    pub fn read(&mut self, n: usize) -> io::Result<Vec<u8>> {
        let mut out = Vec::with_capacity(n.min(1 << 20));
        let mut chunk = vec![0u8; 64 << 10];
        while out.len() < n {
            let want = (n - out.len()).min(chunk.len());
            let got = self.reader.read(&mut chunk[..want])?;
            if got == 0 {
                break;
            }
            out.extend_from_slice(&chunk[..got]);
        }
        Ok(out)
    }
    pub fn set_timeout(&self, t: Option<Duration>) {
        let _ = self.writer.set_read_timeout(t);
        let _ = self.writer.set_write_timeout(t);
    }
}

/// What a request handler sees: the parsed request and the response writer (the BaseHTTPRequestHandler instance).
pub struct Req {
    pub conn: Conn,
    pub command: String,
    pub path: String,
    pub request_version: String,
    pub requestline: String,
    pub headers: Headers,
    pub close_connection: bool,
    headers_buffer: Vec<u8>,
    /// written by end_headers() before the blank line (handler.py's X-Request-Id)
    pub end_header_extra: Option<(String, String)>,
}

impl Req {
    pub fn client_address(&self) -> Option<&SocketAddr> {
        self.conn.peer.as_ref()
    }

    pub fn send_response(&mut self, code: u16, message: Option<&str>) {
        self.send_response_only(code, message);
        self.send_header("Server", SERVER_VERSION);
        self.send_header("Date", &date_time_string());
    }

    pub fn send_response_only(&mut self, code: u16, message: Option<&str>) {
        if self.request_version != "HTTP/0.9" {
            let msg = message.unwrap_or(responses(code).0);
            self.headers_buffer.extend_from_slice(&latin1_bytes(&format!("{PROTOCOL_VERSION} {code} {msg}\r\n")));
        }
    }

    pub fn send_header(&mut self, keyword: &str, value: &str) {
        if self.request_version != "HTTP/0.9" {
            self.headers_buffer.extend_from_slice(&latin1_bytes(&format!("{keyword}: {value}\r\n")));
        }
        if keyword.eq_ignore_ascii_case("connection") {
            if value.eq_ignore_ascii_case("close") {
                self.close_connection = true;
            } else if value.eq_ignore_ascii_case("keep-alive") {
                self.close_connection = false;
            }
        }
    }

    pub fn end_headers(&mut self) -> io::Result<()> {
        if let Some((k, v)) = self.end_header_extra.clone() {
            self.send_header(&k, &v);
        }
        if self.request_version != "HTTP/0.9" {
            self.headers_buffer.extend_from_slice(b"\r\n");
        }
        self.flush_headers()
    }

    pub fn flush_headers(&mut self) -> io::Result<()> {
        let buf = std::mem::take(&mut self.headers_buffer);
        if !buf.is_empty() {
            self.conn.writer.write_all(&buf)?;
        }
        Ok(())
    }

    /// wfile.write
    pub fn write(&mut self, data: &[u8]) -> io::Result<()> {
        self.conn.writer.write_all(data)
    }

    pub fn flush(&mut self) -> io::Result<()> {
        self.conn.writer.flush()
    }

    /// send_error(code, message, explain): the HTML error page, Connection: close
    pub fn send_error(&mut self, code: u16, message: Option<&str>, explain: Option<&str>) -> io::Result<()> {
        let (short, long) = responses(code);
        let message = message.unwrap_or(short).to_string();
        let explain = explain.unwrap_or(long).to_string();
        self.send_response(code, Some(&message));
        self.send_header("Connection", "close");
        let mut body = None;
        if code >= 200 && ![204, 205, 304].contains(&code) {
            let content = ERROR_MESSAGE_FORMAT
                .replace("%(code)d", &code.to_string())
                .replace("%(code)s", &code.to_string())
                .replace("%(message)s", &html_escape(&message))
                .replace("%(explain)s", &html_escape(&explain));
            let b = content.into_bytes();
            self.send_header("Content-Type", "text/html;charset=utf-8");
            self.send_header("Content-Length", &b.len().to_string());
            body = Some(b);
        }
        self.end_headers()?;
        if self.command != "HEAD" {
            if let Some(b) = body {
                self.write(&b)?;
            }
        }
        Ok(())
    }
}

fn latin1_bytes(s: &str) -> Vec<u8> {
    // Python encodes header lines as latin-1 (strict); a char beyond it would raise: written as '?' here
    s.chars().map(|c| if (c as u32) < 256 { c as u32 as u8 } else { b'?' }).collect()
}

/// html.escape(s, quote=False)
pub fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

/// email.utils.formatdate(usegmt=True)
pub fn date_time_string() -> String {
    let secs = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0);
    let days = secs.div_euclid(86400);
    let rem = secs.rem_euclid(86400);
    let (h, m, s) = (rem / 3600, (rem % 3600) / 60, rem % 60);
    let wd = ["Thu", "Fri", "Sat", "Sun", "Mon", "Tue", "Wed"][days.rem_euclid(7) as usize];
    // civil from days (Howard Hinnant)
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z.rem_euclid(146097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let mo = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if mo <= 2 { y + 1 } else { y };
    let mon = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][(mo - 1) as usize];
    format!("{wd}, {d:02} {mon} {y:04} {h:02}:{m:02}:{s:02} GMT")
}

/// The application side: handler.py's do_GET / do_POST / do_OPTIONS and its parse_request extension.
pub trait App: Send + Sync + 'static {
    /// After the base parse succeeded: false when the request was answered (handler.py's Host check).
    fn parse_request_extra(&self, req: &mut Req) -> bool;
    /// do_<command>; false when the handler has no such method (a 501).
    fn dispatch(&self, req: &mut Req) -> bool;
    /// an exception escaped the handler (socketserver.handle_error)
    fn handle_error(&self, _peer: Option<&SocketAddr>, _what: &str) {}
}

pub const HTTP_BODY_S: f64 = 60.0;

enum Parsed {
    Ok,
    Answered,
}

fn parse_request(req: &mut Req, raw: &[u8]) -> io::Result<Parsed> {
    req.command.clear();
    req.request_version = "HTTP/0.9".into();
    req.close_connection = true;
    let requestline = latin1(raw).trim_end_matches(['\r', '\n']).to_string();
    req.requestline = requestline.clone();
    let words: Vec<&str> = requestline.split_whitespace().collect();
    if words.is_empty() {
        return Ok(Parsed::Answered);
    }
    if words.len() >= 3 {
        let version = words[words.len() - 1];
        let bad = || -> bool {
            let Some(base) = version.strip_prefix("HTTP/") else { return true };
            let parts: Vec<&str> = base.split('.').collect();
            if parts.len() != 2 {
                return true;
            }
            if parts.iter().any(|p| p.is_empty() || !p.chars().all(|c| c.is_ascii_digit()) || p.len() > 10) {
                return true;
            }
            false
        };
        if !version.starts_with("HTTP/") || bad() {
            req.send_error(400, Some(&format!("Bad request version ({})", serve_text::pyjson::py_str_repr(version))), None)?;
            return Ok(Parsed::Answered);
        }
        req.request_version = version.to_string();
    }
    if !(2..=3).contains(&words.len()) {
        req.send_error(400, Some(&format!("Bad request syntax ({})", serve_text::pyjson::py_str_repr(&requestline))), None)?;
        return Ok(Parsed::Answered);
    }
    let (command, path) = (words[0].to_string(), words[1].to_string());
    if words.len() == 2 {
        req.close_connection = true;
        if command != "GET" {
            req.command = command.clone();
            req.send_error(400, Some(&format!("Bad HTTP/0.9 request type ({})", serve_text::pyjson::py_str_repr(&command))), None)?;
            return Ok(Parsed::Answered);
        }
    }
    req.command = command;
    req.path = if path.starts_with("//") { format!("/{}", path.trim_start_matches('/')) } else { path };
    // http.client.parse_headers
    let mut lines: Vec<String> = Vec::new();
    loop {
        let line = req.conn.readline(MAX_LINE + 1)?;
        if line.len() > MAX_LINE {
            req.send_error(431, Some("Line too long"), Some("header line"))?;
            return Ok(Parsed::Answered);
        }
        let done = line == b"\r\n" || line == b"\n" || line.is_empty();
        if !done {
            lines.push(latin1(&line));
        }
        if lines.len() > MAX_HEADERS {
            req.send_error(431, Some("Too many headers"), Some(&format!("got more than {MAX_HEADERS} headers")))?;
            return Ok(Parsed::Answered);
        }
        if done {
            break;
        }
    }
    req.headers = parse_header_block(&lines);
    let conntype = req.headers.get("Connection").unwrap_or("").to_lowercase();
    if conntype == "close" {
        req.close_connection = true;
    }
    Ok(Parsed::Ok)
}

fn handle_one(app: &dyn App, req: &mut Req) -> io::Result<()> {
    let raw = match req.conn.readline(MAX_LINE + 1) {
        Ok(r) => r,
        Err(e) if is_timeout(&e) => return Ok(()),
        Err(e) => return Err(e),
    };
    if raw.len() > MAX_LINE {
        req.requestline.clear();
        req.request_version.clear();
        req.command.clear();
        req.send_error(414, None, None)?;
        return Ok(());
    }
    if raw.is_empty() {
        req.close_connection = true;
        return Ok(());
    }
    match parse_request(req, &raw) {
        Ok(Parsed::Answered) => return Ok(()),
        Ok(Parsed::Ok) => {}
        Err(e) if is_timeout(&e) => return Ok(()),
        Err(e) => return Err(e),
    }
    if !app.parse_request_extra(req) {
        return Ok(());
    }
    if !app.dispatch(req) {
        let cmd = req.command.clone();
        req.send_error(501, Some(&format!("Unsupported method ({})", serve_text::pyjson::py_str_repr(&cmd))), None)?;
        return Ok(());
    }
    let _ = req.flush();
    Ok(())
}

pub fn is_timeout(e: &io::Error) -> bool {
    matches!(e.kind(), io::ErrorKind::TimedOut | io::ErrorKind::WouldBlock)
}

fn serve_conn(app: Arc<dyn App>, stream: TcpStream, peer: Option<SocketAddr>) {
    let Ok(writer) = stream.try_clone() else { return };
    let conn = Conn { reader: BufReader::with_capacity(64 << 10, stream), writer, peer };
    conn.set_timeout(Some(Duration::from_secs_f64(HTTP_BODY_S)));
    let mut req = Req {
        conn,
        command: String::new(),
        path: String::new(),
        request_version: "HTTP/0.9".into(),
        requestline: String::new(),
        headers: Headers::default(),
        close_connection: true,
        headers_buffer: Vec::new(),
        end_header_extra: None,
    };
    let r = handle_one(&*app, &mut req);
    if let Err(e) = r {
        if !matches!(e.kind(), io::ErrorKind::ConnectionReset | io::ErrorKind::ConnectionAborted | io::ErrorKind::BrokenPipe) {
            app.handle_error(peer.as_ref(), &e.to_string());
        }
    }
    // socketserver.shutdown_request: SHUT_WR, then close
    let _ = req.conn.writer.shutdown(std::net::Shutdown::Write);
}

/// The listening server (ThreadingHTTPServer): one thread per connection.
pub struct Server {
    pub addr: SocketAddr,
    listener: TcpListener,
    stop: Arc<AtomicBool>,
}

/// Bind `host:port` exclusively (never SO_REUSEADDR on Windows: it stacks a second listener), backlog 256, IPv4 only
/// as Python's AF_INET server.
pub fn bind(host: &str, port: u16) -> io::Result<TcpListener> {
    let addrs: Vec<SocketAddr> = (host, port).to_socket_addrs()?.filter(|a| a.is_ipv4()).collect();
    let addr = *addrs.first().ok_or_else(|| io::Error::new(io::ErrorKind::AddrNotAvailable, "no IPv4 address"))?;
    let sock = socket2::Socket::new(socket2::Domain::IPV4, socket2::Type::STREAM, Some(socket2::Protocol::TCP))?;
    #[cfg(windows)]
    {
        use std::os::windows::io::AsRawSocket;
        use windows_sys::Win32::Networking::WinSock::{setsockopt, SOCKET, SOL_SOCKET, SO_EXCLUSIVEADDRUSE};
        let one: i32 = 1;
        unsafe {
            setsockopt(sock.as_raw_socket() as SOCKET, SOL_SOCKET, SO_EXCLUSIVEADDRUSE, &one as *const i32 as *const u8, 4);
        }
    }
    #[cfg(unix)]
    sock.set_reuse_address(true)?;
    sock.bind(&addr.into())?;
    sock.listen(256)?;
    Ok(sock.into())
}

impl Server {
    pub fn new(host: &str, port: u16) -> io::Result<Server> {
        let listener = bind(host, port)?;
        let addr = listener.local_addr()?;
        Ok(Server { addr, listener, stop: Arc::new(AtomicBool::new(false)) })
    }

    /// serve_forever on a thread
    pub fn start(&self, app: Arc<dyn App>) -> io::Result<std::thread::JoinHandle<()>> {
        let l = self.listener.try_clone()?;
        let stop = self.stop.clone();
        Ok(std::thread::Builder::new().name("http".into()).spawn(move || {
            for s in l.incoming() {
                if stop.load(Ordering::SeqCst) {
                    break;
                }
                let Ok(s) = s else { continue };
                let peer = s.peer_addr().ok();
                let app = app.clone();
                let _ = std::thread::Builder::new().name("http-conn".into()).spawn(move || serve_conn(app, s, peer));
            }
        })?)
    }

    /// shutdown(): no new connections
    pub fn shutdown(&self) {
        self.stop.store(true, Ordering::SeqCst);
        let _ = TcpStream::connect_timeout(&self.addr, Duration::from_millis(200));
    }
}

/// Whether the peer closed: a readable socket with nothing to read (handler.py's watcher).
pub fn peer_gone(s: &TcpStream) -> Option<bool> {
    let mut b = [0u8; 1];
    if !readable(s)? {
        return Some(false);
    }
    match s.peek(&mut b) {
        Ok(0) => Some(true),
        Ok(_) => Some(false),
        Err(e) if matches!(e.kind(), io::ErrorKind::ConnectionReset | io::ErrorKind::ConnectionAborted | io::ErrorKind::TimedOut) => Some(true),
        Err(e) if is_timeout(&e) => Some(false),
        Err(_) => None,
    }
}

fn readable(s: &TcpStream) -> Option<bool> {
    #[cfg(windows)]
    unsafe {
        use std::os::windows::io::AsRawSocket;
        use windows_sys::Win32::Networking::WinSock::{WSAPoll, POLLRDNORM, SOCKET, WSAPOLLFD};
        let mut fd = WSAPOLLFD { fd: s.as_raw_socket() as SOCKET, events: POLLRDNORM, revents: 0 };
        let r = WSAPoll(&mut fd, 1, 0);
        if r < 0 {
            return None;
        }
        Some(r > 0)
    }
    #[cfg(unix)]
    unsafe {
        use std::os::unix::io::AsRawFd;
        let mut fd = libc::pollfd { fd: s.as_raw_fd(), events: libc::POLLIN, revents: 0 };
        let r = libc::poll(&mut fd, 1, 0);
        if r < 0 {
            return None;
        }
        Some(r > 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn headers() {
        let h = parse_header_block(&["Host: a\r\n".into(), "X-Y:  b \r\n".into(), " c\r\n".into(), "bad line\r\n".into(), "Z: 1\r\n".into()]);
        assert_eq!(h.get("host"), Some("a"));
        assert_eq!(h.get("x-y"), Some("b \r\n c"));
        assert!(h.defect);
        assert_eq!(h.get("z"), None);
        assert!(date_time_string().ends_with(" GMT"));
    }
}
