/**
 * Deterministic URL parser — TypeScript port of `phishdetect/urlparse.py`.
 *
 * The browser's built-in `URL` API diverges from Python's `urllib.parse` on
 * exactly the inputs a phishing detector cares about (hosts with `@`, missing
 * schemes, IPv6 literals, trailing dots). To get byte-identical feature vectors
 * on both sides, this file re-implements the same explicit algorithm the Python
 * module uses — it does NOT call the `URL` constructor.
 *
 * Keep this in lockstep with `urlparse.py`; the shared `parity_urls.json`
 * fixture fails if they diverge.
 */

/** The deterministic decomposition of a URL string. Mirrors Python ParsedURL. */
export interface ParsedURL {
  /** Input after whitespace trimming, before any scheme was synthesised. */
  readonly original: string
  /** The full URL actually parsed (with a synthesised scheme if one was added). */
  readonly href: string
  /** Lowercased scheme without `://`. */
  readonly scheme: string
  /** Text before the last `@` in the authority (`''` if absent). */
  readonly userinfo: string
  /** Lowercased host, trailing dot removed; IPv6 keeps its `[...]`. */
  readonly host: string
  /** Port string if present, else `''`. */
  readonly port: string
  /** Path component including its leading `/` (`''` if absent). */
  readonly path: string
  /** Raw query string without the leading `?` (`''` if absent). */
  readonly query: string
  /** Raw fragment without the leading `#` (`''` if absent). */
  readonly fragment: string
  /** True if the input already carried a `scheme://` prefix. */
  readonly hadScheme: boolean
}

/** Characters that terminate the authority component. */
const AUTHORITY_TERMINATORS = ['/', '?', '#'] as const

function isAlpha(ch: string): boolean {
  return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z')
}

function isDigit(ch: string): boolean {
  return ch >= '0' && ch <= '9'
}

/**
 * Return true if `text` starts with a `scheme://` prefix.
 * A scheme is `ALPHA *( ALPHA / DIGIT / "+" / "-" / "." )` followed by `://`.
 */
function hasScheme(text: string): boolean {
  const idx = text.indexOf('://')
  if (idx <= 0) {
    return false
  }
  const scheme = text.slice(0, idx)
  if (!isAlpha(scheme[0])) {
    return false
  }
  for (const ch of scheme) {
    const ok = isAlpha(ch) || isDigit(ch) || ch === '+' || ch === '-' || ch === '.'
    if (!ok) {
      return false
    }
  }
  return true
}

/**
 * Split a `host[:port]` string into `[host, port]`.
 * The split is on the last `:` not inside an IPv6 `[...]` literal; a non-numeric
 * "port" is treated as part of the host.
 */
function splitAuthorityPort(hostPort: string): [string, string] {
  if (hostPort.startsWith('[')) {
    const close = hostPort.indexOf(']')
    if (close !== -1) {
      const host = hostPort.slice(0, close + 1)
      const rest = hostPort.slice(close + 1)
      if (rest.startsWith(':')) {
        return [host, rest.slice(1)]
      }
      return [host, '']
    }
  }
  const colon = hostPort.lastIndexOf(':')
  if (colon === -1) {
    return [hostPort, '']
  }
  const candidatePort = hostPort.slice(colon + 1)
  if (candidatePort !== '' && [...candidatePort].every(isDigit)) {
    return [hostPort.slice(0, colon), candidatePort]
  }
  return [hostPort, '']
}

/**
 * Parse `raw` into a {@link ParsedURL} using the shared algorithm.
 * URLs with no `scheme://` prefix have `http://` prepended first, exactly as
 * the Python `parse_url` does.
 */
export function parseUrl(raw: string): ParsedURL {
  const original = raw.trim()

  const hadScheme = hasScheme(original)
  const href = hadScheme ? original : 'http://' + original

  // 1. Scheme.
  const schemeEnd = href.indexOf('://')
  const scheme = href.slice(0, schemeEnd).toLowerCase()
  const afterScheme = href.slice(schemeEnd + 3)

  // 2. Authority = up to the first terminator.
  let authorityEnd = afterScheme.length
  for (const term of AUTHORITY_TERMINATORS) {
    const pos = afterScheme.indexOf(term)
    if (pos !== -1 && pos < authorityEnd) {
      authorityEnd = pos
    }
  }
  const authority = afterScheme.slice(0, authorityEnd)
  const rest = afterScheme.slice(authorityEnd)

  // 3. Userinfo splits on the LAST '@'.
  let userinfo = ''
  let hostPort = authority
  const at = authority.lastIndexOf('@')
  if (at !== -1) {
    userinfo = authority.slice(0, at)
    hostPort = authority.slice(at + 1)
  }

  // 4. Host / port.
  let [host, port] = splitAuthorityPort(hostPort)
  host = host.toLowerCase()
  if (host.endsWith('.') && !host.endsWith(']')) {
    host = host.slice(0, -1)
  }

  // 5. Path / query / fragment.
  let query = ''
  let fragment = ''
  let path = rest

  const hashPos = path.indexOf('#')
  if (hashPos !== -1) {
    fragment = path.slice(hashPos + 1)
    path = path.slice(0, hashPos)
  }

  const qPos = path.indexOf('?')
  if (qPos !== -1) {
    query = path.slice(qPos + 1)
    path = path.slice(0, qPos)
  }

  return {
    original,
    href,
    scheme,
    userinfo,
    host,
    port,
    path,
    query,
    fragment,
    hadScheme,
  }
}
