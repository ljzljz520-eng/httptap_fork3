"""Session-bound secret redaction policy.

A single :class:`RedactionPolicy` is created for an analysis session and
shared by the analyzer (which produces the metrics), the renderers (which
display them), and the exporter (which serializes them). The policy is the
only place that knows which query keys and header names carry secrets.

The policy distinguishes two views of every URL:

* the **raw** value, used exclusively for transport (issuing the request)
  and redirect computation (:func:`urllib.parse.urljoin`), and
* the **safe** view returned by :meth:`RedactionPolicy.redact_url`, which is
  the only form allowed on screen, in :class:`~httptap.models.StepMetrics`,
  or in exported reports.

URL redaction is structural: the query string (and a query-shaped fragment)
is parsed into ``key=value`` pairs so that key spelling, parameter order,
duplicate parameters, empty values, and percent-encoding are all preserved.
Only the values of sensitive parameters are replaced by the mask; the
request itself is never modified.

The policy also *learns* the literal secret values it encounters (sensitive
query values, userinfo passwords, sensitive header values) while the session
runs. :meth:`RedactionPolicy.redact_text` uses both URL rewriting and the
learned literals to scrub free-form failure messages, which may embed the
target URL, a timeout description, or a proxy address verbatim. Learned
values stay in process memory for the lifetime of the policy and are never
displayed or exported themselves.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING
from urllib.parse import unquote, unquote_plus, urlsplit, urlunsplit

from .utils import MASK_PATTERN, SENSITIVE_HEADERS, mask_sensitive_value, redact_url_credentials

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

__all__ = [
    "DEFAULT_SENSITIVE_HEADERS",
    "DEFAULT_SENSITIVE_QUERY_KEYS",
    "RedactionPolicy",
]

#: Header names masked without any user configuration. Mirrors the legacy
#: fixed set used by :func:`httptap.utils.sanitize_headers`.
DEFAULT_SENSITIVE_HEADERS: frozenset[str] = frozenset(SENSITIVE_HEADERS)

#: Query parameter names whose values are secrets by convention. Matching is
#: case-insensitive. Covers bearer-style tokens, API keys, passwords, OAuth
#: artifacts, and the query components of presigned URLs (notably AWS SigV4).
DEFAULT_SENSITIVE_QUERY_KEYS: frozenset[str] = frozenset(
    {
        "access_token",
        "api_key",
        "apikey",
        "client_secret",
        "id_token",
        "passwd",
        "password",
        "refresh_token",
        "secret",
        "sig",
        "signature",
        "token",
        "x-amz-security-token",
        "x-amz-signature",
    }
)

# Matches absolute URIs embedded in free text. The character class is limited
#: to RFC 3986 URI characters so a URL is terminated by whitespace, quotes,
#: angle brackets, or other delimiters rather than consumed greedily.
_URL_IN_TEXT_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9+.-]*://[A-Za-z0-9\-._~%!$&'()*+,;=:@/?#\[\]=]+")

#: Punctuation kept outside a matched URL span when rewriting error text.
_TRAILING_URL_PUNCTUATION = ".,;:!?)]}>"


class RedactionPolicy:
    """Redaction rules bound to one analysis session.

    Args:
        query_keys: Additional sensitive query parameter names. Combined
            with :data:`DEFAULT_SENSITIVE_QUERY_KEYS`; matching is
            case-insensitive and surrounding whitespace is stripped.
        header_names: Additional sensitive request/response header names.
            Combined with :data:`DEFAULT_SENSITIVE_HEADERS`; matching is
            case-insensitive and surrounding whitespace is stripped.

    Examples:
        >>> policy = RedactionPolicy(query_keys=["tenant_key"])
        >>> policy.redact_url("https://api.example/v1?tenant_key=shh&page=2")
        'https://api.example/v1?tenant_key=****&page=2'

    """

    __slots__ = ("_header_names", "_query_keys", "_secrets")

    def __init__(
        self,
        *,
        query_keys: Iterable[str] | None = None,
        header_names: Iterable[str] | None = None,
    ) -> None:
        """Initialize the policy with built-in key sets plus optional extras."""
        self._query_keys: set[str] = set(DEFAULT_SENSITIVE_QUERY_KEYS)
        self._header_names: set[str] = set(DEFAULT_SENSITIVE_HEADERS)
        if query_keys:
            self._query_keys.update(name.strip().lower() for name in query_keys if name.strip())
        if header_names:
            self._header_names.update(name.strip().lower() for name in header_names if name.strip())
        self._secrets: set[str] = set()

    @property
    def query_keys(self) -> frozenset[str]:
        """Frozen view of every sensitive query key (built-in and custom)."""
        return frozenset(self._query_keys)

    @property
    def header_names(self) -> frozenset[str]:
        """Frozen view of every sensitive header name (built-in and custom)."""
        return frozenset(self._header_names)

    def redact_url(self, url: str | None) -> str | None:
        """Return the display-safe view of a URL.

        Masks userinfo credentials (proxy passwords, bare tokens) and the
        values of sensitive query (and query-shaped fragment) parameters.
        Key names, parameter order, duplicates, empty values, and the
        original percent-encoding of every other component are preserved.

        Args:
            url: Raw URL, or ``None``.

        Returns:
            The redacted URL. ``None`` passes through unchanged; a URL that
            cannot be parsed is returned as-is so redaction never raises.

        """
        if not url:
            return url
        try:
            parts = urlsplit(url)
        except ValueError:
            return url

        masked = urlsplit(redact_url_credentials(url))
        query = self._redact_pairs(parts.query)
        fragment = self._redact_pairs(parts.fragment)
        return urlunsplit(masked._replace(query=query, fragment=fragment))

    def redact_headers(self, headers: Mapping[str, str] | None) -> dict[str, str]:
        """Return a copy of headers with sensitive values masked.

        Args:
            headers: Header mapping to sanitize.

        Returns:
            New dictionary preserving key casing and non-sensitive values.

        """
        if not headers:
            return {}
        return {
            name: mask_sensitive_value(value) if name.lower() in self._header_names else value
            for name, value in headers.items()
        }

    def redact_text(self, text: str | None) -> str | None:
        """Scrub secrets from a free-form message such as an exception string.

        Every embedded absolute URL is replaced by its
        :meth:`redact_url` view, and every literal secret value learned by
        this session is masked wherever it appears. Non-sensitive text
        (including ordinary query values like ``page=2``) is untouched.

        Args:
            text: Raw message, or ``None``.

        Returns:
            Sanitized message suitable for display and export.

        """
        if not text:
            return text

        def _replace_url(match: re.Match[str]) -> str:
            span = match.group(0)
            cut = len(span)
            while cut > 0 and span[cut - 1] in _TRAILING_URL_PUNCTUATION:
                cut -= 1
            return f"{self.redact_url(span[:cut])}{span[cut:]}"

        redacted = _URL_IN_TEXT_RE.sub(_replace_url, text)

        for secret in sorted(self._secrets, key=len, reverse=True):
            pattern = rf"(?<![A-Za-z0-9]){re.escape(secret)}(?![A-Za-z0-9])"
            redacted = re.sub(pattern, MASK_PATTERN, redacted)
        return redacted

    def learn_url(self, url: str | None) -> None:
        """Remember secrets carried by a raw URL for later text scrubbing.

        Records sensitive query values (raw and percent-decoded) and
        userinfo credentials. Calling this never modifies the URL.

        Args:
            url: Raw URL the session is about to request, or ``None``.

        """
        if not url:
            return
        try:
            parts = urlsplit(url)
        except ValueError:
            return

        userinfo, separator, _hostport = parts.netloc.rpartition("@")
        if separator:
            username, has_password, password = userinfo.partition(":")
            credential = password if has_password else username
            if credential:
                self._remember(credential)
                self._remember(unquote(credential))

        self._learn_pairs(parts.query)
        self._learn_pairs(parts.fragment)

    def learn_headers(self, headers: Mapping[str, str] | None) -> None:
        """Remember values of sensitive headers for later text scrubbing.

        Args:
            headers: Raw request headers the session is about to send.

        """
        if not headers:
            return
        for name, value in headers.items():
            if name.lower() in self._header_names:
                self._remember(value)

    def _redact_pairs(self, serialized: str) -> str:
        """Mask sensitive values in a ``key=value&key=value`` string."""
        if not serialized:
            return serialized
        return "&".join(self._redact_pair(pair) for pair in serialized.split("&"))

    def _redact_pair(self, pair: str) -> str:
        if not pair:
            return pair
        key, separator, _value = pair.partition("=")
        if separator and unquote_plus(key).lower() in self._query_keys:
            return f"{key}={MASK_PATTERN}"
        return pair

    def _learn_pairs(self, serialized: str) -> None:
        if not serialized:
            return
        for pair in serialized.split("&"):
            key, separator, value = pair.partition("=")
            if not separator:
                continue
            if unquote_plus(key).lower() in self._query_keys:
                self._remember(value)
                self._remember(unquote_plus(value))

    def _remember(self, value: str) -> None:
        if value and value != MASK_PATTERN:
            self._secrets.add(value)
