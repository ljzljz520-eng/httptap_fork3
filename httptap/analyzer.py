"""Main HTTP request analyzer orchestration.

This module coordinates the analysis of HTTP requests, handling redirects,
collecting metrics, and managing the overall request flow.
"""

from __future__ import annotations

from http import HTTPStatus
from typing import TYPE_CHECKING
from urllib.parse import urljoin, urlsplit

if TYPE_CHECKING:
    from httpx._types import ProxyTypes
else:  # pragma: no cover - typing helper
    ProxyTypes = object  # type: ignore[assignment]

from .constants import (
    BODY_HEADERS,
    DEFAULT_TIMEOUT_SECONDS,
    HTTP_DEFAULT_PORT,
    HTTPS_DEFAULT_PORT,
    ORIGIN_BOUND_HEADERS,
    POST_TO_GET_REDIRECT_STATUSES,
    HTTPMethod,
)
from .http_client import HTTPClientError
from .models import StepMetrics
from .redaction import RedactionPolicy
from .request_executor import HTTPClientRequestExecutor, RequestExecutor, RequestOptions, RequestOutcome

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .interfaces import DNSResolver, TimingCollector, TLSInspector


def _origin(url: str) -> tuple[str, str, int]:
    """Return the (scheme, host, port) origin of a URL with default ports filled in."""
    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    default_port = HTTPS_DEFAULT_PORT if scheme == "https" else HTTP_DEFAULT_PORT
    return scheme, (parts.hostname or "").lower(), parts.port or default_port


def _is_same_origin_or_https_upgrade(current_url: str, next_url: str) -> bool:
    """Check whether credentials may be kept when redirecting from current_url to next_url.

    Mirrors httpx: credentials survive a redirect to the same origin and an
    ``http`` to ``https`` upgrade on the same host with default ports.
    """
    current = _origin(current_url)
    target = _origin(next_url)
    if current == target:
        return True
    return current == ("http", target[1], HTTP_DEFAULT_PORT) and target == ("https", current[1], HTTPS_DEFAULT_PORT)


def _redirect_method(status: int, method: HTTPMethod) -> HTTPMethod:
    """Return the method for the next hop per RFC 9110 and curl/browser behavior."""
    if status == HTTPStatus.SEE_OTHER and method != HTTPMethod.HEAD:
        return HTTPMethod.GET
    if status in POST_TO_GET_REDIRECT_STATUSES and method == HTTPMethod.POST:
        return HTTPMethod.GET
    return method


def _redirect_headers(
    headers: Mapping[str, str] | None,
    *,
    keep_credentials: bool,
    keep_body_headers: bool,
) -> dict[str, str] | None:
    """Drop origin-bound credentials and body headers that must not follow a redirect."""
    if headers is None:
        return None
    return {
        name: value
        for name, value in headers.items()
        if (keep_credentials or name.lower() not in ORIGIN_BOUND_HEADERS)
        and (keep_body_headers or name.lower() not in BODY_HEADERS)
    }


class HTTPTapAnalyzer:
    """Orchestrates HTTP request analysis with redirect following.

    This class manages the high-level flow of analyzing HTTP requests,
    including following redirect chains and collecting metrics at each step.

    Attributes:
        follow_redirects: Whether to follow HTTP redirects.
        timeout: Request timeout in seconds.
        http2: Whether to enable HTTP/2 support.
        max_redirects: Maximum number of redirects to follow.

    """

    __slots__ = (
        "_dns_resolver",
        "_noproxy",
        "_policy",
        "_proxy",
        "_request",
        "_timing_collector",
        "_tls_inspector",
        "ca_bundle_path",
        "follow_redirects",
        "http2",
        "max_redirects",
        "timeout",
        "verify_ssl",
    )

    def __init__(  # noqa: PLR0913
        self,
        *,
        follow_redirects: bool = False,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        http2: bool = True,
        verify_ssl: bool = True,
        ca_bundle_path: str | None = None,
        max_redirects: int = 10,
        request_executor: RequestExecutor | None = None,
        proxy: ProxyTypes | None = None,
        noproxy: bool = False,
        dns_resolver: DNSResolver | None = None,
        tls_inspector: TLSInspector | None = None,
        timing_collector_factory: type[TimingCollector] | None = None,
        redaction_policy: RedactionPolicy | None = None,
    ) -> None:
        """Initialize HTTP analyzer.

        Args:
            follow_redirects: Whether to follow 3xx redirects.
            timeout: Request timeout in seconds.
            http2: Enable HTTP/2 support.
            verify_ssl: Whether to verify TLS certificates.
            ca_bundle_path: Path to custom CA certificate bundle (PEM format).
                Only used when verify_ssl is True. If None, uses system CA bundle.
            max_redirects: Maximum number of redirects to follow.
            request_executor: Object responsible for performing HTTP requests.
                Must implement the RequestExecutor protocol. Defaults to the
                built-in httpx implementation.
            proxy: Optional proxy URL (http/https/socks5/socks5h) applied to all
                requests in the analysis chain.
            noproxy: When True, ignore proxy environment variables and connect
                directly. Triggered by --proxy "".
            dns_resolver: Custom DNS resolver implementation. If None, make_request
                will use its default (SystemDNSResolver).
            tls_inspector: Custom TLS inspector implementation. If None, make_request
                will use its default (SocketTLSInspector).
            timing_collector_factory: Factory class for creating timing collectors.
                If None, make_request will use its default (PerfCounterTimingCollector).
                Note: This should be a class, not an instance, as a new collector
                is created for each request.
            redaction_policy: Session policy deciding which query keys and
                header values are masked in every safe view. Defaults to a
                policy with the built-in sensitive key sets. The same instance
                should be shared with the output renderer and JSON exporter.

        """
        self.follow_redirects = follow_redirects
        self.timeout = timeout
        self.http2 = http2
        self.verify_ssl = verify_ssl
        self.ca_bundle_path = ca_bundle_path
        self.max_redirects = max_redirects
        self._request = request_executor or HTTPClientRequestExecutor()
        self._dns_resolver = dns_resolver
        self._tls_inspector = tls_inspector
        self._timing_collector = timing_collector_factory
        self._proxy = proxy
        self._noproxy = noproxy
        self._policy = redaction_policy or RedactionPolicy()
        if proxy is not None:
            self._policy.learn_url(str(getattr(proxy, "url", proxy)))

    def analyze_url(
        self,
        url: str,
        *,
        method: HTTPMethod = HTTPMethod.GET,
        content: bytes | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> list[StepMetrics]:
        """Analyze URL with optional redirect following.

        Performs HTTP request(s) and collects comprehensive metrics.
        If follow_redirects is enabled and server returns 3xx with Location,
        continues following redirects up to max_redirects.

        Args:
            url: Initial URL to analyze. Must be valid HTTP/HTTPS URL.
            method: HTTP method to use (GET, POST, PUT, PATCH, DELETE, HEAD, OPTIONS).
            content: Optional request body as bytes.
            headers: Optional mapping of request headers applied to every step.

        Returns:
            List of StepMetrics, one per request in the chain. Each step contains
            timing, network, and response information. Returns at least one step
            even if request fails.

        Examples:
            Basic usage without redirects:
                >>> analyzer = HTTPTapAnalyzer()
                >>> steps = analyzer.analyze_url("https://example.com")
                >>> print(f"Total time: {steps[0].timing.total_ms}ms")
                Total time: 234.5ms

            Following redirect chain:
                >>> analyzer = HTTPTapAnalyzer(follow_redirects=True)
                >>> steps = analyzer.analyze_url("http://example.com")
                >>> for i, step in enumerate(steps, 1):
                ...     print(f"Step {i}: {step.response.status}")
                Step 1: 301
                Step 2: 200

        """
        steps: list[StepMetrics] = []
        current_url = url
        redirect_count = 0

        while redirect_count <= self.max_redirects:
            step_number = len(steps) + 1
            step = self._analyze_single_request(
                current_url,
                step_number,
                method=method,
                content=content,
                headers=headers,
            )
            steps.append(step)

            # Check if we should follow redirect
            if not self.follow_redirects:
                break

            if step.has_error:
                # Stop on error
                break

            if step.is_redirect:
                # Follow redirect. The raw Location is required for correct
                # relative-reference resolution; the safe view must never be
                # joined (its masked values would corrupt the next request).
                next_url = step.response.raw_location
                if next_url:
                    # Handle relative URLs
                    next_url = urljoin(current_url, next_url)
                    next_method = _redirect_method(step.response.status or 0, method)
                    headers = _redirect_headers(
                        headers,
                        keep_credentials=_is_same_origin_or_https_upgrade(current_url, next_url),
                        keep_body_headers=next_method == method,
                    )
                    if next_method != method:
                        content = None
                    method = next_method
                    current_url = next_url
                    redirect_count += 1
                else:
                    # No Location header despite 3xx status
                    break
            else:
                # Not a redirect, we're done
                break

        return steps

    def _analyze_single_request(
        self,
        url: str,
        step_number: int,
        *,
        method: HTTPMethod = HTTPMethod.GET,
        content: bytes | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> StepMetrics:
        """Analyze a single HTTP request.

        Args:
            url: URL to request.
            step_number: Step number in redirect chain (1-indexed).
            method: HTTP method to use.
            content: Optional request body as bytes.
            headers: Optional request headers for this step.

        Returns:
            StepMetrics with collected data. If request fails, error field
            will be populated, but step is still returned with partial data.

        Note:
            This method catches all exceptions and converts them to StepMetrics
            with error information, ensuring the analysis chain can continue.

        """
        policy = self._policy
        # Learn this hop's secrets before execution so failure messages can
        # be scrubbed even when the transport never completes.
        policy.learn_url(url)
        policy.learn_headers(headers)

        # The step carries the safe URL view only; the raw URL is retained
        # separately for transport and redirect resolution.
        safe_url = policy.redact_url(url) or url
        step = StepMetrics(url=safe_url, raw_url=url, step_number=step_number)

        # Populate request metadata. The executor still receives the raw
        # headers via RequestOptions; only the stored view is sanitized.
        step.request_method = method.value
        step.request_headers = policy.redact_headers(headers)
        step.request_body_bytes = len(content) if content else 0

        # Show the configured proxy even when the transport fails; the
        # outcome below overrides it with the effective proxy when known.
        if self._proxy is not None:
            step.proxied_via = policy.redact_url(str(getattr(self._proxy, "url", self._proxy)))

        try:
            # Create timing collector instance if factory provided
            timing_collector = self._timing_collector() if self._timing_collector else None

            options = RequestOptions(
                url=url,
                timeout=self.timeout,
                method=method,
                content=content,
                http2=self.http2,
                verify_ssl=self.verify_ssl,
                ca_bundle_path=self.ca_bundle_path,
                dns_resolver=self._dns_resolver,
                tls_inspector=self._tls_inspector,
                timing_collector=timing_collector,
                force_new_connection=True,
                headers=headers,
                proxy=self._proxy,
                noproxy=self._noproxy,
            )
            outcome: RequestOutcome = self._request.execute(options)

            # Populate step metrics
            step.timing = outcome.timing
            step.network = outcome.network

            # The executor reports the raw Location (needed for urljoin) and
            # default-sanitized headers. Re-apply the session policy so custom
            # sensitive names are masked as well, then split raw/safe Location.
            response = outcome.response
            raw_location = response.location
            if raw_location is not None:
                # The raw Location feeds urljoin (above) only; the stored
                # view and headers mapping carry the masked copy.
                safe_location = policy.redact_url(raw_location) or raw_location
                response.raw_location = raw_location
                response.location = safe_location
                for header_name in tuple(response.headers):
                    if header_name.lower() == "location":
                        response.headers[header_name] = safe_location
            response.headers = policy.redact_headers(response.headers)
            step.response = response

            if outcome.network.proxy_url:
                # Effective proxy may come from configuration or environment.
                step.network.proxy_url = policy.redact_url(outcome.network.proxy_url)
                step.proxied_via = step.network.proxy_url

        except HTTPClientError as e:
            # Request failed, but we have partial data. Scrub the message
            # (httpx may embed the target URL, timeout detail, or proxy).
            step.error = policy.redact_text(str(e))
            step.note = f"Step {step_number}: Request failed"

        except Exception as exc:  # noqa: BLE001
            # Unexpected error
            step.error = policy.redact_text(str(exc))
            step.note = f"Step {step_number}: Unexpected error"

        return step
