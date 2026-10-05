from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import pytest

from httptap.analyzer import HTTPTapAnalyzer
from httptap.constants import HTTPMethod
from httptap.http_client import HTTPClientError
from httptap.models import NetworkInfo, ResponseInfo, TimingMetrics
from httptap.redaction import RedactionPolicy
from httptap.request_executor import RequestOptions, RequestOutcome

if TYPE_CHECKING:
    from collections.abc import Mapping


class StubExecutor:
    def __init__(self, results: list[tuple[int, str | None]]) -> None:
        self.results = results
        self.calls: list[Mapping[str, str] | None] = []

    def execute(self, options: RequestOptions) -> RequestOutcome:
        if not self.results:
            msg = "no more results"
            raise HTTPClientError(msg)

        self.calls.append(options.headers)
        status, location = self.results.pop(0)

        timing = TimingMetrics(total_ms=100.0)
        network = NetworkInfo(ip="203.0.113.5", ip_family="IPv4")
        response = ResponseInfo(status=status, location=location)
        return RequestOutcome(timing=timing, network=network, response=response)


def test_analyze_url_without_redirect() -> None:
    executor = StubExecutor([(200, None)])
    analyzer = HTTPTapAnalyzer(request_executor=executor)

    steps = analyzer.analyze_url("https://example.test", headers={"X": "1"})

    assert len(steps) == 1
    assert steps[0].response.status == 200
    assert executor.calls == [{"X": "1"}]
    assert steps[0].proxied_via is None


def test_analyze_url_with_redirect_following() -> None:
    executor = StubExecutor([(301, "https://example.test/final"), (200, None)])
    analyzer = HTTPTapAnalyzer(follow_redirects=True, request_executor=executor)

    steps = analyzer.analyze_url("https://example.test")

    assert [step.response.status for step in steps] == [301, 200]
    assert steps[1].url == "https://example.test/final"


def test_analyze_url_records_error() -> None:
    executor = StubExecutor([])
    analyzer = HTTPTapAnalyzer(request_executor=executor)

    steps = analyzer.analyze_url("https://example.test")
    assert steps[0].has_error
    assert "no more results" in (steps[0].error or "")


def test_analyze_url_stops_on_error_when_following_redirects() -> None:
    """Test that analyzer stops following redirects when an error occurs."""
    executor = StubExecutor([(301, "https://example.test/redirect")])
    analyzer = HTTPTapAnalyzer(follow_redirects=True, request_executor=executor)

    steps = analyzer.analyze_url("https://example.test")

    # First step should be redirect, second should have error
    assert len(steps) == 2
    assert steps[0].response.status == 301
    assert steps[1].has_error
    # Should stop after error
    assert "no more results" in (steps[1].error or "")


def test_analyze_url_with_redirect_missing_location_header() -> None:
    """Test handling of 3xx response without Location header."""
    executor = StubExecutor([(302, None)])  # Redirect with no location
    analyzer = HTTPTapAnalyzer(follow_redirects=True, request_executor=executor)

    steps = analyzer.analyze_url("https://example.test")

    # Should stop after first step due to missing Location
    assert len(steps) == 1
    assert steps[0].response.status == 302
    assert steps[0].response.location is None


def test_analyze_url_with_redirect_blank_location_header() -> None:
    """Test handling of 3xx response with blank Location header."""
    executor = StubExecutor([(302, "")])  # Redirect with empty location string
    analyzer = HTTPTapAnalyzer(follow_redirects=True, request_executor=executor)

    steps = analyzer.analyze_url("https://example.test/initial")

    # Redirect should not be followed when Location header is blank
    assert len(steps) == 1
    assert steps[0].response.status == 302
    assert steps[0].response.location == ""


def test_analyze_url_respects_max_redirects() -> None:
    """Test that analyzer respects max_redirects limit."""
    # Create infinite redirect chain
    executor = StubExecutor(
        [(301, "https://example.test/1") for _ in range(20)],  # More than max
    )
    analyzer = HTTPTapAnalyzer(
        follow_redirects=True,
        max_redirects=5,
        request_executor=executor,
    )

    steps = analyzer.analyze_url("https://example.test")

    # Should stop at max_redirects + 1 (initial request + max redirects)
    assert len(steps) == 6  # Initial + 5 redirects
    assert all(step.response.status == 301 for step in steps)


def test_analyze_url_passes_verify_flag_when_supported() -> None:
    class VerifyAwareExecutor:
        def __init__(self) -> None:
            self.flags: list[bool] = []
            self.proxies: list[object | None] = []

        def execute(self, options: RequestOptions) -> RequestOutcome:
            self.flags.append(options.verify_ssl)
            self.proxies.append(options.proxy)

            timing = TimingMetrics(total_ms=10.0)
            network = NetworkInfo(ip="198.51.100.1", ip_family="IPv4")
            response = ResponseInfo(status=200)
            return RequestOutcome(timing=timing, network=network, response=response)

    executor = VerifyAwareExecutor()
    analyzer = HTTPTapAnalyzer(
        request_executor=executor,
        verify_ssl=False,
        proxy="http://proxy:8080",
    )

    steps = analyzer.analyze_url("https://example.test")

    assert len(steps) == 1
    assert steps[0].response.status == 200
    assert executor.flags == [False]
    assert executor.proxies == ["http://proxy:8080"]
    assert steps[0].proxied_via == "http://proxy:8080"
    assert executor.proxies == ["http://proxy:8080"]


def test_analyze_url_accepts_object_executor() -> None:
    class ObjectExecutor:
        def __init__(self) -> None:
            self.calls: list[RequestOptions] = []

        def execute(self, options: RequestOptions) -> RequestOutcome:
            self.calls.append(options)
            timing = TimingMetrics(total_ms=8.0)
            network = NetworkInfo(ip="198.51.100.2", ip_family="IPv4")
            response = ResponseInfo(status=204)
            return RequestOutcome(timing=timing, network=network, response=response)

    executor = ObjectExecutor()
    analyzer = HTTPTapAnalyzer(request_executor=executor)

    steps = analyzer.analyze_url("https://example.test")

    assert len(steps) == 1
    assert steps[0].response.status == 204
    assert executor.calls
    assert executor.calls[0].verify_ssl is True
    assert executor.calls[0].proxy is None


def test_analyze_url_handles_unexpected_exception() -> None:
    """Test handling of unexpected exceptions during request."""

    class FailingExecutor:
        def execute(self, _options: RequestOptions) -> RequestOutcome:
            msg = "Unexpected failure"
            raise RuntimeError(msg)

    analyzer = HTTPTapAnalyzer(request_executor=FailingExecutor())

    steps = analyzer.analyze_url("https://example.test")

    assert len(steps) == 1
    assert steps[0].has_error
    assert "Unexpected failure" in (steps[0].error or "")
    assert "Unexpected error" in (steps[0].note or "")


def test_analyze_url_with_post_method() -> None:
    """Test POST request with method parameter."""
    from httptap.constants import HTTPMethod

    executor = StubExecutor([(200, None)])
    analyzer = HTTPTapAnalyzer(request_executor=executor)

    steps = analyzer.analyze_url(
        "https://httpbin.test/post",
        method=HTTPMethod.POST,
        content=b'{"key": "value"}',
    )

    assert len(steps) == 1
    assert steps[0].request_method == "POST"
    assert steps[0].request_body_bytes == 16
    assert steps[0].response.status == 200


def test_analyze_url_with_put_method() -> None:
    """Test PUT request with method parameter."""
    from httptap.constants import HTTPMethod

    executor = StubExecutor([(200, None)])
    analyzer = HTTPTapAnalyzer(request_executor=executor)

    steps = analyzer.analyze_url(
        "https://httpbin.test/put",
        method=HTTPMethod.PUT,
        content=b'{"status": "updated"}',
    )

    assert len(steps) == 1
    assert steps[0].request_method == "PUT"
    assert steps[0].request_body_bytes == 21


def test_analyze_url_with_patch_method() -> None:
    """Test PATCH request with method parameter."""
    from httptap.constants import HTTPMethod

    executor = StubExecutor([(200, None)])
    analyzer = HTTPTapAnalyzer(request_executor=executor)

    steps = analyzer.analyze_url(
        "https://httpbin.test/patch",
        method=HTTPMethod.PATCH,
        content=b'{"field": "value"}',
    )

    assert len(steps) == 1
    assert steps[0].request_method == "PATCH"
    assert steps[0].request_body_bytes == 18


def test_analyze_url_with_delete_method() -> None:
    """Test DELETE request with method parameter."""
    from httptap.constants import HTTPMethod

    executor = StubExecutor([(204, None)])
    analyzer = HTTPTapAnalyzer(request_executor=executor)

    steps = analyzer.analyze_url(
        "https://httpbin.test/delete",
        method=HTTPMethod.DELETE,
    )

    assert len(steps) == 1
    assert steps[0].request_method == "DELETE"
    assert steps[0].request_body_bytes == 0


def test_analyze_url_with_head_method() -> None:
    """Test HEAD request with method parameter."""
    from httptap.constants import HTTPMethod

    executor = StubExecutor([(200, None)])
    analyzer = HTTPTapAnalyzer(request_executor=executor)

    steps = analyzer.analyze_url(
        "https://httpbin.test/get",
        method=HTTPMethod.HEAD,
    )

    assert len(steps) == 1
    assert steps[0].request_method == "HEAD"


def test_analyze_url_with_options_method() -> None:
    """Test OPTIONS request with method parameter."""
    from httptap.constants import HTTPMethod

    executor = StubExecutor([(200, None)])
    analyzer = HTTPTapAnalyzer(request_executor=executor)

    steps = analyzer.analyze_url(
        "https://httpbin.test/",
        method=HTTPMethod.OPTIONS,
    )

    assert len(steps) == 1
    assert steps[0].request_method == "OPTIONS"


def test_analyze_url_sanitizes_request_headers() -> None:
    """Test that request headers are sanitized in step metrics."""
    from httptap.constants import HTTPMethod

    executor = StubExecutor([(200, None)])
    analyzer = HTTPTapAnalyzer(request_executor=executor)

    steps = analyzer.analyze_url(
        "https://httpbin.test/post",
        method=HTTPMethod.POST,
        headers={
            "Authorization": "Bearer secret-token-12345",
            "Content-Type": "application/json",
        },
    )

    assert len(steps) == 1
    assert "Authorization" in steps[0].request_headers
    assert "secret" not in steps[0].request_headers["Authorization"]
    assert "****" in steps[0].request_headers["Authorization"]
    assert steps[0].request_headers["Content-Type"] == "application/json"


def test_analyze_url_with_get_method_default() -> None:
    """Test that GET is the default method when not specified."""
    executor = StubExecutor([(200, None)])
    analyzer = HTTPTapAnalyzer(request_executor=executor)

    steps = analyzer.analyze_url("https://httpbin.test/get")

    assert len(steps) == 1
    assert steps[0].request_method == "GET"
    assert steps[0].request_body_bytes == 0
    assert steps[0].request_headers == {}


def test_analyze_url_with_socks5h_proxy() -> None:
    """Test that socks5h proxy (remote DNS) is properly passed through analyzer."""

    class ProxyAwareExecutor:
        def __init__(self) -> None:
            self.proxies: list[object | None] = []

        def execute(self, options: RequestOptions) -> RequestOutcome:
            self.proxies.append(options.proxy)
            timing = TimingMetrics(total_ms=10.0)
            network = NetworkInfo(ip="198.51.100.1", ip_family="IPv4")
            response = ResponseInfo(status=200)
            return RequestOutcome(timing=timing, network=network, response=response)

    executor = ProxyAwareExecutor()
    analyzer = HTTPTapAnalyzer(
        request_executor=executor,
        proxy="socks5h://gateway:1080",
    )

    steps = analyzer.analyze_url("https://example.test")

    assert len(steps) == 1
    assert steps[0].response.status == 200
    assert executor.proxies == ["socks5h://gateway:1080"]
    assert steps[0].proxied_via == "socks5h://gateway:1080"


def test_analyze_url_with_http_proxy() -> None:
    """Test that HTTP proxy is properly passed through analyzer."""

    class ProxyAwareExecutor:
        def __init__(self) -> None:
            self.proxies: list[object | None] = []

        def execute(self, options: RequestOptions) -> RequestOutcome:
            self.proxies.append(options.proxy)
            timing = TimingMetrics(total_ms=10.0)
            network = NetworkInfo(ip="198.51.100.2", ip_family="IPv4")
            response = ResponseInfo(status=200)
            return RequestOutcome(timing=timing, network=network, response=response)

    executor = ProxyAwareExecutor()
    analyzer = HTTPTapAnalyzer(
        request_executor=executor,
        proxy="http://proxy.example.com:8080",
    )

    steps = analyzer.analyze_url("https://example.test")

    assert len(steps) == 1
    assert steps[0].response.status == 200
    assert executor.proxies == ["http://proxy.example.com:8080"]
    assert steps[0].proxied_via == "http://proxy.example.com:8080"


def test_analyze_url_with_https_proxy() -> None:
    """Test that HTTPS proxy is properly passed through analyzer."""

    class ProxyAwareExecutor:
        def __init__(self) -> None:
            self.proxies: list[object | None] = []

        def execute(self, options: RequestOptions) -> RequestOutcome:
            self.proxies.append(options.proxy)
            timing = TimingMetrics(total_ms=10.0)
            network = NetworkInfo(ip="198.51.100.3", ip_family="IPv4")
            response = ResponseInfo(status=200)
            return RequestOutcome(timing=timing, network=network, response=response)

    executor = ProxyAwareExecutor()
    analyzer = HTTPTapAnalyzer(
        request_executor=executor,
        proxy="https://secure-proxy.example.com:8443",
    )

    steps = analyzer.analyze_url("https://example.test")

    assert len(steps) == 1
    assert steps[0].response.status == 200
    assert executor.proxies == ["https://secure-proxy.example.com:8443"]
    assert steps[0].proxied_via == "https://secure-proxy.example.com:8443"


def test_analyze_url_with_socks5_proxy() -> None:
    """Test that socks5 proxy (local DNS) is properly passed through analyzer."""

    class ProxyAwareExecutor:
        def __init__(self) -> None:
            self.proxies: list[object | None] = []

        def execute(self, options: RequestOptions) -> RequestOutcome:
            self.proxies.append(options.proxy)
            timing = TimingMetrics(total_ms=10.0)
            network = NetworkInfo(ip="198.51.100.4", ip_family="IPv4")
            response = ResponseInfo(status=200)
            return RequestOutcome(timing=timing, network=network, response=response)

    executor = ProxyAwareExecutor()
    analyzer = HTTPTapAnalyzer(
        request_executor=executor,
        proxy="socks5://gateway:1080",
    )

    steps = analyzer.analyze_url("https://example.test")

    assert len(steps) == 1
    assert steps[0].response.status == 200
    assert executor.proxies == ["socks5://gateway:1080"]
    assert steps[0].proxied_via == "socks5://gateway:1080"


def test_analyze_url_with_proxy_and_redirects() -> None:
    """Test that proxy is used for all requests in redirect chain."""

    class ProxyAwareExecutor:
        def __init__(self) -> None:
            self.proxies: list[object | None] = []
            self.call_count = 0

        def execute(self, options: RequestOptions) -> RequestOutcome:
            self.proxies.append(options.proxy)
            self.call_count += 1

            timing = TimingMetrics(total_ms=10.0)
            network = NetworkInfo(ip="198.51.100.5", ip_family="IPv4")

            if self.call_count == 1:
                response = ResponseInfo(status=301, location="https://example.test/final")
            else:
                response = ResponseInfo(status=200)

            return RequestOutcome(timing=timing, network=network, response=response)

    executor = ProxyAwareExecutor()
    analyzer = HTTPTapAnalyzer(
        request_executor=executor,
        proxy="socks5h://gateway:1080",
        follow_redirects=True,
    )

    steps = analyzer.analyze_url("https://example.test/initial")

    assert len(steps) == 2
    assert steps[0].response.status == 301
    assert steps[1].response.status == 200
    # Verify proxy was used for both requests
    assert executor.proxies == ["socks5h://gateway:1080", "socks5h://gateway:1080"]
    assert steps[0].proxied_via == "socks5h://gateway:1080"
    assert steps[1].proxied_via == "socks5h://gateway:1080"


def test_analyze_url_proxy_with_authentication() -> None:
    """Test that proxy with authentication credentials is properly handled."""

    class ProxyAwareExecutor:
        def __init__(self) -> None:
            self.proxies: list[object | None] = []

        def execute(self, options: RequestOptions) -> RequestOutcome:
            self.proxies.append(options.proxy)
            timing = TimingMetrics(total_ms=10.0)
            network = NetworkInfo(ip="198.51.100.6", ip_family="IPv4")
            response = ResponseInfo(status=200)
            return RequestOutcome(timing=timing, network=network, response=response)

    executor = ProxyAwareExecutor()
    analyzer = HTTPTapAnalyzer(
        request_executor=executor,
        proxy="socks5h://user:password@gateway:1080",
    )

    steps = analyzer.analyze_url("https://example.test")

    assert len(steps) == 1
    assert steps[0].response.status == 200
    assert executor.proxies == ["socks5h://user:password@gateway:1080"]
    assert steps[0].proxied_via == "socks5h://user:****@gateway:1080"


class RecordingExecutor:
    def __init__(self, results: list[tuple[int, str | None]]) -> None:
        self.results = results
        self.calls: list[RequestOptions] = []

    def execute(self, options: RequestOptions) -> RequestOutcome:
        self.calls.append(options)
        status, location = self.results.pop(0)
        timing = TimingMetrics(total_ms=10.0)
        network = NetworkInfo(ip="203.0.113.5", ip_family="IPv4")
        response = ResponseInfo(status=status, location=location)
        return RequestOutcome(timing=timing, network=network, response=response)


CREDENTIAL_HEADERS = {
    "Authorization": "Bearer secret",
    "Cookie": "session=1",
    "Proxy-Authorization": "Basic cHJveHk6cHc=",
    "X-Trace": "abc",
}


@pytest.mark.parametrize(
    ("initial_url", "location"),
    [
        ("https://example.test/start", "https://attacker.test/landing"),
        ("https://example.test/start", "http://example.test/landing"),
        ("https://example.test/start", "https://example.test:8443/landing"),
        ("http://example.test:8080/start", "https://example.test/landing"),
    ],
)
def test_analyze_url_drops_credentials_on_cross_origin_redirect(initial_url: str, location: str) -> None:
    executor = RecordingExecutor([(302, location), (200, None)])
    analyzer = HTTPTapAnalyzer(follow_redirects=True, request_executor=executor)

    steps = analyzer.analyze_url(initial_url, headers=CREDENTIAL_HEADERS)

    assert executor.calls[0].headers == CREDENTIAL_HEADERS
    assert executor.calls[1].headers == {"X-Trace": "abc"}
    assert steps[1].request_headers == {"X-Trace": "abc"}


@pytest.mark.parametrize(
    ("initial_url", "location"),
    [
        ("https://example.test/start", "/landing"),
        ("https://example.test/start", "https://EXAMPLE.test:443/landing"),
        ("http://example.test/start", "https://example.test/landing"),
    ],
)
def test_analyze_url_keeps_credentials_on_same_origin_or_https_upgrade(initial_url: str, location: str) -> None:
    executor = RecordingExecutor([(301, location), (200, None)])
    analyzer = HTTPTapAnalyzer(follow_redirects=True, request_executor=executor)

    analyzer.analyze_url(initial_url, headers=CREDENTIAL_HEADERS)

    assert executor.calls[1].headers == CREDENTIAL_HEADERS


def test_analyze_url_drops_credentials_for_rest_of_chain_after_cross_origin_hop() -> None:
    executor = RecordingExecutor(
        [(302, "https://other.test/a"), (302, "https://example.test/b"), (200, None)],
    )
    analyzer = HTTPTapAnalyzer(follow_redirects=True, request_executor=executor)

    analyzer.analyze_url("https://example.test/start", headers=CREDENTIAL_HEADERS)

    assert executor.calls[2].headers == {"X-Trace": "abc"}


@pytest.mark.parametrize(
    ("status", "method", "expected_method"),
    [
        (303, HTTPMethod.POST, HTTPMethod.GET),
        (303, HTTPMethod.PUT, HTTPMethod.GET),
        (301, HTTPMethod.POST, HTTPMethod.GET),
        (302, HTTPMethod.POST, HTTPMethod.GET),
    ],
)
def test_analyze_url_switches_to_get_without_body(
    status: int,
    method: HTTPMethod,
    expected_method: HTTPMethod,
) -> None:
    executor = RecordingExecutor([(status, "/landing"), (200, None)])
    analyzer = HTTPTapAnalyzer(follow_redirects=True, request_executor=executor)
    headers = {"Content-Type": "application/json", "X-Trace": "abc"}

    steps = analyzer.analyze_url("https://example.test/form", method=method, content=b'{"a":1}', headers=headers)

    assert executor.calls[1].method == expected_method
    assert executor.calls[1].content is None
    assert executor.calls[1].headers == {"X-Trace": "abc"}
    assert steps[1].request_method == expected_method.value
    assert steps[1].request_body_bytes == 0


@pytest.mark.parametrize(
    ("status", "method"),
    [
        (307, HTTPMethod.POST),
        (308, HTTPMethod.POST),
        (302, HTTPMethod.PUT),
        (303, HTTPMethod.HEAD),
    ],
)
def test_analyze_url_preserves_method_and_body(status: int, method: HTTPMethod) -> None:
    executor = RecordingExecutor([(status, "/landing"), (200, None)])
    analyzer = HTTPTapAnalyzer(follow_redirects=True, request_executor=executor)
    headers = {"Content-Type": "application/json"}

    analyzer.analyze_url("https://example.test/form", method=method, content=b'{"a":1}', headers=headers)

    assert executor.calls[1].method == method
    assert executor.calls[1].content == b'{"a":1}'
    assert executor.calls[1].headers == headers


def test_analyze_url_redirect_without_headers() -> None:
    executor = RecordingExecutor([(302, "https://other.test/"), (200, None)])
    analyzer = HTTPTapAnalyzer(follow_redirects=True, request_executor=executor)

    analyzer.analyze_url("https://example.test/")

    assert executor.calls[1].headers is None


class SecretErrorExecutor:
    """Executor whose failure message may embed URLs, timeouts, or proxies."""

    def __init__(self, message: str) -> None:
        self.message = message

    def execute(self, options: RequestOptions) -> RequestOutcome:
        raise HTTPClientError(self.message.format(url=options.url))


_RUNTIME_FAILURE_TEXT = "boom tokensecret99"


class SecretRuntimeErrorExecutor:
    def execute(self, options: RequestOptions) -> RequestOutcome:
        del options
        raise RuntimeError(_RUNTIME_FAILURE_TEXT)


class HeaderCapturingExecutor:
    """Returns a response carrying custom-named sensitive headers."""

    def __init__(self, request_headers: dict[str, str] | None = None) -> None:
        self.calls: list[Mapping[str, str] | None] = []
        self._request_headers = request_headers or {}

    def execute(self, options: RequestOptions) -> RequestOutcome:
        self.calls.append(options.headers)
        timing = TimingMetrics(total_ms=10.0)
        network = NetworkInfo(ip="203.0.113.9", ip_family="IPv4")
        response = ResponseInfo(status=200, headers=dict(self._request_headers))
        return RequestOutcome(timing=timing, network=network, response=response)


def test_analyze_url_separates_raw_and_safe_url() -> None:
    """The model carries a raw transport URL and a display-safe view."""
    policy = RedactionPolicy()
    executor = StubExecutor([(200, None)])
    analyzer = HTTPTapAnalyzer(request_executor=executor, redaction_policy=policy)
    raw_url = "https://api.example/v1?token=tokstartsecret&page=1"

    steps = analyzer.analyze_url(raw_url)

    assert steps[0].raw_url == raw_url
    assert steps[0].url == "https://api.example/v1?token=****&page=1"
    assert "tokstartsecret" not in steps[0].url
    assert "page=1" in steps[0].url


def test_analyze_url_to_dict_excludes_raw_fields_and_secrets() -> None:
    import json

    policy = RedactionPolicy()
    executor = RecordingExecutor([(302, "/next?signature=sigrelsecret"), (200, None)])
    analyzer = HTTPTapAnalyzer(follow_redirects=True, request_executor=executor, redaction_policy=policy)

    steps = analyzer.analyze_url("https://example.test/start?token=dictsecret")

    serialized = json.dumps([step.to_dict() for step in steps])
    assert "raw_url" not in serialized
    assert "raw_location" not in serialized
    assert "dictsecret" not in serialized
    assert "sigrelsecret" not in serialized


def test_analyze_url_uses_raw_relative_location_for_join_but_stores_safe_view() -> None:
    policy = RedactionPolicy()
    executor = RecordingExecutor(
        [(302, "/next?signature=sigrelsecret&page=9"), (200, None)],
    )
    analyzer = HTTPTapAnalyzer(follow_redirects=True, request_executor=executor, redaction_policy=policy)

    steps = analyzer.analyze_url("https://example.test/start?token=tokstartsecret&page=1")

    # Transport received the full original values, joined from the raw Location.
    assert executor.calls[0].url == "https://example.test/start?token=tokstartsecret&page=1"
    assert executor.calls[1].url == "https://example.test/next?signature=sigrelsecret&page=9"

    first = steps[0]
    assert first.response.raw_location == "/next?signature=sigrelsecret&page=9"
    assert first.response.location == "/next?signature=****&page=9"
    assert "sigrelsecret" not in first.response.location

    second = steps[1]
    assert second.raw_url == "https://example.test/next?signature=sigrelsecret&page=9"
    assert second.url == "https://example.test/next?signature=****&page=9"
    assert "sigrelsecret" not in second.url
    assert "page=9" in second.url


def test_analyze_url_uses_raw_absolute_location_for_next_hop() -> None:
    policy = RedactionPolicy()
    signed_location = "https://other.test/landing?X-Amz-Signature=sigabssecret&X-Amz-Algorithm=SHA"
    executor = RecordingExecutor([(302, signed_location), (200, None)])
    analyzer = HTTPTapAnalyzer(follow_redirects=True, request_executor=executor, redaction_policy=policy)

    steps = analyzer.analyze_url("https://example.test/start")

    assert executor.calls[1].url == signed_location
    assert steps[0].response.location == ("https://other.test/landing?X-Amz-Signature=****&X-Amz-Algorithm=SHA")
    assert steps[0].response.raw_location == signed_location
    assert "sigabssecret" not in steps[1].url
    assert steps[1].raw_url == signed_location


def test_analyze_url_sanitizes_http_client_error_with_target_url() -> None:
    url_with_credential = "https://api.example/v1?token=errtokensecret&page=3"
    executor = SecretErrorExecutor("Request failed: {url} timed out after 20s")
    analyzer = HTTPTapAnalyzer(request_executor=executor, redaction_policy=RedactionPolicy())

    steps = analyzer.analyze_url(url_with_credential)

    assert steps[0].has_error
    assert "errtokensecret" not in (steps[0].error or "")
    assert "token=****" in (steps[0].error or "")
    assert "page=3" in (steps[0].error or "")
    assert "timed out" in (steps[0].error or "")


def test_analyze_url_sanitizes_error_with_proxy_address() -> None:
    message = "tunnel via http://proxyuser:proxypwsecret@proxy.local:3128 failed"
    executor = SecretErrorExecutor(message)
    analyzer = HTTPTapAnalyzer(
        request_executor=executor,
        proxy="http://proxyuser:proxypwsecret@proxy.local:3128",
        redaction_policy=RedactionPolicy(),
    )

    steps = analyzer.analyze_url("https://api.example/v1")

    assert "proxypwsecret" not in (steps[0].error or "")
    assert "http://proxyuser:****@proxy.local:3128" in (steps[0].error or "")


def test_analyze_url_sanitizes_learned_header_secret_in_error() -> None:
    executor = SecretErrorExecutor("server rejected credential 'Bearer errheadersecret99'")
    analyzer = HTTPTapAnalyzer(request_executor=executor, redaction_policy=RedactionPolicy())

    steps = analyzer.analyze_url(
        "https://api.example/v1",
        headers={"Authorization": "Bearer errheadersecret99"},
    )

    assert "errheadersecret99" not in (steps[0].error or "")


def test_analyze_url_sanitizes_unexpected_exception() -> None:
    analyzer = HTTPTapAnalyzer(
        request_executor=SecretRuntimeErrorExecutor(),
        redaction_policy=RedactionPolicy(),
    )

    steps = analyzer.analyze_url("https://api.example/v1?token=tokensecret99")

    assert "tokensecret99" not in (steps[0].error or "")


def test_analyze_url_masks_custom_request_header_for_storage_but_sends_raw() -> None:
    policy = RedactionPolicy(header_names=["x-tenant-secret"])
    executor = HeaderCapturingExecutor()
    analyzer = HTTPTapAnalyzer(request_executor=executor, redaction_policy=policy)
    headers = {"X-Tenant-Secret": "tennantrequestvalue", "X-Trace": "visible-value"}

    steps = analyzer.analyze_url("https://api.example/v1", headers=headers)

    # The executor (transport) must still receive the original value.
    assert executor.calls == [headers]
    assert steps[0].request_headers["X-Tenant-Secret"] != "tennantrequestvalue"
    assert "****" in steps[0].request_headers["X-Tenant-Secret"]
    assert steps[0].request_headers["X-Trace"] == "visible-value"


def test_analyze_url_masks_custom_response_header() -> None:
    policy = RedactionPolicy(header_names=["x-tenant-secret"])
    executor = HeaderCapturingExecutor(
        {"X-Tenant-Secret": "tennantresponsevalue", "X-Trace": "keep-me"},
    )
    analyzer = HTTPTapAnalyzer(request_executor=executor, redaction_policy=policy)

    steps = analyzer.analyze_url("https://api.example/v1")

    assert steps[0].response.headers["X-Tenant-Secret"] != "tennantresponsevalue"
    assert "****" in steps[0].response.headers["X-Tenant-Secret"]
    assert steps[0].response.headers["X-Trace"] == "keep-me"


def test_analyze_url_proxy_credentials_masked_via_policy() -> None:
    """#302 behavior survives the policy migration for both display fields."""
    policy = RedactionPolicy()

    class ProxyExecutor:
        def execute(self, options: RequestOptions) -> RequestOutcome:
            del options
            timing = TimingMetrics(total_ms=10.0)
            network = NetworkInfo(
                ip="203.0.113.6",
                ip_family="IPv4",
                proxy_url="http://user:password@proxy:3128",
            )
            return RequestOutcome(timing=timing, network=network, response=ResponseInfo(status=200))

    analyzer = HTTPTapAnalyzer(
        request_executor=ProxyExecutor(),
        proxy="http://user:password@proxy:3128",
        redaction_policy=policy,
    )

    steps = analyzer.analyze_url("https://api.example/v1")

    assert steps[0].proxied_via == "http://user:****@proxy:3128"
    assert steps[0].network.proxy_url == "http://user:****@proxy:3128"
    assert "password" not in steps[0].to_dict()["network"]["proxy_url"]
