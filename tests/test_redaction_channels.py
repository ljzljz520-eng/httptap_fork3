"""Regression tests: secrets must never reach any output channel.

Covers full Rich mode, compact mode, metrics-only mode, and JSON export when
a request fails with an exception embedding the target URL, a timeout
message, or an authenticated proxy address. Also covers proxy userinfo
(#302) and Proxy-Authorization masking across the same channels.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from rich.console import Console

from httptap.analyzer import HTTPTapAnalyzer
from httptap.constants import PROXY_SOURCE_CLI
from httptap.exporter import JSONExporter
from httptap.http_client import HTTPClientError
from httptap.models import NetworkInfo, ResponseInfo, TimingMetrics
from httptap.redaction import RedactionPolicy
from httptap.render import OutputRenderer
from httptap.request_executor import RequestOptions, RequestOutcome

if TYPE_CHECKING:
    from pathlib import Path

    from httptap.models import StepMetrics

TARGET_SECRET = "targettokensecret77"  # noqa: S105 - intentional fake credential
PROXY_PASSWORD = "proxypwsecret88"  # noqa: S105 - intentional fake credential
PROXY_AUTH_VALUE = "Basic cHJveHlhdXRoc2VjcmV0OTk"

SECRET_BYTES = (
    TARGET_SECRET,
    PROXY_PASSWORD,
    PROXY_AUTH_VALUE,
)

RAW_TARGET_URL = f"https://api.example/v1?token={TARGET_SECRET}&page=4"
RAW_PROXY_URL = f"http://proxyuser:{PROXY_PASSWORD}@proxy.local:3128"


class FailingWithUrlExecutor:
    """Raises an HTTPClientError embedding URL, timeout detail, and proxy."""

    def execute(self, options: RequestOptions) -> RequestOutcome:
        del options
        # Realistic httpx-style failure: embedded target URL, proxy URL with
        # userinfo, plus a credential echoed back outside URL form.
        message = (
            f"Request timed out after 20s while requesting {RAW_TARGET_URL}; "
            f"connect through proxy {RAW_PROXY_URL} failed; credential rejected: {TARGET_SECRET}"
        )
        raise HTTPClientError(message)


class ProxiedOkExecutor:
    """Returns a successful outcome routed through an authenticated proxy."""

    def execute(self, options: RequestOptions) -> RequestOutcome:
        del options
        timing = TimingMetrics(total_ms=12.5)
        network = NetworkInfo(
            ip="203.0.113.9",
            ip_family="IPv4",
            http_version="HTTP/1.1",
            proxy_url=RAW_PROXY_URL,
            proxy_source=PROXY_SOURCE_CLI,
        )
        response = ResponseInfo(status=200)
        return RequestOutcome(timing=timing, network=network, response=response)


@pytest.fixture
def failing_steps() -> list[StepMetrics]:
    policy = RedactionPolicy()
    analyzer = HTTPTapAnalyzer(
        request_executor=FailingWithUrlExecutor(),
        proxy=RAW_PROXY_URL,
        redaction_policy=policy,
    )
    return analyzer.analyze_url(
        RAW_TARGET_URL,
        headers={"Proxy-Authorization": PROXY_AUTH_VALUE, "X-Trace": "trace-visible"},
    )


@pytest.fixture
def proxied_steps() -> list[StepMetrics]:
    policy = RedactionPolicy()
    analyzer = HTTPTapAnalyzer(
        request_executor=ProxiedOkExecutor(),
        proxy=RAW_PROXY_URL,
        redaction_policy=policy,
    )
    return analyzer.analyze_url(
        RAW_TARGET_URL,
        headers={"Proxy-Authorization": PROXY_AUTH_VALUE, "X-Trace": "trace-visible"},
    )


def _render(steps: list[StepMetrics], policy: RedactionPolicy, **mode: bool) -> str:
    console = Console(record=True, width=300, color_system=None)
    renderer = OutputRenderer(console=console, redaction_policy=policy, **mode)
    renderer.render_analysis(steps, RAW_TARGET_URL)
    return console.export_text()


@pytest.mark.parametrize(
    "mode",
    [{}, {"compact": True}, {"metrics_only": True}],
    ids=["full", "compact", "metrics-only"],
)
def test_failure_output_never_leaks_secrets(
    failing_steps: list[StepMetrics],
    mode: dict[str, bool],
) -> None:
    output = _render(failing_steps, RedactionPolicy(), **mode)

    for secret in SECRET_BYTES:
        assert secret not in output

    # Non-sensitive content remains visible and actionable.
    assert "timed out" in output
    assert "page=4" in output  # harmless query value preserved
    assert "proxy.local:3128" in output  # proxy host remains diagnosable
    if mode:  # compact and metrics-only modes mark failed steps explicitly
        assert "ERROR" in output


def test_failure_json_export_never_leaks_secrets(
    failing_steps: list[StepMetrics],
    tmp_path: Path,
) -> None:
    policy = RedactionPolicy()
    report = tmp_path / "report.json"
    exporter = JSONExporter(Console(record=True), redaction_policy=policy)

    exporter.export(failing_steps, RAW_TARGET_URL, str(report))

    raw = report.read_text()
    for secret in SECRET_BYTES:
        assert secret not in raw

    # Existing consumer structure stays intact.
    data = json.loads(raw)
    assert set(data) == {"initial_url", "total_steps", "steps", "summary"}
    assert data["initial_url"] == "https://api.example/v1?token=****&page=4"
    step = data["steps"][0]
    assert "raw_url" not in step
    assert "raw_location" not in step["response"]
    assert step["error"] is not None
    assert PROXY_PASSWORD not in step["error"]
    assert TARGET_SECRET not in step["error"]
    assert "****" in step["error"]
    assert step["request"]["headers"]["Proxy-Authorization"] != PROXY_AUTH_VALUE
    assert step["request"]["headers"]["X-Trace"] == "trace-visible"
    assert step["proxy"] == "http://proxyuser:****@proxy.local:3128"
    assert data["summary"]["errors"] == 1


@pytest.mark.parametrize(
    "mode",
    [{}, {"compact": True}, {"metrics_only": True}],
    ids=["full", "compact", "metrics-only"],
)
def test_proxy_success_output_never_leaks_credentials(
    proxied_steps: list[StepMetrics],
    mode: dict[str, bool],
) -> None:
    """#302: proxy userinfo and Proxy-Authorization stay masked everywhere."""
    output = _render(proxied_steps, RedactionPolicy(), **mode)

    assert PROXY_PASSWORD not in output
    assert PROXY_AUTH_VALUE not in output
    # Compact/metrics lines do not surface proxy info at all; full mode shows
    # it with credentials masked.
    if not mode:
        assert "proxyuser:****@proxy.local:3128" in output


def test_proxy_userinfo_masked_in_full_mode_panel(proxied_steps: list[StepMetrics]) -> None:
    output = _render(proxied_steps, RedactionPolicy())

    assert "Proxy: http://proxyuser:****@proxy.local:3128 (from arg --proxy)" in output


def test_proxy_credentials_masked_in_json(
    proxied_steps: list[StepMetrics],
    tmp_path: Path,
) -> None:
    policy = RedactionPolicy()
    report = tmp_path / "report.json"
    JSONExporter(Console(record=True), redaction_policy=policy).export(proxied_steps, RAW_TARGET_URL, str(report))

    raw = report.read_text()
    assert PROXY_PASSWORD not in raw
    assert PROXY_AUTH_VALUE not in raw

    data = json.loads(raw)
    step = data["steps"][0]
    assert step["network"]["proxy_url"] == "http://proxyuser:****@proxy.local:3128"
    assert step["proxy"] == "http://proxyuser:****@proxy.local:3128"
    assert step["request"]["headers"]["Proxy-Authorization"] != PROXY_AUTH_VALUE
    assert step["request"]["headers"]["X-Trace"] == "trace-visible"
