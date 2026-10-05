"""Live-server integration tests for the session-bound redaction policy.

These tests start a real loopback HTTP server so that transport fidelity
(the server must receive untouched URLs and headers) is verified together
with output safety (every display/serialization channel must carry only
masked views).
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, ClassVar

import pytest
from rich.console import Console

from httptap.analyzer import HTTPTapAnalyzer
from httptap.cli import EXIT_SUCCESS
from httptap.constants import HTTPMethod
from httptap.exporter import JSONExporter
from httptap.redaction import RedactionPolicy
from httptap.render import OutputRenderer

if TYPE_CHECKING:
    from pathlib import Path

    from httptap.models import StepMetrics

TOKEN_A = "tokAAAsecret11"  # noqa: S105 - intentional fake credential
TOKEN_B = "tokBBsecret22"  # noqa: S105 - intentional fake credential
API_KEY_ENCODED = "ak%40secret"  # percent-encoded form of ak@secret
API_KEY_DECODED = "ak@secret"
SIG_REL = "sigrelsecret33"
SIG_ABS = "sigabssecret44"
TENANT_VAL_1 = "tenantvalsecret55"
TENANT_VAL_2 = "tenantvalsecret66"
TENANT_HEADER_SECRET = "tenantheadersecret77"  # noqa: S105 - intentional fake credential

ALL_SECRETS = (
    TOKEN_A,
    TOKEN_B,
    API_KEY_ENCODED,
    API_KEY_DECODED,
    SIG_REL,
    SIG_ABS,
    TENANT_VAL_1,
    TENANT_VAL_2,
    TENANT_HEADER_SECRET,
)


class RedirectRecordingHandler(BaseHTTPRequestHandler):
    """Records raw request targets and serves a signed redirect chain."""

    received_paths: ClassVar[list[str]] = []
    received_headers: ClassVar[list[str]] = []

    def do_GET(self) -> None:
        type(self).received_paths.append(self.path)
        if self.path.startswith("/entry"):
            self._redirect(f"/rel?signature={SIG_REL}&page=9")
        elif self.path.startswith("/rel"):
            port = self.server.server_port
            self._redirect(f"http://127.0.0.1:{port}/abs?X-Amz-Signature={SIG_ABS}&page=5")
        elif self.path.startswith("/abs"):
            self._record_tenant_headers()
            self._respond(200, b"ok")
        else:
            self._respond(404, b"missing")

    def _record_tenant_headers(self) -> None:
        value = self.headers.get("X-Tenant-Secret")
        if value is not None:
            type(self).received_headers.append(value)

    def _redirect(self, location: str) -> None:
        self.send_response(302)
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _respond(self, status: int, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        del format, args


@pytest.fixture
def redirect_server() -> ThreadingHTTPServer:
    RedirectRecordingHandler.received_paths = []
    RedirectRecordingHandler.received_headers = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), RedirectRecordingHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _base_url(server: ThreadingHTTPServer) -> str:
    return f"http://127.0.0.1:{server.server_port}"


def _entry_url(base: str) -> str:
    return f"{base}/entry?token={TOKEN_A}&token={TOKEN_B}&api_key={API_KEY_ENCODED}&page=7"


def _render(steps: list[StepMetrics], initial_url: str, policy: RedactionPolicy, **mode: bool) -> str:
    console = Console(record=True, width=400, color_system=None)
    renderer = OutputRenderer(console=console, redaction_policy=policy, **mode)
    renderer.render_analysis(steps, initial_url)
    return console.export_text()


def test_transport_receives_raw_values_but_views_are_masked(redirect_server: ThreadingHTTPServer) -> None:
    base = _base_url(redirect_server)
    initial_url = _entry_url(base)
    policy = RedactionPolicy()
    analyzer = HTTPTapAnalyzer(
        follow_redirects=True,
        http2=False,
        redaction_policy=policy,
    )

    steps = analyzer.analyze_url(initial_url)

    # The server must receive the untouched original request targets...
    assert RedirectRecordingHandler.received_paths == [
        f"/entry?token={TOKEN_A}&token={TOKEN_B}&api_key={API_KEY_ENCODED}&page=7",
        f"/rel?signature={SIG_REL}&page=9",
        f"/abs?X-Amz-Signature={SIG_ABS}&page=5",
    ]

    # ...while every stored view is masked and raw fields stay private.
    assert [step.response.status for step in steps] == [302, 302, 200]
    assert steps[0].raw_url == initial_url
    assert steps[0].url == f"{base}/entry?token=****&token=****&api_key=****&page=7"
    assert steps[0].response.raw_location == f"/rel?signature={SIG_REL}&page=9"
    assert steps[0].response.location == "/rel?signature=****&page=9"

    assert steps[1].raw_url == f"{base}/rel?signature={SIG_REL}&page=9"
    assert steps[1].url == f"{base}/rel?signature=****&page=9"
    assert steps[1].response.raw_location == f"{base}/abs?X-Amz-Signature={SIG_ABS}&page=5"
    assert steps[1].response.location == f"{base}/abs?X-Amz-Signature=****&page=5"

    assert steps[2].raw_url == f"{base}/abs?X-Amz-Signature={SIG_ABS}&page=5"
    assert steps[2].url == f"{base}/abs?X-Amz-Signature=****&page=5"


@pytest.mark.parametrize(
    "mode",
    [{}, {"compact": True}, {"metrics_only": True}],
    ids=["full", "compact", "metrics-only"],
)
def test_all_terminal_modes_omit_secret_bytes(
    redirect_server: ThreadingHTTPServer,
    mode: dict[str, bool],
) -> None:
    base = _base_url(redirect_server)
    initial_url = _entry_url(base)
    policy = RedactionPolicy()
    analyzer = HTTPTapAnalyzer(follow_redirects=True, http2=False, redaction_policy=policy)
    steps = analyzer.analyze_url(initial_url)

    output = _render(steps, initial_url, policy, **mode)

    for secret in ALL_SECRETS:
        assert secret not in output

    # Non-sensitive query values remain visible for diagnosis. The
    # metrics-only format is deliberately URL-free (as before redaction).
    if not mode.get("metrics_only"):
        assert "page=7" in output
        assert "page=9" in output
        assert "page=5" in output
        assert "****" in output


def test_json_export_omits_secret_bytes_and_keeps_consumer_shape(
    redirect_server: ThreadingHTTPServer,
    tmp_path: Path,
) -> None:
    base = _base_url(redirect_server)
    initial_url = _entry_url(base)
    policy = RedactionPolicy()
    analyzer = HTTPTapAnalyzer(follow_redirects=True, http2=False, redaction_policy=policy)
    steps = analyzer.analyze_url(initial_url)

    report = tmp_path / "report.json"
    JSONExporter(Console(record=True), redaction_policy=policy).export(steps, initial_url, str(report))
    raw = report.read_text()

    for secret in ALL_SECRETS:
        assert secret not in raw
    assert "page=7" in raw

    # The document stays parseable for existing JSON consumers.
    data = json.loads(raw)
    assert set(data) == {"initial_url", "total_steps", "steps", "summary"}
    assert data["total_steps"] == 3
    assert data["initial_url"] == f"{base}/entry?token=****&token=****&api_key=****&page=7"
    assert data["steps"][0]["response"]["location"] == "/rel?signature=****&page=9"
    assert data["steps"][1]["response"]["location"] == f"{base}/abs?X-Amz-Signature=****&page=5"
    assert data["summary"]["final_url"] == f"{base}/abs?X-Amz-Signature=****&page=5"
    for step in data["steps"]:
        assert "raw_url" not in step
        assert "raw_location" not in step["response"]
        # The Location entry riding in the headers map must be masked too.
        for name, value in step["response"]["headers"].items():
            if name.lower() == "location":
                assert "****" in value
                assert SIG_REL not in value
                assert SIG_ABS not in value


def test_progress_text_uses_masked_url(redirect_server: ThreadingHTTPServer) -> None:
    """The spinner description built by the CLI must not contain secrets."""
    from rich.markup import escape

    base = _base_url(redirect_server)
    initial_url = _entry_url(base)
    policy = RedactionPolicy()

    progress_url = escape(policy.redact_url(initial_url) or initial_url)

    assert TOKEN_A not in progress_url
    assert TOKEN_B not in progress_url
    assert API_KEY_ENCODED not in progress_url
    assert "token=****" in progress_url
    assert "api_key=****" in progress_url
    assert "page=7" in progress_url


def test_custom_query_keys_and_headers_case_variants_and_duplicates(
    redirect_server: ThreadingHTTPServer,
    tmp_path: Path,
) -> None:
    base = _base_url(redirect_server)
    # Mix canonical and upper-case variants of the same custom key, plus an
    # unknown ordinary key that must survive untouched.
    target = f"{base}/abs?tenant_key={TENANT_VAL_1}&TENANT_KEY={TENANT_VAL_2}&page=1"
    policy = RedactionPolicy(query_keys=["tenant_key"], header_names=["x-tenant-secret"])
    analyzer = HTTPTapAnalyzer(http2=False, redaction_policy=policy)

    steps = analyzer.analyze_url(
        target,
        method=HTTPMethod.GET,
        headers={"X-Tenant-Secret": TENANT_HEADER_SECRET, "X-Trace": "trace-keep"},
    )

    # Transport received full raw values and header.
    assert RedirectRecordingHandler.received_paths == [
        f"/abs?tenant_key={TENANT_VAL_1}&TENANT_KEY={TENANT_VAL_2}&page=1"
    ]
    assert RedirectRecordingHandler.received_headers == [TENANT_HEADER_SECRET]

    step = steps[0]
    # Both duplicate, differently-cased query values are masked; order kept.
    assert step.url == f"{base}/abs?tenant_key=****&TENANT_KEY=****&page=1"
    assert step.raw_url == target
    assert step.request_headers["X-Tenant-Secret"] != TENANT_HEADER_SECRET
    assert step.request_headers["X-Trace"] == "trace-keep"

    # Terminal output carries masked URLs and no header secrets.
    output = _render(steps, target, policy)
    assert TENANT_VAL_1 not in output
    assert TENANT_VAL_2 not in output
    assert TENANT_HEADER_SECRET not in output
    assert "page=1" in output

    # JSON masks custom secrets while preserving unknown fields and shape.
    report = tmp_path / "custom.json"
    JSONExporter(Console(record=True), redaction_policy=policy).export(steps, target, str(report))
    raw = report.read_text()
    for secret in (TENANT_VAL_1, TENANT_VAL_2, TENANT_HEADER_SECRET):
        assert secret not in raw
    data = json.loads(raw)
    assert data["initial_url"] == f"{base}/abs?tenant_key=****&TENANT_KEY=****&page=1"
    request_headers = data["steps"][0]["request"]["headers"]
    assert request_headers["X-Tenant-Secret"] != TENANT_HEADER_SECRET
    assert request_headers["X-Trace"] == "trace-keep"


def test_cli_main_masks_custom_secrets_e2e(
    redirect_server: ThreadingHTTPServer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import httptap.cli as cli_module

    base = _base_url(redirect_server)
    target = f"{base}/abs?tenant_key={TENANT_VAL_1}&page=3"
    report = tmp_path / "cli-report.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "httptap",
            "--compact",
            "--redact-query-key",
            "tenant_key",
            "--redact-header",
            "x-tenant-secret",
            "--header",
            f"X-Tenant-Secret: {TENANT_HEADER_SECRET}",
            "--json",
            str(report),
            target,
        ],
    )

    exit_code = cli_module.main()

    assert exit_code == EXIT_SUCCESS
    # The request reached the server with the original value.
    assert RedirectRecordingHandler.received_paths == [f"/abs?tenant_key={TENANT_VAL_1}&page=3"]
    assert RedirectRecordingHandler.received_headers == [TENANT_HEADER_SECRET]

    captured = capsys.readouterr()
    assert TENANT_VAL_1 not in captured.out
    assert TENANT_HEADER_SECRET not in captured.out
    assert TENANT_HEADER_SECRET not in captured.err
    assert "page=3" in captured.out

    raw = report.read_text()
    assert TENANT_VAL_1 not in raw
    assert TENANT_HEADER_SECRET not in raw
    assert "page=3" in raw
    data = json.loads(raw)  # existing consumers can still parse the document
    assert data["initial_url"] == f"{base}/abs?tenant_key=****&page=3"
