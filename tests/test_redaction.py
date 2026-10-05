from __future__ import annotations

import pytest

from httptap.redaction import (
    DEFAULT_SENSITIVE_HEADERS,
    DEFAULT_SENSITIVE_QUERY_KEYS,
    RedactionPolicy,
)


class TestRedactUrlQuery:
    """Structural query redaction: only values are replaced."""

    def test_masks_default_sensitive_keys(self) -> None:
        policy = RedactionPolicy()

        result = policy.redact_url("https://api.example/v1?token=shh&page=2")

        assert result == "https://api.example/v1?token=****&page=2"

    def test_masks_repeated_parameters_individually(self) -> None:
        policy = RedactionPolicy()

        result = policy.redact_url("https://api.example/v1?token=aaa&token=bbb&page=3")

        assert result == "https://api.example/v1?token=****&token=****&page=3"
        assert "aaa" not in result
        assert "bbb" not in result

    def test_preserves_percent_encoding_of_keys_and_values(self) -> None:
        policy = RedactionPolicy()

        result = policy.redact_url("https://api.example/v1?api%5Fkey=ak%40secret&page=7")

        # Key spelling and encoding preserved; only the value is replaced.
        assert result == "https://api.example/v1?api%5Fkey=****&page=7"

    def test_key_matching_is_case_insensitive(self) -> None:
        policy = RedactionPolicy()

        result = policy.redact_url("https://api.example/v1?TOKEN=shh&Api_Key=k&page=1")

        assert result == "https://api.example/v1?TOKEN=****&Api_Key=****&page=1"

    def test_preserves_empty_value_parameter(self) -> None:
        policy = RedactionPolicy()

        result = policy.redact_url("https://api.example/v1?token=&page=1")

        assert result == "https://api.example/v1?token=****&page=1"

    def test_preserves_valueless_parameter(self) -> None:
        policy = RedactionPolicy()

        # A flag parameter without '=' carries no value to replace.
        result = policy.redact_url("https://api.example/v1?token&page=1")

        assert result == "https://api.example/v1?token&page=1"

    def test_preserves_trailing_separator_and_order(self) -> None:
        policy = RedactionPolicy()

        result = policy.redact_url("https://api.example/v1?page=1&token=shh&")

        assert result == "https://api.example/v1?page=1&token=****&"

    def test_keeps_non_sensitive_values_visible(self) -> None:
        policy = RedactionPolicy()

        result = policy.redact_url("https://api.example/v1?page=42&limit=10&sort=asc")

        assert result == "https://api.example/v1?page=42&limit=10&sort=asc"

    def test_masks_presigned_aws_components(self) -> None:
        policy = RedactionPolicy()
        url = "https://s3.example/bucket/key?X-Amz-Signature=deadbeef&X-Amz-Security-Token=tok&page=1"

        result = policy.redact_url(url)

        assert result == ("https://s3.example/bucket/key?X-Amz-Signature=****&X-Amz-Security-Token=****&page=1")
        assert "deadbeef" not in result

    def test_masks_query_shaped_fragment(self) -> None:
        policy = RedactionPolicy()

        result = policy.redact_url("https://app.example/cb#access_token=fragsecret&state=xyz")

        assert result == "https://app.example/cb#access_token=****&state=xyz"

    @pytest.mark.parametrize(
        "key",
        [
            "access_token",
            "refresh_token",
            "id_token",
            "apikey",
            "secret",
            "client_secret",
            "password",
            "passwd",
            "signature",
            "sig",
        ],
    )
    def test_default_key_set_covers_common_secret_names(self, key: str) -> None:
        policy = RedactionPolicy()
        url = f"https://api.example/v1?{key}=value&page=1"

        result = policy.redact_url(url)

        assert f"{key}=****" in result
        assert "value" not in result.replace("****", "")

    def test_none_and_empty_pass_through(self) -> None:
        policy = RedactionPolicy()

        assert policy.redact_url(None) is None
        assert policy.redact_url("") == ""

    def test_unparseable_url_returned_unchanged(self) -> None:
        policy = RedactionPolicy()
        url = "http://["  # malformed IPv6 literal

        assert policy.redact_url(url) == url

    def test_redaction_is_idempotent(self) -> None:
        policy = RedactionPolicy()
        url = "https://api.example/v1?token=shh&page=2"

        once = policy.redact_url(url)
        twice = policy.redact_url(once)

        assert once == twice == "https://api.example/v1?token=****&page=2"


class TestRedactUrlUserinfo:
    """Proxy/URL userinfo masking remains part of the same policy."""

    def test_masks_password_and_query_together(self) -> None:
        policy = RedactionPolicy()

        result = policy.redact_url("http://user:pwsecret@proxy:3128/path?token=qsecret&page=9")

        assert result == "http://user:****@proxy:3128/path?token=****&page=9"
        assert "pwsecret" not in result
        assert "qsecret" not in result

    def test_masks_bare_userinfo_token(self) -> None:
        policy = RedactionPolicy()

        result = policy.redact_url("socks5h://baretoken@gateway:1080")

        assert result == "socks5h://****@gateway:1080"

    def test_url_without_userinfo_or_sensitive_query_unchanged(self) -> None:
        policy = RedactionPolicy()

        assert policy.redact_url("http://proxy:3128/path?page=2") == "http://proxy:3128/path?page=2"


class TestCustomKeysAndHeaders:
    """Users extend the policy with extra query keys and header names."""

    def test_custom_query_keys_masked_case_insensitively(self) -> None:
        policy = RedactionPolicy(query_keys=["tenant_key"])

        result = policy.redact_url("https://api.example/v1?TENANT_KEY=shh&page=2")

        assert result == "https://api.example/v1?TENANT_KEY=****&page=2"

    def test_custom_query_keys_stripped_and_normalized(self) -> None:
        policy = RedactionPolicy(query_keys=["  TenantKey "])

        assert "tenantkey" in policy.query_keys
        assert policy.redact_url("https://h/p?tenantkey=v") == "https://h/p?tenantkey=****"

    def test_unknown_query_fields_are_not_dropped(self) -> None:
        policy = RedactionPolicy(query_keys=["tenant_key"])

        result = policy.redact_url("https://api.example/v1?tenant_key=shh&page=2&filter=name")

        assert result == "https://api.example/v1?tenant_key=****&page=2&filter=name"

    def test_custom_header_names_masked_case_insensitively(self) -> None:
        policy = RedactionPolicy(header_names=["x-tenant-secret"])
        headers = {"X-Tenant-Secret": "tennantvalue1234", "X-Trace": "visible"}

        result = policy.redact_headers(headers)

        assert result["X-Tenant-Secret"] != "tennantvalue1234"
        assert "****" in result["X-Tenant-Secret"]
        assert result["X-Trace"] == "visible"
        # The raw mapping is not mutated.
        assert headers["X-Tenant-Secret"] == "tennantvalue1234"

    def test_default_sensitive_headers_still_masked_with_extras(self) -> None:
        policy = RedactionPolicy(header_names=["x-tenant-secret"])
        headers = {
            "Authorization": "Bearer tokensecret123",
            "Proxy-Authorization": "Basic cHJveHlzZWNyZXQ=",
            "X-Tenant-Secret": "tennantvalue1234",
            "Accept": "application/json",
        }

        result = policy.redact_headers(headers)

        assert "tokensecret123" not in result["Authorization"]
        assert "cHJveHlzZWNyZXQ=" not in result["Proxy-Authorization"]
        assert "tennantvalue1234" not in result["X-Tenant-Secret"]
        assert result["Accept"] == "application/json"

    def test_default_sets_exposed(self) -> None:
        assert "authorization" in DEFAULT_SENSITIVE_HEADERS
        assert "proxy-authorization" in DEFAULT_SENSITIVE_HEADERS
        assert "token" in DEFAULT_SENSITIVE_QUERY_KEYS
        assert "api_key" in DEFAULT_SENSITIVE_QUERY_KEYS

    def test_redact_headers_none_and_empty(self) -> None:
        policy = RedactionPolicy()

        assert policy.redact_headers(None) == {}
        assert policy.redact_headers({}) == {}


class TestRedactText:
    """Free-form messages (exception strings) lose every learned secret."""

    def test_redacts_target_url_embedded_in_message(self) -> None:
        policy = RedactionPolicy()
        policy.learn_url("https://api.example/v1?token=msgsecret&page=2")

        message = "Request failed: failed to connect to https://api.example/v1?token=msgsecret&page=2"

        result = policy.redact_text(message)

        assert "msgsecret" not in (result or "")
        assert "token=****" in (result or "")
        assert "page=2" in (result or "")

    def test_redacts_timeout_message(self) -> None:
        policy = RedactionPolicy()
        policy.learn_url("https://slow.example/path?access_token=timeouttoken")

        result = policy.redact_text(
            "Request timeout: timed out requesting https://slow.example/path?access_token=timeouttoken."
        )

        assert "timeouttoken" not in (result or "")
        assert "access_token=****" in (result or "")
        assert result.endswith(".")

    def test_redacts_proxy_address_with_credentials(self) -> None:
        policy = RedactionPolicy()
        policy.learn_url("http://proxyuser:proxypassword@proxy.local:3128")

        result = policy.redact_text("tunnel to http://proxyuser:proxypassword@proxy.local:3128 failed")

        assert "proxypassword" not in (result or "")
        assert "http://proxyuser:****@proxy.local:3128" in (result or "")

    def test_redacts_learned_header_value_outside_urls(self) -> None:
        policy = RedactionPolicy()
        policy.learn_headers({"Authorization": "Bearer headersecret99"})

        result = policy.redact_text("server rejected credential 'Bearer headersecret99'")

        assert "headersecret99" not in (result or "")

    def test_redacts_percent_encoded_and_decoded_forms(self) -> None:
        policy = RedactionPolicy()
        policy.learn_url("https://api.example/v1?api_key=ak%40secret")

        assert "ak%40secret" not in (policy.redact_text("got ak%40secret twice") or "")
        assert "ak@secret" not in (policy.redact_text("got ak@secret twice") or "")

    def test_non_secret_text_unchanged(self) -> None:
        policy = RedactionPolicy()
        policy.learn_url("https://api.example/v1?token=shh&page=2")

        message = "DNS resolution failed for api.example: name not known"

        assert policy.redact_text(message) == message

    def test_short_values_respect_word_boundaries(self) -> None:
        policy = RedactionPolicy(header_names=["x-secret"])
        policy.learn_headers({"X-Secret": "abc"})

        # A learned value must not be scrubbed out of the middle of another word.
        assert policy.redact_text("xabcx abcd") == "xabcx abcd"
        assert "****" in (policy.redact_text("credential 'abc' rejected") or "")

    def test_none_and_empty_pass_through(self) -> None:
        policy = RedactionPolicy()

        assert policy.redact_text(None) is None
        assert policy.redact_text("") == ""


class TestLearning:
    """learn_url/learn_headers must cope with every URL/header edge case."""

    def test_header_names_property_exposes_defaults_and_extras(self) -> None:
        policy = RedactionPolicy(header_names=["x-tenant-secret"])

        names = policy.header_names

        assert "x-tenant-secret" in names
        assert "authorization" in names
        assert isinstance(names, frozenset)

    def test_learn_url_ignores_none_and_empty(self) -> None:
        policy = RedactionPolicy()

        policy.learn_url(None)
        policy.learn_url("")

        assert policy.redact_text("nothing learned") == "nothing learned"

    def test_learn_url_survives_unparseable_url(self) -> None:
        policy = RedactionPolicy()

        # Invalid IPv6 literal makes urlsplit raise ValueError; learning is
        # best-effort and must never propagate.
        policy.learn_url("http://[")

        assert policy.redact_text("http://[") == "http://["

    def test_learn_url_with_empty_userinfo_credential(self) -> None:
        policy = RedactionPolicy()

        policy.learn_url("http://:@host/")
        policy.learn_url("http://user:@host/?token=v")

        assert policy.redact_url("http://user:@host/?token=v") == "http://user:****@host/?token=****"

    def test_learn_headers_ignores_none_and_empty(self) -> None:
        policy = RedactionPolicy()

        policy.learn_headers(None)
        policy.learn_headers({})

        assert policy.redact_text("nothing learned") == "nothing learned"

    def test_learn_pairs_skips_valueless_parameter(self) -> None:
        policy = RedactionPolicy()

        # `token` without '=' is a valueless flag, not a secret value.
        policy.learn_url("https://api.example/v1?token&page=2")
        policy.learn_url("https://api.example/v1?token=&page=3")  # empty value too

        assert policy.redact_url("https://api.example/v1?token&page=2") == ("https://api.example/v1?token&page=2")
        assert policy.redact_url("https://api.example/v1?token=&page=3") == ("https://api.example/v1?token=****&page=3")
