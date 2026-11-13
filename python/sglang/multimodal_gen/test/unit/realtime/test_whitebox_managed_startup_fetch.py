"""Exercise real urllib redirect dispatch without network access or real tokens."""

import gzip
import importlib.util
import io
import urllib.request
from email.message import Message
from pathlib import Path
from urllib.response import addinfourl

import pytest

ORIGIN = "https://world-test-platform.loopit.com.cn"


@pytest.fixture
def collector():
    path = (
        Path(__file__).resolve().parents[6]
        / "benchmark/whitebox/managed_startup/collect_cases.py"
    )
    spec = importlib.util.spec_from_file_location("managed_startup_collector", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def transport(monkeypatch):
    routes, sent = {}, []

    def respond(request):
        sent.append((request.full_url, request.get_header("Authorization")))
        code, headers, payload = routes.get(request.full_url, (200, {}, b"ok"))
        message = Message()
        for key, value in headers.items():
            message[key] = value
        response = addinfourl(io.BytesIO(payload), message, request.full_url, code)
        response.msg = "fixture"
        return response

    class HTTPS(urllib.request.HTTPSHandler):
        https_open = staticmethod(respond)

    class HTTP(urllib.request.HTTPHandler):
        http_open = staticmethod(respond)

    build = urllib.request.build_opener

    def opener(*handlers):
        return build(urllib.request.ProxyHandler({}), *handlers, HTTPS(), HTTP())

    # Also cover the vulnerable urlopen path when running against the old code.
    monkeypatch.setattr(urllib.request, "build_opener", opener)
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda request, **kw: opener().open(request, **kw)
    )
    return routes, sent


@pytest.mark.parametrize("code", [301, 302, 303, 307, 308])
@pytest.mark.parametrize(
    "target",
    [
        "https://outside.example/target",
        "//outside.example/target",
        "https://world-test-platform.loopit.com.cn.outside.example/target",
        ORIGIN + ":444/target",
        "http://world-test-platform.loopit.com.cn/target",
        "https://user@world-test-platform.loopit.com.cn/target",
    ],
)
def test_authenticated_redirect_cannot_leave_official_origin(
    collector, transport, code, target
):
    routes, sent = transport
    routes[ORIGIN + "/start"] = (code, {"Location": target}, b"")
    with pytest.raises(ValueError):
        collector.fetch(ORIGIN + "/start", "fixture-token")
    assert sent == [(ORIGIN + "/start", "Bearer fixture-token")]


@pytest.mark.parametrize(
    "url",
    [
        "https://outside.example/start",
        "http://world-test-platform.loopit.com.cn/start",
        ORIGIN + ":444/start",
        ORIGIN + ":0/start",
        "https://user:password@world-test-platform.loopit.com.cn/start",
    ],
)
def test_authenticated_initial_url_is_checked_before_sending(collector, transport, url):
    with pytest.raises(ValueError):
        collector.fetch(url, "fixture-token")
    assert transport[1] == []


def test_same_origin_chain_keeps_auth_then_rejects_cross_origin(collector, transport):
    routes, sent = transport
    routes[ORIGIN + "/start"] = (302, {"Location": "/next"}, b"")
    routes[ORIGIN + "/next"] = (307, {"Location": "//outside.example/end"}, b"")
    with pytest.raises(ValueError):
        collector.fetch(ORIGIN + "/start", "fixture-token")
    assert sent == [
        (ORIGIN + "/start", "Bearer fixture-token"),
        (ORIGIN + "/next", "Bearer fixture-token"),
    ]


def test_same_origin_https_redirect_succeeds_with_explicit_default_port(
    collector, transport
):
    routes, sent = transport
    routes[ORIGIN + "/start"] = (302, {"Location": ORIGIN + ":443/end"}, b"")
    assert collector.fetch(ORIGIN + "/start", "fixture-token") == b"ok"
    assert sent[-1] == (ORIGIN + ":443/end", "Bearer fixture-token")


def test_public_cdn_redirect_is_https_and_has_no_authorization(collector, transport):
    routes, sent = transport
    routes[ORIGIN + "/public"] = (
        302,
        {"Location": "https://cdn.example/manifest"},
        b"",
    )
    routes["https://cdn.example/manifest"] = (
        200,
        {"Content-Encoding": "gzip"},
        gzip.compress(b"manifest"),
    )
    assert collector.fetch(ORIGIN + "/public") == b"manifest"
    assert sent == [(ORIGIN + "/public", None), ("https://cdn.example/manifest", None)]


def test_public_cdn_cannot_redirect_to_http(collector, transport):
    routes, sent = transport
    routes["https://cdn.example/start"] = (
        302,
        {"Location": "http://cdn.example/end"},
        b"",
    )
    with pytest.raises(ValueError):
        collector.fetch("https://cdn.example/start")
    assert sent == [("https://cdn.example/start", None)]


def test_rejected_redirect_closes_response(collector):
    response = io.BytesIO(b"redirect")
    handler = collector.HTTPSRedirects(collector.https_origin(ORIGIN))
    with pytest.raises(ValueError):
        handler.redirect_request(
            urllib.request.Request(ORIGIN),
            response,
            302,
            "Found",
            {},
            "https://outside.example/",
        )
    assert response.closed


@pytest.mark.parametrize("suffix", ["", "/", ":443", ":443/"])
def test_configured_platform_origin_is_canonical(collector, suffix):
    assert collector.platform_origin(ORIGIN + suffix) == ORIGIN


@pytest.mark.parametrize(
    "url",
    [
        "https://outside.example",
        "http://world-test-platform.loopit.com.cn",
        ORIGIN + "/path",
        ORIGIN + "?q=1",
        ORIGIN + "#fragment",
    ],
)
def test_configured_platform_origin_rejects_untrusted_or_non_origin_url(collector, url):
    with pytest.raises(ValueError):
        collector.platform_origin(url)
