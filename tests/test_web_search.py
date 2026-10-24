"""内置只读 WebSearchClient 单元测试（全虚构数据，零真实网络请求）。"""
from __future__ import annotations

import io
import json
import logging
import unittest
from urllib.error import HTTPError, URLError

from wxbot_framework.mcp_client import MCPConfigError, WebSearchConfig
from wxbot_framework.web_search import SEARCH_ERROR_MESSAGE, WebSearchClient


class _FakeHTTPResponse:
    def __init__(self, body: bytes, *, status: int = 200, headers: dict[str, str] | None = None) -> None:
        self._stream = io.BytesIO(body)
        self.status = status
        self.headers = headers or {"Content-Type": "application/json"}
        self.read_sizes: list[int | None] = []

    def read(self, size: int | None = None) -> bytes:
        self.read_sizes.append(size)
        return self._stream.read() if size is None else self._stream.read(size)

    def __enter__(self) -> _FakeHTTPResponse:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class _RecordingOpener:
    def __init__(self, handler) -> None:
        self._handler = handler
        self.requests: list[tuple[object, float | None]] = []

    def open(self, req, timeout: float | None = None):
        self.requests.append((req, timeout))
        return self._handler(req, timeout)


class TestWebSearchClient(unittest.TestCase):
    def test_tavily_post_request_and_parsing(self) -> None:
        def handler(req, timeout):
            payload = {
                "results": [
                    {
                        "title": "Python 3.13 发布说明",
                        "url": "https://docs.example.test/py313",
                        "content": "新增实验性自由线程与 JIT 基础能力。",
                    }
                ]
            }
            return _FakeHTTPResponse(json.dumps(payload, ensure_ascii=False).encode("utf-8"))

        opener = _RecordingOpener(handler)
        cfg = WebSearchConfig(
            enabled=True,
            provider="tavily",
            base_url="https://search.example.test",
            api_key="tvly_fake_key_alpha",
            timeout_seconds=8.0,
            max_results=5,
        )
        client = WebSearchClient(cfg, opener=opener)
        res = client.search("  Python 3.13 新特性  ", limit=3)

        self.assertTrue(res["ok"])
        self.assertEqual(res["count"], 1)
        self.assertEqual(res["results"][0]["title"], "Python 3.13 发布说明")
        self.assertEqual(res["results"][0]["url"], "https://docs.example.test/py313")
        self.assertIn("JIT", res["results"][0]["snippet"])

        req, used_timeout = opener.requests[0]
        self.assertEqual(req.get_method(), "POST")
        self.assertEqual(req.full_url, "https://search.example.test/search")
        self.assertEqual(used_timeout, 8.0)
        sent = json.loads(req.data.decode("utf-8"))
        self.assertEqual(sent["api_key"], "tvly_fake_key_alpha")
        self.assertEqual(sent["query"], "Python 3.13 新特性")
        self.assertEqual(sent["max_results"], 3)
        self.assertIs(sent["include_answer"], False)

    def test_bocha_nested_and_top_level_results(self) -> None:
        responses = [
            {
                "data": {
                    "webPages": {
                        "value": [
                            {
                                "name": "博查结果一",
                                "url": "https://bocha.example.test/1",
                                "summary": "摘要内容一",
                            }
                        ]
                    }
                }
            },
            {
                "results": [
                    {
                        "title": "博查顶层兼容结果",
                        "url": "https://bocha.example.test/2",
                        "snippet": "摘要内容二",
                    }
                ]
            },
        ]
        idx = 0

        def handler(req, timeout):
            nonlocal idx
            body = json.dumps(responses[idx], ensure_ascii=False).encode("utf-8")
            idx += 1
            return _FakeHTTPResponse(body)

        opener = _RecordingOpener(handler)
        cfg = WebSearchConfig.from_dict(
            {
                "enabled": True,
                "provider": "bocha",
                "base_url": "https://bocha.example.test",
                "api_key": "bocha_fake_token_beta",
            }
        )
        client = WebSearchClient(cfg, opener=opener)

        res1 = client.search("测试查询1", limit=4)
        self.assertTrue(res1["ok"])
        self.assertEqual(res1["results"][0]["title"], "博查结果一")
        self.assertEqual(res1["results"][0]["snippet"], "摘要内容一")

        req1, _ = opener.requests[0]
        self.assertEqual(req1.get_method(), "POST")
        self.assertEqual(req1.full_url, "https://bocha.example.test/v1/web-search")
        hdrs_ci = {k.lower(): v for k, v in req1.header_items()}
        self.assertEqual(hdrs_ci.get("authorization"), "Bearer bocha_fake_token_beta")
        sent1 = json.loads(req1.data.decode("utf-8"))
        self.assertEqual(sent1, {"query": "测试查询1", "count": 4, "summary": True})

        res2 = client.search("测试查询2")
        self.assertTrue(res2["ok"])
        self.assertEqual(res2["results"][0]["title"], "博查顶层兼容结果")

    def test_brave_get_request_and_parsing(self) -> None:
        def handler(req, timeout):
            payload = {
                "web": {
                    "results": [
                        {
                            "title": "Brave 检索条目",
                            "url": "https://brave.example.test/item",
                            "description": "Brave 摘要说明",
                        }
                    ]
                }
            }
            return _FakeHTTPResponse(json.dumps(payload, ensure_ascii=False).encode("utf-8"))

        opener = _RecordingOpener(handler)
        cfg = WebSearchConfig(
            enabled=True,
            provider="brave",
            base_url="https://brave-api.example.test",
            api_key="brave_sub_token_gamma",
        )
        client = WebSearchClient(cfg, opener=opener)
        res = client.search("MCP JSON-RPC", limit=2)

        self.assertTrue(res["ok"])
        self.assertEqual(res["results"][0]["title"], "Brave 检索条目")
        self.assertEqual(res["results"][0]["snippet"], "Brave 摘要说明")

        req, _ = opener.requests[0]
        self.assertEqual(req.get_method(), "GET")
        self.assertIn("/res/v1/web/search?", req.full_url)
        self.assertIn("count=2", req.full_url)
        hdrs_ci = {k.lower(): v for k, v in req.header_items()}
        self.assertEqual(hdrs_ci.get("x-subscription-token"), "brave_sub_token_gamma")
        self.assertEqual(hdrs_ci.get("accept"), "application/json")

    def test_searxng_without_api_key_loopback_http(self) -> None:
        def handler(req, timeout):
            payload = {
                "results": [
                    {
                        "title": "本地 SearXNG 条目",
                        "url": "http://127.0.0.1:8080/page",
                        "content": "自建元搜索返回摘要",
                    }
                ]
            }
            return _FakeHTTPResponse(json.dumps(payload, ensure_ascii=False).encode("utf-8"))

        opener = _RecordingOpener(handler)
        cfg = WebSearchConfig(
            enabled=True,
            provider="searxng",
            base_url="http://127.0.0.1:8888",
            api_key="",
        )
        client = WebSearchClient(cfg, opener=opener)
        res = client.search("自建搜索")

        self.assertTrue(res["ok"])
        self.assertEqual(res["count"], 1)
        req, _ = opener.requests[0]
        self.assertEqual(req.get_method(), "GET")
        self.assertIn("http://127.0.0.1:8888/search?", req.full_url)
        self.assertIn("format=json", req.full_url)

    def test_query_and_limit_validation(self) -> None:
        opener = _RecordingOpener(
            lambda req, timeout: _FakeHTTPResponse(b'{"results": []}')
        )
        cfg = WebSearchConfig(
            enabled=True,
            provider="tavily",
            base_url="https://search.example.test",
            api_key="tvly_fake_key",
        )
        client = WebSearchClient(cfg, opener=opener)

        with self.assertRaises(ValueError):
            client.search("")
        with self.assertRaises(ValueError):
            client.search("   \n\t  ")
        with self.assertRaises(ValueError):
            client.search("测" * 201)
        with self.assertRaises(ValueError):
            client.search(123)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            client.search("合法关键词", limit=True)  # type: ignore[arg-type]

        # limit 钳制在 1..10
        client.search("合法关键词", limit=0)
        client.search("合法关键词", limit=99)
        sent_low = json.loads(opener.requests[0][0].data.decode("utf-8"))
        sent_high = json.loads(opener.requests[1][0].data.decode("utf-8"))
        self.assertEqual(sent_low["max_results"], 1)
        self.assertEqual(sent_high["max_results"], 10)

    def test_result_sanitization_field_truncation_and_scheme_filter(self) -> None:
        fake_sk = "sk-" + "z" * 18
        fake_win_path = "C:" + "\\Users\\demo_bob\\Documents\\private.txt"
        raw_snippet = (
            f"图标 data:image/png;base64,iVBORw0KGgoAAAANSUhEUg== "
            f"泄露 Bearer leaked_bearer_token_987654 与 {fake_sk} "
            f"以及 wxid_user_bob 在 90000001@chatroom 路径 {fake_win_path} "
            + ("摘" * 350)
        )
        payload = {
            "results": [
                {
                    "title": "标题" * 80,
                    "url": "https://good.example.test/" + ("a" * 350),
                    "content": raw_snippet,
                },
                {
                    "title": "恶意伪协议",
                    "url": "javascript:alert(1)",
                    "content": "应被过滤",
                },
                {
                    "title": "内嵌凭据链接",
                    "url": "https://user:pass@evil.example.test/secret",
                    "content": "应被过滤",
                },
            ]
        }
        opener = _RecordingOpener(
            lambda req, timeout: _FakeHTTPResponse(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        )
        cfg = WebSearchConfig(
            enabled=True,
            provider="tavily",
            base_url="https://search.example.test",
            api_key="tvly_fake_key",
        )
        client = WebSearchClient(cfg, opener=opener)
        res = client.search("安全脱敏测试")

        self.assertTrue(res["ok"])
        self.assertEqual(res["count"], 1)
        item = res["results"][0]
        self.assertLessEqual(len(item["title"]), 100)
        self.assertLessEqual(len(item["url"]), 300)
        self.assertLessEqual(len(item["snippet"]), 300)
        self.assertIn("[图片]", item["snippet"])
        self.assertIn("Bearer [REDACTED]", item["snippet"])
        self.assertIn("[REDACTED]", item["snippet"])
        self.assertIn("[微信ID]", item["snippet"])
        self.assertIn("[群聊]", item["snippet"])
        self.assertIn("[路径]", item["snippet"])
        self.assertNotIn("leaked_bearer_token_987654", item["snippet"])
        self.assertNotIn(fake_sk, item["snippet"])
        self.assertNotIn("wxid_user_bob", item["snippet"])
        self.assertNotIn("90000001@chatroom", item["snippet"])
        self.assertNotIn("demo_bob", item["snippet"])

    def test_oversized_http_body_and_char_budget(self) -> None:
        huge_body = b'{"results":[{"title":"x","url":"https://a.test","content":"' + (b"x" * 60_000) + b'"}]}'
        fake_resp = _FakeHTTPResponse(huge_body)
        opener = _RecordingOpener(lambda req, timeout: fake_resp)
        cfg = WebSearchConfig(
            enabled=True,
            provider="tavily",
            base_url="https://search.example.test",
            api_key="tvly_secret_never_log",
        )
        client = WebSearchClient(cfg, opener=opener)
        res = client.search("超大响应测试")
        self.assertFalse(res["ok"])
        self.assertEqual(res["error"], SEARCH_ERROR_MESSAGE)
        self.assertEqual(fake_resp.read_sizes[0], 50_000)

        # 测试 max_result_chars 整体预算裁剪
        many_items = {
            "results": [
                {
                    "title": f"条目{i}",
                    "url": f"https://example.test/{i}",
                    "content": "内" * 280,
                }
                for i in range(6)
            ]
        }
        opener2 = _RecordingOpener(
            lambda req, timeout: _FakeHTTPResponse(json.dumps(many_items, ensure_ascii=False).encode("utf-8"))
        )
        cfg2 = WebSearchConfig(
            enabled=True,
            provider="tavily",
            base_url="https://search.example.test",
            api_key="tvly_fake",
            max_results=6,
            max_result_chars=600,
        )
        client2 = WebSearchClient(cfg2, opener=opener2)
        res2 = client2.search("预算裁剪")
        self.assertTrue(res2["ok"])
        self.assertTrue(res2.get("truncated"))
        serialized = json.dumps(res2, ensure_ascii=False, separators=(",", ":"))
        self.assertLessEqual(len(serialized), 600)

    def test_network_and_http_errors_never_leak_api_key(self) -> None:
        secret_key = "secret_search_key_do_not_leak_999"

        def raise_http_error(req, timeout):
            raise HTTPError(req.full_url, 401, f"Unauthorized key={secret_key}", {}, None)

        opener = _RecordingOpener(raise_http_error)
        cfg = WebSearchConfig(
            enabled=True,
            provider="bocha",
            base_url="https://bocha.example.test",
            api_key=secret_key,
        )
        client = WebSearchClient(cfg, opener=opener)

        with self.assertLogs("wxbot_framework.web_search", level=logging.WARNING) as cm:
            res = client.search("查天气")
        self.assertEqual(res, {"ok": False, "error": SEARCH_ERROR_MESSAGE})
        joined_logs = "\n".join(cm.output)
        self.assertNotIn(secret_key, joined_logs)

        opener_timeout = _RecordingOpener(lambda req, timeout: (_ for _ in ()).throw(URLError(TimeoutError("timed out"))))
        client_timeout = WebSearchClient(cfg, opener=opener_timeout)
        res_timeout = client_timeout.search("查新闻")
        self.assertEqual(res_timeout, {"ok": False, "error": SEARCH_ERROR_MESSAGE})

    def test_config_rejects_public_http_and_missing_key(self) -> None:
        with self.assertRaises(MCPConfigError):
            WebSearchConfig(
                enabled=True,
                provider="tavily",
                base_url="http://api.tavily.com",
                api_key="k",
            )
        cfg = WebSearchConfig(enabled=True, provider="tavily", base_url="https://api.tavily.com", api_key="")
        with self.assertRaises(MCPConfigError):
            WebSearchClient(cfg)


if __name__ == "__main__":
    unittest.main()
