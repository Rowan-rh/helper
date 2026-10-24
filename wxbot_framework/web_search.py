"""内置零依赖只读网页搜索客户端（标准库 urllib.request 实现）。

支持 4 类搜索源：
- tavily: POST {base_url}/search
- bocha:  POST {base_url}/v1/web-search (Authorization: Bearer <api_key>)
- brave:  GET  {base_url}/res/v1/web/search?q=...&count=... (X-Subscription-Token: <api_key>)
- searxng: GET {base_url}/search?q=...&format=json

安全保证：
- 单次 HTTP 响应最多读取 50,000 字节防内存溢出；
- 结果字段逐项经 sanitize_external_text 清洗并截断（title<=100, url<=300 仅限 http/https, snippet<=300）；
- 异常统一脱敏为可读降级字典，日志绝不输出 API Key 或查询正文。
"""
from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request

from .mcp_client import (
    MAX_HTTP_RESPONSE_BYTES,
    MCPConfigError,
    WebSearchConfig,
    build_safe_http_opener,
    sanitize_external_text,
)

logger = logging.getLogger(__name__)

MAX_SEARCH_RESPONSE_BYTES = MAX_HTTP_RESPONSE_BYTES
MAX_QUERY_CHARS = 200
MAX_TITLE_CHARS = 100
MAX_URL_CHARS = 300
MAX_SNIPPET_CHARS = 300
SEARCH_ERROR_MESSAGE = "网页搜索暂时不可用或超时，请勿重试"


def _join_endpoint(base_url: str, suffix: str) -> str:
    trimmed = base_url.rstrip("/")
    if trimmed.lower().endswith(suffix.lower()):
        return trimmed
    return trimmed + suffix


def _clean_result_url(raw_url: Any) -> str:
    if not isinstance(raw_url, str):
        return ""
    candidate = sanitize_external_text(raw_url).strip()
    if not candidate:
        return ""
    if any(ch.isspace() for ch in candidate):
        return ""
    parsed = urlsplit(candidate)
    scheme = (parsed.scheme or "").lower()
    if scheme not in ("http", "https") or not parsed.netloc:
        return ""
    if parsed.username is not None or parsed.password is not None or "@" in parsed.netloc:
        return ""
    clipped = candidate[:MAX_URL_CHARS]
    if not clipped.lower().startswith(("http://", "https://")):
        return ""
    return clipped


def _json_size(payload: Mapping[str, Any]) -> int:
    return len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))


class WebSearchClient:
    """轻量同步只读网页搜索客户端。"""

    def __init__(
        self,
        config: WebSearchConfig,
        *,
        opener: Any = None,
        env: Mapping[str, str] | None = None,
        max_response_bytes: int = MAX_SEARCH_RESPONSE_BYTES,
    ) -> None:
        if not isinstance(config, WebSearchConfig):
            raise MCPConfigError("config 必须是 WebSearchConfig 实例")
        resolved = config.resolve(env, require_credentials=bool(config.enabled))
        self.config = resolved
        self._base_url = resolved.base_url.rstrip("/")
        self._api_key = resolved.api_key
        self._opener = opener if opener is not None else build_safe_http_opener()
        self.max_response_bytes = int(max_response_bytes)

    def _build_request(self, query: str, limit: int) -> Request:
        provider = self.config.provider
        if provider == "tavily":
            endpoint = _join_endpoint(self._base_url, "/search")
            body = json.dumps(
                {
                    "api_key": self._api_key,
                    "query": query,
                    "max_results": limit,
                    "include_answer": False,
                },
                ensure_ascii=False,
            ).encode("utf-8")
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            return Request(endpoint, data=body, headers=headers, method="POST")

        if provider == "bocha":
            endpoint = _join_endpoint(self._base_url, "/v1/web-search")
            body = json.dumps(
                {
                    "query": query,
                    "count": limit,
                    "summary": True,
                },
                ensure_ascii=False,
            ).encode("utf-8")
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            }
            return Request(endpoint, data=body, headers=headers, method="POST")

        if provider == "brave":
            endpoint = _join_endpoint(self._base_url, "/res/v1/web/search")
            qs = urlencode({"q": query, "count": limit})
            sep = "&" if "?" in endpoint else "?"
            headers = {
                "Accept": "application/json",
                "X-Subscription-Token": self._api_key,
            }
            return Request(f"{endpoint}{sep}{qs}", headers=headers, method="GET")

        if provider == "searxng":
            endpoint = _join_endpoint(self._base_url, "/search")
            qs = urlencode({"q": query, "format": "json"})
            sep = "&" if "?" in endpoint else "?"
            headers = {"Accept": "application/json"}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"
            return Request(f"{endpoint}{sep}{qs}", headers=headers, method="GET")

        raise MCPConfigError(f"不支持的搜索 provider: {provider}")

    def _extract_raw_items(self, data: Any) -> list[Mapping[str, Any]]:
        if not isinstance(data, Mapping):
            return []
        provider = self.config.provider
        raw_list: Any = None

        if provider == "tavily":
            raw_list = data.get("results")
        elif provider == "bocha":
            nested_data = data.get("data")
            if isinstance(nested_data, Mapping):
                web_pages = nested_data.get("webPages")
                if isinstance(web_pages, Mapping):
                    raw_list = web_pages.get("value")
            if raw_list is None:
                web_pages = data.get("webPages")
                if isinstance(web_pages, Mapping):
                    raw_list = web_pages.get("value")
            if raw_list is None:
                raw_list = data.get("results")
        elif provider == "brave":
            web_obj = data.get("web")
            if isinstance(web_obj, Mapping):
                raw_list = web_obj.get("results")
            if raw_list is None:
                raw_list = data.get("results")
        elif provider == "searxng":
            raw_list = data.get("results")

        if not isinstance(raw_list, list):
            return []
        return [item for item in raw_list if isinstance(item, Mapping)]

    def _normalize_item(self, item: Mapping[str, Any]) -> dict[str, str] | None:
        url = _clean_result_url(item.get("url"))
        if not url:
            return None

        provider = self.config.provider
        if provider == "bocha":
            raw_title = item.get("name") or item.get("title") or ""
            raw_snippet = item.get("summary") or item.get("snippet") or item.get("content") or ""
        elif provider == "brave":
            raw_title = item.get("title") or ""
            raw_snippet = item.get("description") or item.get("snippet") or ""
        else:
            raw_title = item.get("title") or ""
            raw_snippet = item.get("content") or item.get("snippet") or ""

        title = sanitize_external_text(str(raw_title)).strip()[:MAX_TITLE_CHARS]
        snippet = sanitize_external_text(str(raw_snippet)).strip()[:MAX_SNIPPET_CHARS]
        return {
            "title": title,
            "url": url,
            "snippet": snippet,
        }

    def _fit_within_char_budget(
        self,
        clean_query: str,
        items: list[dict[str, str]],
    ) -> dict[str, Any]:
        max_chars = self.config.max_result_chars
        kept = list(items)
        payload: dict[str, Any] = {
            "ok": True,
            "query": clean_query,
            "results": kept,
            "count": len(kept),
        }
        if _json_size(payload) <= max_chars:
            return payload

        while kept:
            candidate = {
                "ok": True,
                "query": clean_query,
                "results": kept,
                "count": len(kept),
                "truncated": True,
            }
            if _json_size(candidate) <= max_chars:
                return candidate
            if len(kept) == 1 and kept[0]["snippet"]:
                # 单条仍超限时裁剪该条 snippet
                excess = _json_size(candidate) - max_chars
                new_len = max(0, len(kept[0]["snippet"]) - excess - 8)
                kept[0] = {**kept[0], "snippet": kept[0]["snippet"][:new_len]}
                candidate = {
                    "ok": True,
                    "query": clean_query,
                    "results": kept,
                    "count": len(kept),
                    "truncated": True,
                }
                if _json_size(candidate) <= max_chars:
                    return candidate
            kept.pop()

        return {
            "ok": True,
            "query": clean_query[:MAX_TITLE_CHARS],
            "results": [],
            "count": 0,
            "truncated": True,
        }

    def search(
        self,
        query: str,
        *,
        limit: int | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """执行只读关键词搜索并返回清洗截断后的结构化结果字典。"""
        if not isinstance(query, str):
            raise ValueError("搜索关键词 query 必须是字符串")
        clean_query = query.strip()
        if not (1 <= len(clean_query) <= MAX_QUERY_CHARS):
            raise ValueError(f"搜索关键词长度须为 1–{MAX_QUERY_CHARS} 个字符")

        if limit is None:
            eff_limit = self.config.max_results
        else:
            if type(limit) is not int:
                raise ValueError("limit 应为整数")
            eff_limit = max(1, min(10, limit))

        if timeout is None:
            eff_timeout = self.config.timeout_seconds
        else:
            if type(timeout) not in (int, float) or not math.isfinite(timeout) or float(timeout) <= 0:
                raise ValueError("timeout 应为大于 0 的秒数")
            eff_timeout = max(0.05, min(float(timeout), self.config.timeout_seconds))

        if self.config.provider != "searxng" and not self._api_key:
            logger.warning("网页搜索未配置有效 api_key (provider=%s)", self.config.provider)
            return {"ok": False, "error": SEARCH_ERROR_MESSAGE}

        try:
            req = self._build_request(clean_query, eff_limit)
            with self._opener.open(req, timeout=eff_timeout) as resp:
                first_chunk = resp.read(self.max_response_bytes)
                extra_byte = resp.read(1)
                if extra_byte:
                    raise ValueError("HTTP 响应超过 50000 字节上限")
            parsed = json.loads(first_chunk.decode("utf-8", errors="replace"))
            raw_items = self._extract_raw_items(parsed)
            normalized_items: list[dict[str, str]] = []
            for item in raw_items:
                norm = self._normalize_item(item)
                if norm is not None:
                    normalized_items.append(norm)
                    if len(normalized_items) >= eff_limit:
                        break
            return self._fit_within_char_budget(sanitize_external_text(clean_query), normalized_items)
        except HTTPError as exc:
            logger.warning("网页搜索请求失败 (provider=%s, status=%s)", self.config.provider, exc.code)
            return {"ok": False, "error": SEARCH_ERROR_MESSAGE}
        except (URLError, TimeoutError, OSError, ValueError) as exc:
            logger.warning("网页搜索不可用或响应无效 (provider=%s, err=%s)", self.config.provider, type(exc).__name__)
            return {"ok": False, "error": SEARCH_ERROR_MESSAGE}
