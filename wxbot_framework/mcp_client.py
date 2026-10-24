"""标准库轻量同步 MCP JSON-RPC 2.0 客户端、配置校验、安全脱敏与子进程树回收。

零第三方依赖，仅使用 Python 标准库（urllib.request + subprocess + ctypes）：
- 支持 streamable_http（application/json 与 text/event-stream SSE、Mcp-Session-Id 会话管理）
- 支持 stdio（最小安全环境变量白名单、stderr 脱敏落盘、Windows Job Object + taskkill / POSIX 进程组整树回收）
- 命名空间隔离（mcp__<server>__<tool>，64 字符哈希截断，保留工具名防覆盖）
- JSON Schema 归一化（definitions -> $defs，清理悬空 required，保留上下文字段自动重命名还原）
- Fail-Closed 只读门禁、输出脱敏与连续 3 次失败 60 秒熔断器（CircuitBreaker）
"""
from __future__ import annotations

import ctypes
import dataclasses
import fnmatch
import hashlib
import json
import logging
import math
import os
import re
import shutil
import signal
import subprocess
import threading
import time
import unicodedata
from collections.abc import Mapping, Set as AbstractSet
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .agent import RESERVED_PARAMETERS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------- 常量与异常

MCP_PROTOCOL_VERSION = "2025-03-26"
MAX_HTTP_RESPONSE_BYTES = 50_000
MAX_STDIO_LINE_BYTES = 512_000
MAX_RAW_TOOL_TEXT_CHARS = 50_000
MAX_TOOLS_LIST_PAGES = 20

RESERVED_TOOL_NAMES: frozenset[str] = frozenset({
    "get_group_context",
    "get_member_recent_messages",
    "list_group_members",
    "get_group_status",
    "memory_save",
    "memory_delete",
    "reminder_create",
    "reminder_list",
    "reminder_cancel",
    "web_search",
})

VALID_WEB_SEARCH_PROVIDERS: frozenset[str] = frozenset({
    "tavily",
    "bocha",
    "brave",
    "searxng",
})

PROVIDER_DEFAULT_BASE_URLS: Mapping[str, str] = MappingProxyType({
    "tavily": "https://api.tavily.com",
    "bocha": "https://api.bochaai.com",
    "brave": "https://api.search.brave.com",
    "searxng": "",
})

VALID_TRANSPORTS: frozenset[str] = frozenset({"streamable_http", "stdio"})
VALID_ACCESS_LEVELS: frozenset[str] = frozenset({"public", "vip", "self"})
LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "::1", "localhost"})

# stdio 子进程最小安全系统环境变量白名单（大小写不敏感匹配）
SAFE_STDIO_ENV_KEYS: frozenset[str] = frozenset({
    # Windows 核心系统变量
    "SYSTEMROOT",
    "WINDIR",
    "COMSPEC",
    "PATHEXT",
    "PATH",
    "TEMP",
    "TMP",
    "APPDATA",
    "LOCALAPPDATA",
    "USERPROFILE",
    "HOMEDRIVE",
    "HOMEPATH",
    "PROGRAMDATA",
    "PROGRAMFILES",
    "PROGRAMFILES(X86)",
    "COMMONPROGRAMFILES",
    "COMMONPROGRAMW6432",
    "PROGRAMW6432",
    "NUMBER_OF_PROCESSORS",
    "PROCESSOR_ARCHITECTURE",
    "OS",
    # POSIX 核心系统变量
    "HOME",
    "TMPDIR",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "USER",
    "LOGNAME",
    "SHELL",
    "XDG_CACHE_HOME",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
})


class MCPConfigError(ValueError):
    """MCP / WebSearch 静态配置语法、类型、安全约束或环境变量占位解析错误。"""


class MCPError(RuntimeError):
    """MCP 运行期连接、握手、协议、超时、熔断或远端执行错误。"""


# ---------------------------------------------------------------- 安全校验与外部文本脱敏

_ENV_PLACEHOLDER_RE = re.compile(r"\$\{([^}]*)\}")
_ENV_VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_DATA_IMAGE_BASE64_RE = re.compile(
    r"data:image/[^;\s]+;base64,\s*[A-Za-z0-9+/=]+(?:\s+[A-Za-z0-9+/=]{16,}|\s+[A-Za-z0-9+/=]*=+)*",
    re.IGNORECASE,
)
_BEARER_TOKEN_RE = re.compile(r"Bearer\s+[A-Za-z0-9._\-+/=]{8,}", re.IGNORECASE)
_SK_SECRET_RE = re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{12,}")
_WXID_RE = re.compile(r"wxid_[a-zA-Z0-9_-]+", re.IGNORECASE)
_CHATROOM_RE = re.compile(r"[a-zA-Z0-9_-]+@chatroom", re.IGNORECASE)
_WIN_USER_PATH_RE = re.compile(
    r"[A-Za-z]:[\\/]+Users[\\/]+[^\\/\s\"'<>|*?，。；：！？）》】]+(?:[\\/]+[^\\/\s\"'<>|*?，。；：！？）》】]+)*",
    re.IGNORECASE,
)

# 工具描述中的常见间接 Prompt 注入特征（仅记录 WARNING，不误杀正常服务）
_DESCRIPTION_INJECTION_PATTERNS = tuple(
    (re.compile(pattern, re.IGNORECASE), label)
    for pattern, label in (
        (r"ignore\s+(all\s+)?previous\s+instructions", "ignore_previous_instructions"),
        (r"you\s+are\s+now\s+a", "role_override"),
        (r"system\s*:\s*", "system_tag"),
        (r"<\s*(system|human|assistant)\s*>", "chat_role_xml"),
        (r"忽略(之前|上述|所有).{0,8}(指令|规则|约束)", "cn_ignore_instructions"),
    )
)


def validate_endpoint_url(url: str, *, field_name: str = "url") -> str:
    """校验远程端点 URL：仅允许 https:// 或本机回环 http://，且禁止内嵌认证凭据。"""
    if not isinstance(url, str):
        raise MCPConfigError(f"{field_name} 必须是字符串")
    cleaned = url.strip()
    if not cleaned:
        raise MCPConfigError(f"{field_name} 不能为空")
    if any(ch.isspace() or ord(ch) < 0x20 for ch in cleaned):
        raise MCPConfigError(f"{field_name} 不能包含空白或控制字符")

    parsed = urlsplit(cleaned)
    if parsed.username is not None or parsed.password is not None or "@" in (parsed.netloc or ""):
        raise MCPConfigError(f"{field_name} 不得包含用户名或密码认证信息")

    scheme = (parsed.scheme or "").lower()
    hostname = (parsed.hostname or "").lower()
    if not hostname or not parsed.netloc:
        raise MCPConfigError(f"{field_name} 缺少有效主机名")

    is_loopback = hostname in LOOPBACK_HOSTS
    if scheme == "https" or (scheme == "http" and is_loopback):
        return cleaned
    raise MCPConfigError(f"{field_name} 必须使用 https://（仅本机回环 127.0.0.1/::1/localhost 允许 http://）")


def resolve_env_placeholders(
    value: str,
    env: Mapping[str, str] | None = None,
    *,
    strict: bool = True,
) -> str:
    """解析字符串中的 ${VAR} 环境变量占位符；strict=True 时缺失或为空即抛 MCPConfigError。

    任何异常消息中仅包含变量名，绝不拼接或回显环境变量值或原字符串内容。
    """
    if not isinstance(value, str):
        raise MCPConfigError("环境变量解析目标必须是字符串")
    if "${" not in value:
        return value

    source = os.environ if env is None else env

    def _replace(match: re.Match[str]) -> str:
        var_name = match.group(1)
        if not _ENV_VAR_NAME_RE.fullmatch(var_name):
            if strict:
                raise MCPConfigError("配置中存在格式无效的环境变量占位符")
            return ""
        resolved = source.get(var_name, "")
        if not isinstance(resolved, str) or not resolved:
            if strict:
                raise MCPConfigError(f"环境变量 {var_name} 未设置或为空")
            return ""
        return resolved

    # 检查是否存在未闭合的 ${ 片段
    cursor = 0
    pieces: list[str] = []
    while cursor < len(value):
        idx = value.find("${", cursor)
        if idx == -1:
            pieces.append(value[cursor:])
            break
        pieces.append(value[cursor:idx])
        end_idx = value.find("}", idx + 2)
        if end_idx == -1:
            if strict:
                raise MCPConfigError("配置中存在未闭合的环境变量占位符")
            pieces.append(value[idx:])
            break
        segment = value[idx : end_idx + 1]
        pieces.append(_ENV_PLACEHOLDER_RE.sub(_replace, segment))
        cursor = end_idx + 1
    return "".join(pieces)


def _strip_dangerous_chars(text: str) -> str:
    """剔除 NUL、不可见控制字符/格式字符（保留 \\n 与 \\t）及 Unicode Tags (U+E0000..U+E007F)。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    out: list[str] = []
    for ch in text:
        if ch in ("\n", "\t"):
            out.append(ch)
            continue
        code = ord(ch)
        if code == 0 or 0xE0000 <= code <= 0xE007F or code < 0x20 or 0x7F <= code <= 0x9F:
            continue
        if unicodedata.category(ch) in ("Cc", "Cf"):
            continue
        out.append(ch)
    return "".join(out)


def sanitize_external_text(text: str) -> str:
    """清洗并脱敏外部搜索或 MCP 返回文本及错误消息。"""
    if not isinstance(text, str) or not text:
        return ""
    cleaned = _DATA_IMAGE_BASE64_RE.sub("[图片]", text)
    cleaned = _strip_dangerous_chars(cleaned)
    cleaned = _BEARER_TOKEN_RE.sub("Bearer [REDACTED]", cleaned)
    cleaned = _SK_SECRET_RE.sub("[REDACTED]", cleaned)
    cleaned = _WXID_RE.sub("[微信ID]", cleaned)
    cleaned = _CHATROOM_RE.sub("[群聊]", cleaned)
    cleaned = _WIN_USER_PATH_RE.sub("[路径]", cleaned)
    return cleaned


# ---------------------------------------------------------------- 配置模型与静态校验

def _require_bool(value: Any, field_name: str) -> bool:
    if type(value) is not bool:
        raise MCPConfigError(f"{field_name} 应为布尔值")
    return value


def _require_int_range(value: Any, field_name: str, low: int, high: int) -> int:
    if type(value) is not int or not (low <= value <= high):
        raise MCPConfigError(f"{field_name} 应为 {low}–{high} 的整数")
    return value


def _require_float_range(value: Any, field_name: str, low: float, high: float) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or not (low <= float(value) <= high):
        raise MCPConfigError(f"{field_name} 应为 {low}–{high} 的数字")
    return float(value)


def _require_str_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise MCPConfigError(f"{field_name} 应为字符串列表")
    items: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise MCPConfigError(f"{field_name} 中的元素必须是字符串")
        items.append(item)
    return tuple(items)


def _require_str_frozenset(value: Any, field_name: str) -> frozenset[str]:
    if isinstance(value, str) or not isinstance(value, (list, tuple, set, frozenset)):
        raise MCPConfigError(f"{field_name} 应为字符串列表或集合")
    items: set[str] = set()
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise MCPConfigError(f"{field_name} 中的元素必须是非空字符串")
        items.add(item.strip())
    return frozenset(items)


def _require_str_mapping(value: Any, field_name: str) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise MCPConfigError(f"{field_name} 应为 JSON 字符串映射对象")
    result: dict[str, str] = {}
    for k, v in value.items():
        if not isinstance(k, str) or not k.strip() or not isinstance(v, str):
            raise MCPConfigError(f"{field_name} 的键值都必须是字符串且键非空")
        if "\x00" in k or "\x00" in v:
            raise MCPConfigError(f"{field_name} 不能包含空字节")
        result[k] = v
    return MappingProxyType(result)


@dataclasses.dataclass(frozen=True)
class WebSearchConfig:
    enabled: bool = False
    provider: str = "tavily"
    base_url: str = "https://api.tavily.com"
    api_key: str = ""
    timeout_seconds: float = 10.0
    max_results: int = 5
    max_result_chars: int = 3000

    def __post_init__(self) -> None:
        _require_bool(self.enabled, "web_search.enabled")
        if not isinstance(self.provider, str) or self.provider not in VALID_WEB_SEARCH_PROVIDERS:
            supported = ", ".join(sorted(VALID_WEB_SEARCH_PROVIDERS))
            raise MCPConfigError(f"web_search.provider 只能是 {supported}")
        if not isinstance(self.base_url, str):
            raise MCPConfigError("web_search.base_url 必须是字符串")
        if not isinstance(self.api_key, str):
            raise MCPConfigError("web_search.api_key 必须是字符串")
        object.__setattr__(
            self,
            "timeout_seconds",
            _require_float_range(self.timeout_seconds, "web_search.timeout_seconds", 1.0, 30.0),
        )
        _require_int_range(self.max_results, "web_search.max_results", 1, 10)
        _require_int_range(self.max_result_chars, "web_search.max_result_chars", 500, 12000)

        stripped_url = self.base_url.strip()
        if stripped_url and "${" not in stripped_url:
            validated = validate_endpoint_url(stripped_url, field_name="web_search.base_url")
            object.__setattr__(self, "base_url", validated)

    def resolve(self, env: Mapping[str, str] | None = None, *, require_credentials: bool = True) -> WebSearchConfig:
        """解析环境变量占位符并校验启用态必填字段。"""
        resolved_url = resolve_env_placeholders(self.base_url, env, strict=True).strip()
        validated_url = validate_endpoint_url(resolved_url, field_name="web_search.base_url")
        resolved_key = resolve_env_placeholders(self.api_key, env, strict=True).strip()
        if require_credentials and self.provider != "searxng" and not resolved_key:
            raise MCPConfigError(f"启用 web_search ({self.provider}) 时必须填写有效的 api_key")
        return dataclasses.replace(self, base_url=validated_url, api_key=resolved_key)

    @classmethod
    def from_dict(cls, data: Any) -> WebSearchConfig:
        if data is None:
            return cls()
        if not isinstance(data, Mapping):
            raise MCPConfigError("web_search 配置必须是 JSON 对象")
        provider = data.get("provider", "tavily")
        if not isinstance(provider, str) or provider not in VALID_WEB_SEARCH_PROVIDERS:
            supported = ", ".join(sorted(VALID_WEB_SEARCH_PROVIDERS))
            raise MCPConfigError(f"web_search.provider 只能是 {supported}")
        default_base_url = PROVIDER_DEFAULT_BASE_URLS.get(provider, "https://api.tavily.com")
        return cls(
            enabled=data.get("enabled", False),
            provider=provider,
            base_url=data.get("base_url", default_base_url),
            api_key=data.get("api_key", ""),
            timeout_seconds=data.get("timeout_seconds", 10.0),
            max_results=data.get("max_results", 5),
            max_result_chars=data.get("max_result_chars", 3000),
        )


@dataclasses.dataclass(frozen=True)
class MCPServerConfig:
    name: str
    enabled: bool = False
    transport: str = "streamable_http"
    url: str = ""
    headers: Mapping[str, str] = dataclasses.field(default_factory=lambda: MappingProxyType({}))
    command: str = ""
    args: tuple[str, ...] = ()
    env: Mapping[str, str] = dataclasses.field(default_factory=lambda: MappingProxyType({}))
    cwd: str | None = None
    access: str = "vip"
    allow_in_group: bool = False
    read_only: bool = False
    read_only_tools: frozenset[str] = frozenset()
    trust_annotations: bool = False
    allow_write: bool = False
    include_tools: tuple[str, ...] = ()
    exclude_tools: tuple[str, ...] = ()
    timeout_seconds: float = 10.0
    max_result_chars: int = 4000

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise MCPConfigError("MCP Server 名称必须是非空字符串")
        prefix = f"mcp_servers.{self.name}"
        _require_bool(self.enabled, f"{prefix}.enabled")
        if not isinstance(self.transport, str) or self.transport not in VALID_TRANSPORTS:
            raise MCPConfigError(f"{prefix}.transport 仅支持 streamable_http 或 stdio")
        if not isinstance(self.url, str):
            raise MCPConfigError(f"{prefix}.url 必须是字符串")
        object.__setattr__(self, "headers", _require_str_mapping(self.headers, f"{prefix}.headers"))
        if not isinstance(self.command, str):
            raise MCPConfigError(f"{prefix}.command 必须是字符串")
        object.__setattr__(self, "args", _require_str_tuple(self.args, f"{prefix}.args"))
        object.__setattr__(self, "env", _require_str_mapping(self.env, f"{prefix}.env"))
        if self.cwd is not None and (not isinstance(self.cwd, str) or not self.cwd.strip()):
            raise MCPConfigError(f"{prefix}.cwd 必须是非空字符串或 null")
        if not isinstance(self.access, str) or self.access not in VALID_ACCESS_LEVELS:
            raise MCPConfigError(f"{prefix}.access 只能是 public、vip 或 self")
        _require_bool(self.allow_in_group, f"{prefix}.allow_in_group")
        _require_bool(self.read_only, f"{prefix}.read_only")
        object.__setattr__(
            self,
            "read_only_tools",
            _require_str_frozenset(self.read_only_tools, f"{prefix}.read_only_tools"),
        )
        _require_bool(self.trust_annotations, f"{prefix}.trust_annotations")
        _require_bool(self.allow_write, f"{prefix}.allow_write")
        if self.access == "public" and self.allow_write:
            raise MCPConfigError(f"{prefix} 在 access='public' 时禁止开启 allow_write")
        object.__setattr__(self, "include_tools", _require_str_tuple(self.include_tools, f"{prefix}.include_tools"))
        object.__setattr__(self, "exclude_tools", _require_str_tuple(self.exclude_tools, f"{prefix}.exclude_tools"))
        object.__setattr__(
            self,
            "timeout_seconds",
            _require_float_range(self.timeout_seconds, f"{prefix}.timeout_seconds", 1.0, 30.0),
        )
        _require_int_range(self.max_result_chars, f"{prefix}.max_result_chars", 500, 12000)

        stripped_url = self.url.strip()
        if self.transport == "streamable_http" and stripped_url and "${" not in stripped_url:
            validated = validate_endpoint_url(stripped_url, field_name=f"{prefix}.url")
            object.__setattr__(self, "url", validated)

    def resolve(self, env_source: Mapping[str, str] | None = None) -> MCPServerConfig:
        """解析该 Server 配置中的环境变量占位符并校验启用态必填项。"""
        prefix = f"mcp_servers.{self.name}"
        if self.transport == "streamable_http":
            resolved_url = resolve_env_placeholders(self.url, env_source, strict=True).strip()
            validated_url = validate_endpoint_url(resolved_url, field_name=f"{prefix}.url")
            resolved_headers = {
                k: resolve_env_placeholders(v, env_source, strict=True)
                for k, v in self.headers.items()
            }
            return dataclasses.replace(
                self,
                url=validated_url,
                headers=MappingProxyType(resolved_headers),
            )

        resolved_command = resolve_env_placeholders(self.command, env_source, strict=True).strip()
        if not resolved_command:
            raise MCPConfigError(f"{prefix}.command 在 stdio 模式下不能为空")
        resolved_args = tuple(resolve_env_placeholders(arg, env_source, strict=True) for arg in self.args)
        resolved_env = {
            k: resolve_env_placeholders(v, env_source, strict=True)
            for k, v in self.env.items()
        }
        resolved_cwd = (
            resolve_env_placeholders(self.cwd, env_source, strict=True).strip()
            if self.cwd is not None
            else None
        )
        if resolved_cwd == "":
            raise MCPConfigError(f"{prefix}.cwd 不能为空字符串")
        return dataclasses.replace(
            self,
            command=resolved_command,
            args=resolved_args,
            env=MappingProxyType(resolved_env),
            cwd=resolved_cwd,
        )

    @classmethod
    def from_dict(cls, name: str, data: Any) -> MCPServerConfig:
        if not isinstance(data, Mapping):
            raise MCPConfigError(f"mcp_servers.{name} 必须是 JSON 对象")
        return cls(
            name=name,
            enabled=data.get("enabled", False),
            transport=data.get("transport", "streamable_http"),
            url=data.get("url", ""),
            headers=data.get("headers", {}),
            command=data.get("command", ""),
            args=data.get("args", ()),
            env=data.get("env", {}),
            cwd=data.get("cwd"),
            access=data.get("access", "vip"),
            allow_in_group=data.get("allow_in_group", False),
            read_only=data.get("read_only", False),
            read_only_tools=data.get("read_only_tools", ()),
            trust_annotations=data.get("trust_annotations", False),
            allow_write=data.get("allow_write", False),
            include_tools=data.get("include_tools", ()),
            exclude_tools=data.get("exclude_tools", ()),
            timeout_seconds=data.get("timeout_seconds", 10.0),
            max_result_chars=data.get("max_result_chars", 4000),
        )


@dataclasses.dataclass(frozen=True)
class MCPConfig:
    enabled: bool = False
    web_search: WebSearchConfig = dataclasses.field(default_factory=WebSearchConfig)
    mcp_servers: Mapping[str, MCPServerConfig] = dataclasses.field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        _require_bool(self.enabled, "enabled")
        if not isinstance(self.web_search, WebSearchConfig):
            raise MCPConfigError("web_search 必须是 WebSearchConfig 实例")
        if not isinstance(self.mcp_servers, Mapping):
            raise MCPConfigError("mcp_servers 必须是 JSON 对象")
        servers: dict[str, MCPServerConfig] = {}
        for name, srv in self.mcp_servers.items():
            if not isinstance(name, str) or not name.strip() or not isinstance(srv, MCPServerConfig):
                raise MCPConfigError("mcp_servers 条目无效")
            servers[name] = srv
        object.__setattr__(self, "mcp_servers", MappingProxyType(servers))

    def validate_active(self, env: Mapping[str, str] | None = None) -> MCPConfig:
        """当顶层 enabled=True 时，解析并严格校验所有已启用的 web_search 与 mcp_servers。"""
        if not self.enabled:
            return self
        ws = self.web_search.resolve(env, require_credentials=True) if self.web_search.enabled else self.web_search
        resolved_servers: dict[str, MCPServerConfig] = {}
        for name, srv in self.mcp_servers.items():
            resolved_servers[name] = srv.resolve(env) if srv.enabled else srv
        return dataclasses.replace(
            self,
            web_search=ws,
            mcp_servers=MappingProxyType(resolved_servers),
        )

    @classmethod
    def from_dict(cls, data: Any, *, env: Mapping[str, str] | None = None) -> MCPConfig:
        if not isinstance(data, Mapping):
            raise MCPConfigError("MCP 配置文件根节点必须是 JSON 对象")
        enabled = _require_bool(data.get("enabled", False), "enabled")
        ws = WebSearchConfig.from_dict(data.get("web_search"))
        raw_servers = data.get("mcp_servers", {})
        if raw_servers is None:
            raw_servers = {}
        if not isinstance(raw_servers, Mapping):
            raise MCPConfigError("mcp_servers 必须是 JSON 对象")
        servers: dict[str, MCPServerConfig] = {}
        for name, raw_srv in raw_servers.items():
            if not isinstance(name, str) or not name.strip():
                raise MCPConfigError("mcp_servers 的键必须是非空字符串")
            servers[name] = MCPServerConfig.from_dict(name, raw_srv)
        cfg = cls(enabled=enabled, web_search=ws, mcp_servers=servers)
        return cfg.validate_active(env)

    @classmethod
    def from_file(cls, path: str | Path, *, env: Mapping[str, str] | None = None) -> MCPConfig:
        file_path = Path(path)
        if not file_path.exists():
            return cls(enabled=False)
        try:
            raw_text = file_path.read_text(encoding="utf-8-sig")
        except OSError as exc:
            raise MCPConfigError(f"无法读取 MCP 配置文件 {file_path.name}") from exc
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise MCPConfigError(
                f"MCP 配置文件 {file_path.name} JSON 语法错误（第 {exc.lineno} 行第 {exc.colno} 列）"
            ) from exc
        return cls.from_dict(parsed, env=env)


# ---------------------------------------------------------------- 命名空间、只读判定与 Schema 归一化

def _sanitize_name_part(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", str(value or "").strip())
    return cleaned or "_"


def make_mcp_tool_name(server_name: str, tool_name: str) -> str:
    """生成隔离工具名 mcp__<server>__<tool>；超 64 字符保留前 55 字符 + _<8位确定性SHA256>。"""
    if not isinstance(server_name, str) or not server_name.strip():
        raise ValueError("server_name 不能为空")
    if not isinstance(tool_name, str) or not tool_name.strip():
        raise ValueError("tool_name 不能为空")
    full_name = f"mcp__{_sanitize_name_part(server_name)}__{_sanitize_name_part(tool_name)}"
    if len(full_name) <= 64:
        return full_name
    digest = hashlib.sha256(full_name.encode("utf-8")).hexdigest()[:8]
    return f"{full_name[:55]}_{digest}"


def _matches_patterns(name: str, patterns: tuple[str, ...]) -> bool:
    for pattern in patterns:
        if name == pattern or fnmatch.fnmatchcase(name, pattern):
            return True
    return False


def is_mcp_tool_included(tool_name: str, server: MCPServerConfig) -> bool:
    """按 include_tools / exclude_tools 判断工具是否暴露；非空 include_tools 优先级高于 exclude_tools。"""
    if server.include_tools:
        return _matches_patterns(tool_name, server.include_tools)
    if server.exclude_tools:
        return not _matches_patterns(tool_name, server.exclude_tools)
    return True


def classify_mcp_tool_read_only(raw_tool: Mapping[str, Any], server: MCPServerConfig) -> bool:
    """Fail-Closed 只读判定：默认视为写工具（False），仅显式命中可信只读条件时返回 True。"""
    if server.read_only is True:
        return True
    tool_name = raw_tool.get("name")
    if isinstance(tool_name, str) and tool_name in server.read_only_tools:
        return True
    if server.access != "public" and server.trust_annotations is True:
        annotations = raw_tool.get("annotations")
        if isinstance(annotations, Mapping) and annotations.get("readOnlyHint") is True:
            return True
    return False


_SCHEMA_MAP_KEYWORDS = ("properties", "patternProperties", "$defs", "definitions", "dependentSchemas")


def _rewrite_definitions_and_refs(node: Any) -> Any:
    """将 JSON Schema 元关键字 definitions 转为 $defs，不触碰 properties 内的真实属性名。"""
    if isinstance(node, list):
        return [_rewrite_definitions_and_refs(item) for item in node]
    if not isinstance(node, Mapping):
        return node

    out: dict[str, Any] = {}
    for key, val in node.items():
        if key in ("properties", "patternProperties") and isinstance(val, Mapping):
            out[key] = {prop: _rewrite_definitions_and_refs(sub) for prop, sub in val.items()}
        elif key in ("$defs", "definitions") and isinstance(val, Mapping):
            target_defs = out.setdefault("$defs", {})
            for def_name, def_schema in val.items():
                target_defs[def_name] = _rewrite_definitions_and_refs(def_schema)
        else:
            out[key] = _rewrite_definitions_and_refs(val)

    ref = out.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/definitions/"):
        out["$ref"] = "#/$defs/" + ref[len("#/definitions/") :]
    return out


def _repair_schema_node(node: Any, *, is_root: bool = False) -> Any:
    """补齐 object 节点的 type/properties 并清理悬空 required。"""
    if isinstance(node, list):
        return [_repair_schema_node(item, is_root=False) for item in node]
    if not isinstance(node, Mapping):
        return node

    repaired: dict[str, Any] = {}
    for key, val in node.items():
        if key in _SCHEMA_MAP_KEYWORDS and isinstance(val, Mapping):
            repaired[key] = {k: _repair_schema_node(v, is_root=False) for k, v in val.items()}
        else:
            repaired[key] = _repair_schema_node(val, is_root=False)

    if is_root:
        repaired["type"] = "object"
        props = repaired.get("properties")
        if not isinstance(props, dict):
            repaired["properties"] = {}
    elif "properties" in repaired and isinstance(repaired.get("properties"), dict) and not repaired.get("type"):
        repaired["type"] = "object"

    if repaired.get("type") == "object":
        props = repaired.get("properties")
        if not isinstance(props, dict):
            props = {}
            repaired["properties"] = props
        raw_req = repaired.get("required")
        if isinstance(raw_req, (list, tuple)):
            seen_req: set[str] = set()
            clean_req: list[str] = []
            for item in raw_req:
                if isinstance(item, str) and item in props and item not in seen_req:
                    seen_req.add(item)
                    clean_req.append(item)
            repaired["required"] = clean_req
        elif "required" in repaired:
            repaired["required"] = []

    return repaired


def normalize_mcp_schema(
    raw_schema: Any,
    *,
    reserved_params: AbstractSet[str] = RESERVED_PARAMETERS,
) -> tuple[dict[str, Any], dict[str, str]]:
    """归一化 MCP 工具 inputSchema，并将与保留上下文字段冲突的属性重命名为 mcp_arg_<原名>。

    返回:
        (normalized_schema, rename_map)，其中 rename_map 为 {暴露参数名: 原始远端参数名}。
    """
    if not isinstance(raw_schema, Mapping):
        return {"type": "object", "properties": {}}, {}

    rewritten = _rewrite_definitions_and_refs(raw_schema)
    normalized = _repair_schema_node(rewritten, is_root=True)
    if not isinstance(normalized, dict):
        return {"type": "object", "properties": {}}, {}

    orig_props: dict[str, Any] = normalized.get("properties") or {}
    rename_map: dict[str, str] = {}
    orig_to_exposed: dict[str, str] = {}
    new_props: dict[str, Any] = {}
    occupied = set(orig_props.keys())

    for prop_name, prop_schema in orig_props.items():
        if not isinstance(prop_name, str):
            continue
        if prop_name in reserved_params:
            candidate = f"mcp_arg_{prop_name}"
            suffix = 2
            while candidate in occupied or candidate in new_props or candidate in reserved_params:
                candidate = f"mcp_arg_{suffix}_{prop_name}"
                suffix += 1
            new_props[candidate] = prop_schema
            rename_map[candidate] = prop_name
            orig_to_exposed[prop_name] = candidate
        else:
            new_props[prop_name] = prop_schema

    normalized["properties"] = new_props
    raw_required = rewritten.get("required") if isinstance(rewritten, Mapping) else None
    if isinstance(raw_required, (list, tuple)):
        clean_required: list[str] = []
        seen: set[str] = set()
        for req_name in raw_required:
            if not isinstance(req_name, str):
                continue
            mapped = orig_to_exposed.get(req_name, req_name)
            if mapped in new_props and mapped not in seen:
                seen.add(mapped)
                clean_required.append(mapped)
        normalized["required"] = clean_required
    elif "required" in normalized:
        normalized.pop("required", None)

    return json.loads(json.dumps(normalized, ensure_ascii=False)), rename_map


def restore_mcp_arguments(
    arguments: Mapping[str, Any] | None,
    rename_map: Mapping[str, str],
    *,
    reserved_params: AbstractSet[str] = RESERVED_PARAMETERS,
) -> dict[str, Any]:
    """将模型传入的参数还原为远端 MCP 参数名，同时剥离任何未重命名的保留上下文字段。"""
    if not isinstance(arguments, Mapping):
        return {}
    restored: dict[str, Any] = {}
    for key, val in arguments.items():
        if key in rename_map:
            restored[rename_map[key]] = val
        elif key not in reserved_params:
            restored[key] = val
    return restored


# ---------------------------------------------------------------- 连续失败熔断器

class CircuitBreaker:
    """连续 N 次失败后进入冷却期快速返回，冷却期满半开探测一次成功即恢复。"""

    def __init__(
        self,
        *,
        failure_threshold: int = 3,
        cooldown_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold 必须 >= 1")
        if cooldown_seconds <= 0:
            raise ValueError("cooldown_seconds 必须 > 0")
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = float(cooldown_seconds)
        self._clock = clock
        self._lock = threading.Lock()
        self._consecutive_failures = 0
        self._opened_until = 0.0
        self._half_open = False

    @property
    def consecutive_failures(self) -> int:
        with self._lock:
            return self._consecutive_failures

    @property
    def is_open(self) -> bool:
        with self._lock:
            return self._opened_until > self._clock()

    def allow_request(self) -> bool:
        with self._lock:
            now = self._clock()
            if self._opened_until > now:
                return False
            if self._opened_until > 0.0:
                self._half_open = True
            return True

    def record_success(self) -> None:
        with self._lock:
            self._consecutive_failures = 0
            self._opened_until = 0.0
            self._half_open = False

    def record_failure(self) -> None:
        with self._lock:
            self._consecutive_failures += 1
            if self._half_open or self._consecutive_failures >= self.failure_threshold:
                self._opened_until = self._clock() + self.cooldown_seconds
                self._half_open = False

    def reset(self) -> None:
        with self._lock:
            self._consecutive_failures = 0
            self._opened_until = 0.0
            self._half_open = False


# ---------------------------------------------------------------- HTTP 安全重定向与 Streamable HTTP 传输

class _SafeRedirectHandler(HTTPRedirectHandler):
    """校验重定向目标协议安全并在跨主机重定向时剥离敏感认证头。"""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        resolved_target = urljoin(req.full_url, newurl)
        try:
            validate_endpoint_url(resolved_target, field_name="redirect_url")
        except MCPConfigError as exc:
            raise URLError(str(exc)) from exc
        redirected = super().redirect_request(req, fp, code, msg, headers, resolved_target)
        if redirected is None:
            return None
        orig_host = (urlsplit(req.full_url).hostname or "").lower()
        new_host = (urlsplit(resolved_target).hostname or "").lower()
        if orig_host != new_host:
            for header_Map in (redirected.headers, redirected.unredirected_hdrs):
                for key in list(header_Map.keys()):
                    if key.lower() in ("authorization", "x-subscription-token", "x-api-key", "mcp-session-id"):
                        header_Map.pop(key, None)
        return redirected


def build_safe_http_opener():
    return build_opener(_SafeRedirectHandler())


def _extract_header_ci(headers: Any, target_name: str) -> str | None:
    if headers is None:
        return None
    target_lower = target_name.lower()
    if hasattr(headers, "items"):
        for k, v in headers.items():
            if isinstance(k, str) and k.lower() == target_lower and isinstance(v, str) and v.strip():
                return v.strip()
    if hasattr(headers, "get"):
        val = headers.get(target_name) or headers.get(target_lower)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def _match_jsonrpc_response(parsed: Any, req_id: int) -> tuple[bool, Any]:
    """在单个或批量 JSON-RPC 载荷中查找匹配 req_id 的响应。"""
    candidates = parsed if isinstance(parsed, list) else [parsed]
    for item in candidates:
        if not isinstance(item, Mapping):
            continue
        if item.get("id") != req_id or "method" in item:
            continue
        if "error" in item and item["error"] is not None:
            err = item["error"]
            if isinstance(err, Mapping):
                code = err.get("code", "")
                msg = sanitize_external_text(str(err.get("message") or "远端返回 JSON-RPC 错误"))
                raise MCPError(f"MCP 远端错误 ({code}): {msg}")
            raise MCPError(f"MCP 远端错误: {sanitize_external_text(str(err))}")
        return True, item.get("result")
    return False, None


class StreamableHTTPMCPTransport:
    """基于 urllib.request 的 MCP Streamable HTTP 同步传输层。"""

    def __init__(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        timeout: float = 10.0,
        opener: Any = None,
        max_response_bytes: int = MAX_HTTP_RESPONSE_BYTES,
    ) -> None:
        self.url = validate_endpoint_url(url, field_name="url")
        self.timeout = float(timeout)
        self.max_response_bytes = int(max_response_bytes)
        self._headers = {
            str(k): str(v)
            for k, v in (headers or {}).items()
            if isinstance(v, str) and v != ""
        }
        self._opener = opener if opener is not None else build_safe_http_opener()
        self.session_id: str | None = None
        self._closed = False
        self._next_id = 1
        self._id_lock = threading.Lock()

    def is_alive(self) -> bool:
        return not self._closed

    def _alloc_id(self) -> int:
        with self._id_lock:
            req_id = self._next_id
            self._next_id += 1
            return req_id

    def _build_headers(self, *, has_body: bool = True) -> dict[str, str]:
        hdrs = dict(self._headers)
        if has_body:
            hdrs["Content-Type"] = "application/json"
            hdrs["Accept"] = "application/json, text/event-stream"
        if self.session_id:
            hdrs["Mcp-Session-Id"] = self.session_id
        return hdrs

    def _capture_session_id(self, resp: Any) -> None:
        sid = _extract_header_ci(getattr(resp, "headers", None), "Mcp-Session-Id")
        if sid:
            self.session_id = sid

    def _read_sse_response(self, resp: Any, req_id: int) -> Any:
        total_bytes = 0
        data_lines: list[str] = []

        def _flush_event() -> tuple[bool, Any]:
            if not data_lines:
                return False, None
            payload_text = "\n".join(data_lines).strip()
            data_lines.clear()
            if not payload_text:
                return False, None
            try:
                parsed = json.loads(payload_text)
            except json.JSONDecodeError:
                return False, None
            return _match_jsonrpc_response(parsed, req_id)

        while True:
            remaining = self.max_response_bytes + 1 - total_bytes
            if remaining <= 0:
                raise MCPError("MCP SSE 响应超过最大字节限制")
            if hasattr(resp, "readline"):
                raw_line = resp.readline(remaining)
            else:
                raw_body = resp.read(self.max_response_bytes + 1)
                if len(raw_body) > self.max_response_bytes:
                    raise MCPError("MCP SSE 响应超过最大字节限制")
                for synthetic in raw_body.splitlines(keepends=True):
                    line_text = synthetic.decode("utf-8", errors="replace").rstrip("\r\n")
                    if line_text == "":
                        matched, result = _flush_event()
                        if matched:
                            return result
                    elif line_text.startswith("data:"):
                        val = line_text[5:]
                        data_lines.append(val[1:] if val.startswith(" ") else val)
                matched, result = _flush_event()
                if matched:
                    return result
                raise MCPError("MCP SSE 流未返回匹配的 JSON-RPC 响应")

            if not raw_line:
                matched, result = _flush_event()
                if matched:
                    return result
                raise MCPError("MCP SSE 流未返回匹配的 JSON-RPC 响应")

            total_bytes += len(raw_line)
            if total_bytes > self.max_response_bytes:
                raise MCPError("MCP SSE 响应超过最大字节限制")

            line_text = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
            if line_text == "":
                matched, result = _flush_event()
                if matched:
                    return result
            elif line_text.startswith(":"):
                continue
            elif line_text.startswith("data:"):
                val = line_text[5:]
                data_lines.append(val[1:] if val.startswith(" ") else val)

    def request(
        self,
        method: str,
        params: Mapping[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> Any:
        if self._closed:
            raise MCPError("MCP HTTP 传输已关闭")
        req_id = self._alloc_id()
        payload: dict[str, Any] = {"jsonrpc": "2.0", "id": req_id, "method": method}
        if params is not None:
            payload["params"] = dict(params)
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = Request(self.url, data=body, headers=self._build_headers(has_body=True), method="POST")
        eff_timeout = self.timeout if timeout is None else max(0.05, float(timeout))

        try:
            with self._opener.open(req, timeout=eff_timeout) as resp:
                self._capture_session_id(resp)
                raw_ct = _extract_header_ci(getattr(resp, "headers", None), "Content-Type") or ""
                content_type = raw_ct.split(";")[0].strip().lower()
                if content_type == "text/event-stream":
                    return self._read_sse_response(resp, req_id)

                first_chunk = resp.read(self.max_response_bytes)
                extra_byte = resp.read(1)
                if extra_byte:
                    raise MCPError("MCP HTTP 响应超过最大字节限制")

                if content_type and content_type not in ("application/json", "application/json-rpc"):
                    raise MCPError(f"不支持的 MCP 响应 Content-Type: {sanitize_external_text(content_type)}")

                text = first_chunk.decode("utf-8", errors="replace").strip()
                if not text:
                    raise MCPError("MCP HTTP 响应为空")
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise MCPError("MCP HTTP 响应不是合法 JSON") from exc

                matched, result = _match_jsonrpc_response(parsed, req_id)
                if not matched:
                    raise MCPError("MCP HTTP 响应缺少匹配的请求 id")
                return result
        except MCPError:
            raise
        except HTTPError as exc:
            raise MCPError(f"MCP HTTP 请求失败 (HTTP {exc.code})") from None
        except TimeoutError as exc:
            raise MCPError("MCP HTTP 请求超时") from exc
        except URLError as exc:
            reason = exc.reason
            if isinstance(reason, TimeoutError):
                raise MCPError("MCP HTTP 请求超时") from None
            raise MCPError(f"MCP HTTP 连接失败: {sanitize_external_text(type(reason).__name__)}") from None
        except OSError as exc:
            raise MCPError(f"MCP HTTP 网络错误: {type(exc).__name__}") from None

    def notify(
        self,
        method: str,
        params: Mapping[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> None:
        if self._closed:
            raise MCPError("MCP HTTP 传输已关闭")
        payload: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = dict(params)
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = Request(self.url, data=body, headers=self._build_headers(has_body=True), method="POST")
        eff_timeout = self.timeout if timeout is None else max(0.05, float(timeout))
        try:
            with self._opener.open(req, timeout=eff_timeout) as resp:
                self._capture_session_id(resp)
                resp.read(min(4096, self.max_response_bytes))
        except HTTPError as exc:
            if 200 <= exc.code < 300:
                return
            raise MCPError(f"MCP HTTP 通知失败 (HTTP {exc.code})") from None
        except TimeoutError as exc:
            raise MCPError("MCP HTTP 通知超时") from exc
        except (URLError, OSError) as exc:
            raise MCPError(f"MCP HTTP 通知连接失败: {type(exc).__name__}") from None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        sid = self.session_id
        self.session_id = None
        if not sid:
            return
        hdrs = dict(self._headers)
        hdrs["Mcp-Session-Id"] = sid
        req = Request(self.url, headers=hdrs, method="DELETE")
        try:
            with self._opener.open(req, timeout=min(self.timeout, 2.0)) as resp:
                resp.read(1024)
        except Exception:  # noqa: BLE001 - 关闭会话尽力而为，不抛异常阻塞退出
            pass


# ---------------------------------------------------------------- Win32 Job Object 与 Stdio 进程树回收

class _IO_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_uint64),
        ("WriteOperationCount", ctypes.c_uint64),
        ("OtherOperationCount", ctypes.c_uint64),
        ("ReadTransferCount", ctypes.c_uint64),
        ("WriteTransferCount", ctypes.c_uint64),
        ("OtherTransferCount", ctypes.c_uint64),
    ]


class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", ctypes.c_uint32),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", ctypes.c_uint32),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", ctypes.c_uint32),
        ("SchedulingClass", ctypes.c_uint32),
    ]


class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", _IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


_JOB_OBJECT_EXTENDED_LIMIT_INFO_CLASS = 9
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_PROCESS_SET_QUOTA = 0x0100
_PROCESS_TERMINATE = 0x0001


def _is_windows() -> bool:
    return os.name == "nt"


class _Win32JobHandle:
    """Windows Job Object 包装器：绑定 KILL_ON_JOB_CLOSE 并在关闭时终止整棵子进程树。"""

    def __init__(self, kernel32: Any, handle: int) -> None:
        self._kernel32 = kernel32
        self.handle: int | None = handle

    @classmethod
    def create_and_assign(cls, proc: subprocess.Popen[Any]) -> _Win32JobHandle | None:
        if not _is_windows() or not hasattr(ctypes, "windll"):
            return None
        try:
            k32 = ctypes.windll.kernel32
            k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
            k32.CreateJobObjectW.restype = ctypes.c_void_p
            k32.SetInformationJobObject.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.c_uint32,
            ]
            k32.SetInformationJobObject.restype = ctypes.c_int
            k32.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            k32.AssignProcessToJobObject.restype = ctypes.c_int
            k32.TerminateJobObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
            k32.TerminateJobObject.restype = ctypes.c_int
            k32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
            k32.OpenProcess.restype = ctypes.c_void_p
            k32.CloseHandle.argtypes = [ctypes.c_void_p]
            k32.CloseHandle.restype = ctypes.c_int

            job = k32.CreateJobObjectW(None, None)
            if not job:
                return None

            info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
            info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            ok = k32.SetInformationJobObject(
                job,
                _JOB_OBJECT_EXTENDED_LIMIT_INFO_CLASS,
                ctypes.byref(info),
                ctypes.sizeof(info),
            )
            if not ok:
                k32.CloseHandle(job)
                return None

            raw_proc_handle = getattr(proc, "_handle", None)
            opened_handle = None
            if raw_proc_handle is not None:
                proc_handle = int(raw_proc_handle)
            else:
                opened_handle = k32.OpenProcess(_PROCESS_SET_QUOTA | _PROCESS_TERMINATE, 0, int(proc.pid))
                proc_handle = opened_handle

            try:
                if not proc_handle or not k32.AssignProcessToJobObject(job, proc_handle):
                    k32.CloseHandle(job)
                    return None
            finally:
                if opened_handle:
                    k32.CloseHandle(opened_handle)

            return cls(k32, int(job))
        except Exception as exc:  # noqa: BLE001
            logger.debug("绑定 Windows Job Object 失败，将回退到进程树终止: %s", type(exc).__name__)
            return None

    def terminate_and_close(self) -> None:
        handle = self.handle
        if not handle:
            return
        self.handle = None
        try:
            self._kernel32.TerminateJobObject(handle, 1)
        except Exception:  # noqa: BLE001
            pass
        try:
            self._kernel32.CloseHandle(handle)
        except Exception:  # noqa: BLE001
            pass


def build_safe_stdio_env(
    server_env: Mapping[str, str] | None = None,
    *,
    base_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """构建最小安全环境变量白名单，严禁把主进程中的无关密钥全量传给子进程。"""
    source = os.environ if base_env is None else base_env
    safe_env: dict[str, str] = {}
    for key, val in source.items():
        if isinstance(key, str) and isinstance(val, str) and key.upper() in SAFE_STDIO_ENV_KEYS and "\x00" not in val:
            safe_env[key] = val
    safe_env.setdefault("PYTHONIOENCODING", "utf-8")
    safe_env.setdefault("PYTHONUTF8", "1")
    for key, val in (server_env or {}).items():
        if isinstance(key, str) and isinstance(val, str) and "\x00" not in key and "\x00" not in val:
            safe_env[key] = val
    return safe_env


def _posix_pgroup_alive(pgid: int | None) -> bool:
    if not pgid or pgid <= 0 or not hasattr(os, "killpg"):
        return False
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def _run_windows_taskkill(pid: int, timeout: float) -> None:
    if pid <= 0 or timeout <= 0:
        return
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    try:
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(int(pid))],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=max(0.2, timeout),
            check=False,
            creationflags=flags if _is_windows() else 0,
        )
    except Exception:  # noqa: BLE001
        pass


class StdioMCPTransport:
    """基于 subprocess 按行管道的 MCP stdio 传输层，带安全环境隔离与整棵进程树清理。"""

    def __init__(
        self,
        command: str,
        args: tuple[str, ...] | list[str] = (),
        *,
        env: Mapping[str, str] | None = None,
        cwd: str | None = None,
        log_dir: Path,
        timeout: float = 10.0,
        base_env: Mapping[str, str] | None = None,
        popen_factory: Callable[..., subprocess.Popen[Any]] = subprocess.Popen,
    ) -> None:
        if not isinstance(command, str) or not command.strip():
            raise MCPConfigError("stdio command 不能为空")
        self.timeout = float(timeout)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.stderr_log_path = self.log_dir / "mcp-stderr.log"
        if not self.stderr_log_path.exists():
            self.stderr_log_path.touch()

        child_env = build_safe_stdio_env(env, base_env=base_env)
        is_win = _is_windows()
        executable_cmd = command.strip()
        if is_win:
            resolved = shutil.which(executable_cmd, path=child_env.get("PATH") or child_env.get("Path"))
            if resolved:
                executable_cmd = resolved
        argv = [executable_cmd, *(str(a) for a in args)]

        popen_kwargs: dict[str, Any] = {
            "stdin": subprocess.PIPE,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "env": child_env,
            "cwd": cwd,
            "bufsize": 0,
        }
        if is_win:
            no_win = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
            new_grp = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
            popen_kwargs["creationflags"] = no_win | new_grp
        else:
            popen_kwargs["start_new_session"] = True

        try:
            self._proc = popen_factory(argv, **popen_kwargs)
        except OSError as exc:
            raise MCPError(f"启动 MCP stdio 子进程失败: {type(exc).__name__}") from None

        self._pgid: int | None = (
            int(self._proc.pid)
            if not is_win and isinstance(getattr(self._proc, "pid", None), int) and self._proc.pid > 0
            else None
        )
        self._job: _Win32JobHandle | None = _Win32JobHandle.create_and_assign(self._proc) if is_win else None

        self._closed = False
        self._eof = False
        self._next_id = 1
        self._write_lock = threading.Lock()
        self._pending_lock = threading.Lock()
        self._pending: dict[int, tuple[threading.Event, list[Any]]] = {}

        self._stdout_thread = threading.Thread(
            target=self._stdout_reader_loop,
            name="mcp-stdio-stdout",
            daemon=True,
        )
        self._stderr_thread = threading.Thread(
            target=self._stderr_reader_loop,
            name="mcp-stdio-stderr",
            daemon=True,
        )
        self._stdout_thread.start()
        self._stderr_thread.start()

    @property
    def pid(self) -> int | None:
        return getattr(self._proc, "pid", None)

    def is_alive(self) -> bool:
        return not self._closed and not self._eof and self._proc.poll() is None

    def _read_bounded_line(self, stream: Any, limit: int) -> tuple[bytes, bool]:
        """从字节流按行读取，单行超过 limit 字节时排空至换行并标记过长。"""
        if hasattr(stream, "readline"):
            chunks: list[bytes] = []
            total = 0
            overflow = False
            while True:
                piece = stream.readline(min(65536, max(1, limit + 1 - total)) if not overflow else 65536)
                if not piece:
                    return b"".join(chunks), overflow
                if not overflow:
                    total += len(piece)
                    if total <= limit:
                        chunks.append(piece)
                    else:
                        overflow = True
                        chunks.clear()
                if piece.endswith(b"\n"):
                    return b"".join(chunks), overflow
        return b"", False

    def _stdout_reader_loop(self) -> None:
        stream = getattr(self._proc, "stdout", None)
        if stream is None:
            self._mark_eof()
            return
        try:
            while not self._closed:
                raw_line, overflow = self._read_bounded_line(stream, MAX_STDIO_LINE_BYTES)
                if not raw_line and not overflow:
                    break
                if overflow:
                    self._fail_oldest_or_log("MCP stdio 单行响应超过最大字节限制")
                    continue
                text = raw_line.decode("utf-8", errors="replace").strip()
                if not text:
                    continue
                try:
                    msg = json.loads(text)
                except json.JSONDecodeError:
                    continue
                self._dispatch_incoming(msg)
        except Exception:  # noqa: BLE001
            pass
        finally:
            self._mark_eof()

    def _fail_oldest_or_log(self, message: str) -> None:
        with self._pending_lock:
            if self._pending:
                req_id = next(iter(self._pending))
                event, box = self._pending.pop(req_id)
                box.append(MCPError(message))
                event.set()

    def _dispatch_incoming(self, msg: Any) -> None:
        items = msg if isinstance(msg, list) else [msg]
        for item in items:
            if not isinstance(item, Mapping) or "method" in item or "id" not in item:
                continue
            req_id = item.get("id")
            if not isinstance(req_id, int):
                continue
            with self._pending_lock:
                entry = self._pending.pop(req_id, None)
            if entry is None:
                continue
            event, box = entry
            if "error" in item and item["error"] is not None:
                err = item["error"]
                if isinstance(err, Mapping):
                    code = err.get("code", "")
                    err_msg = sanitize_external_text(str(err.get("message") or "远端返回 JSON-RPC 错误"))
                    box.append(MCPError(f"MCP 远端错误 ({code}): {err_msg}"))
                else:
                    box.append(MCPError(f"MCP 远端错误: {sanitize_external_text(str(err))}"))
            else:
                box.append(item.get("result"))
            event.set()

    def _mark_eof(self) -> None:
        with self._pending_lock:
            self._eof = True
            pending_entries = list(self._pending.values())
            self._pending.clear()
        for event, box in pending_entries:
            if not box:
                box.append(MCPError("MCP stdio 子进程已退出或关闭管道"))
            event.set()

    def _stderr_reader_loop(self) -> None:
        stream = getattr(self._proc, "stderr", None)
        if stream is None:
            return
        try:
            while not self._closed:
                raw_line, _ = self._read_bounded_line(stream, 16384)
                if not raw_line:
                    break
                sanitized = sanitize_external_text(raw_line.decode("utf-8", errors="replace")).rstrip("\n")
                if not sanitized:
                    continue
                try:
                    with self.stderr_log_path.open("a", encoding="utf-8") as fp:
                        fp.write(sanitized + "\n")
                except OSError:
                    pass
        except Exception:  # noqa: BLE001
            pass

    def _write_json_line(self, payload: Mapping[str, Any]) -> None:
        if self._closed or self._eof or self._proc.poll() is not None:
            raise MCPError("MCP stdio 子进程未运行")
        stdin = getattr(self._proc, "stdin", None)
        if stdin is None:
            raise MCPError("MCP stdio stdin 不可用")
        data = (json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        with self._write_lock:
            try:
                stdin.write(data)
                stdin.flush()
            except OSError as exc:
                raise MCPError(f"MCP stdio 写入管道失败: {type(exc).__name__}") from None

    def request(
        self,
        method: str,
        params: Mapping[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> Any:
        if self._closed:
            raise MCPError("MCP stdio 传输已关闭")
        event = threading.Event()
        box: list[Any] = []
        with self._pending_lock:
            if self._eof or self._proc.poll() is not None:
                raise MCPError("MCP stdio 子进程已退出")
            req_id = self._next_id
            self._next_id += 1
            self._pending[req_id] = (event, box)

        payload: dict[str, Any] = {"jsonrpc": "2.0", "id": req_id, "method": method}
        if params is not None:
            payload["params"] = dict(params)

        try:
            self._write_json_line(payload)
        except BaseException:
            with self._pending_lock:
                self._pending.pop(req_id, None)
            raise

        eff_timeout = self.timeout if timeout is None else max(0.05, float(timeout))
        if not event.wait(timeout=eff_timeout):
            with self._pending_lock:
                self._pending.pop(req_id, None)
            raise MCPError("MCP stdio 请求超时")

        if not box:
            raise MCPError("MCP stdio 未收到有效响应")
        outcome = box[0]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def notify(
        self,
        method: str,
        params: Mapping[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> None:
        del timeout
        payload: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = dict(params)
        self._write_json_line(payload)

    def close(self) -> None:
        """受控回收子进程与整棵孙进程树（总耗时严格 <= 2.5s）。"""
        if self._closed:
            return
        self._closed = True
        deadline = time.monotonic() + 2.4

        # 1. 先关 stdin，给规范实现 0.5s 优雅退出窗口
        stdin = getattr(self._proc, "stdin", None)
        if stdin is not None:
            try:
                stdin.close()
            except Exception:  # noqa: BLE001
                pass

        graceful = False
        try:
            self._proc.wait(timeout=0.5)
            graceful = True
        except Exception:  # noqa: BLE001
            graceful = False

        # 2. 无论父进程是否已退出，强制清理可能残留的整棵孙进程树
        if _is_windows():
            had_job = self._job is not None
            if self._job is not None:
                self._job.terminate_and_close()
                self._job = None
            pid = getattr(self._proc, "pid", None)
            if (not graceful or not had_job or self._proc.poll() is None) and isinstance(pid, int) and pid > 0:
                remaining = max(0.2, min(1.2, deadline - time.monotonic()))
                _run_windows_taskkill(pid, remaining)
            if self._proc.poll() is None:
                try:
                    self._proc.kill()
                except Exception:  # noqa: BLE001
                    pass
        else:
            pgid = self._pgid
            if pgid and (_posix_pgroup_alive(pgid) or self._proc.poll() is None):
                try:
                    os.killpg(pgid, signal.SIGTERM)
                except OSError:
                    pass
                term_deadline = min(deadline, time.monotonic() + 0.5)
                while time.monotonic() < term_deadline and _posix_pgroup_alive(pgid):
                    try:
                        self._proc.wait(timeout=0.05)
                    except Exception:  # noqa: BLE001
                        time.sleep(0.03)
                if _posix_pgroup_alive(pgid) or self._proc.poll() is None:
                    try:
                        os.killpg(pgid, signal.SIGKILL)
                    except OSError:
                        pass
            if self._proc.poll() is None:
                try:
                    self._proc.kill()
                except Exception:  # noqa: BLE001
                    pass

        # 3. 回收父进程防僵尸进程，并关闭管道与等待读取线程释放文件句柄
        wait_rem = max(0.05, min(0.5, deadline - time.monotonic()))
        try:
            self._proc.wait(timeout=wait_rem)
        except Exception:  # noqa: BLE001
            pass

        for attr in ("stdout", "stderr"):
            pipe = getattr(self._proc, attr, None)
            if pipe is not None:
                try:
                    pipe.close()
                except Exception:  # noqa: BLE001
                    pass

        self._mark_eof()
        for worker in (self._stdout_thread, self._stderr_thread):
            join_rem = max(0.02, min(0.2, deadline - time.monotonic()))
            if worker.is_alive():
                worker.join(timeout=join_rem)


# ---------------------------------------------------------------- MCP 工具描述符与 MCPManager

@dataclasses.dataclass(frozen=True)
class MCPToolDescriptor:
    name: str
    server_name: str
    original_name: str
    description: str
    parameters_schema: dict[str, Any]
    rename_map: Mapping[str, str]
    read_only: bool
    write: bool
    privileged_only: bool
    access: str
    allow_in_group: bool
    timeout_seconds: float
    max_result_chars: int


class _ServerRuntimeState:
    def __init__(self, config: MCPServerConfig, clock: Callable[[], float]) -> None:
        self.config = config
        self.breaker = CircuitBreaker(failure_threshold=3, cooldown_seconds=60.0, clock=clock)
        self.lock = threading.RLock()
        self.transport: Any = None
        self.tools: tuple[MCPToolDescriptor, ...] | None = None
        self.tool_by_name: dict[str, MCPToolDescriptor] = {}


def _extract_mcp_call_text(result: Any) -> str:
    """从 MCP tools/call 返回结构中提取文本内容，剥离二进制资源并限制协议层长度上限。"""
    if not isinstance(result, Mapping):
        raw = json.dumps(result, ensure_ascii=False, default=str)
        return raw[:MAX_RAW_TOOL_TEXT_CHARS]

    parts: list[str] = []
    total_len = 0
    content = result.get("content")
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, Mapping):
                continue
            btype = block.get("type")
            piece = ""
            if btype == "text" and isinstance(block.get("text"), str):
                piece = block["text"]
            elif btype == "image":
                piece = "[图片]"
            elif btype == "audio":
                piece = "[音频]"
            elif btype == "resource":
                res = block.get("resource")
                if isinstance(res, Mapping):
                    if isinstance(res.get("text"), str):
                        piece = res["text"]
                    elif "blob" in res:
                        piece = "[二进制资源已省略]"
            if piece:
                remaining = MAX_RAW_TOOL_TEXT_CHARS - total_len
                if remaining <= 0:
                    break
                clipped = piece[:remaining]
                parts.append(clipped)
                total_len += len(clipped)

    if not parts and "structuredContent" in result and result["structuredContent"] is not None:
        raw_struct = json.dumps(result["structuredContent"], ensure_ascii=False, default=str)
        parts.append(raw_struct[:MAX_RAW_TOOL_TEXT_CHARS])

    return "\n".join(parts)


class MCPManager:
    """MCP 客户端与内置 WebSearch 统一生命周期管理器。

    初始化阶段零网络请求、零子进程创建；首次请求工具清单或调用工具时按需懒连接。
    """

    def __init__(
        self,
        config: MCPConfig,
        *,
        log_dir: Path,
        clock: Callable[[], float] = time.monotonic,
        http_opener: Any = None,
        env: Mapping[str, str] | None = None,
        transport_factory: Callable[[MCPServerConfig], Any] | None = None,
    ) -> None:
        if not isinstance(config, MCPConfig):
            raise MCPConfigError("config 必须是 MCPConfig 实例")
        self.log_dir = Path(log_dir)
        self.clock = clock
        self._http_opener = http_opener
        self._env = env
        self._transport_factory = transport_factory
        self._lock = threading.RLock()
        self._closed = False
        self.config: MCPConfig = MCPConfig(enabled=False)
        self._states: dict[str, _ServerRuntimeState] = {}
        self._web_search_client: Any = None
        self._apply_config_locked(config)

    def _apply_config_locked(self, config: MCPConfig) -> None:
        validated = config.validate_active(self._env)
        self.config = validated
        self._closed = False
        self._states = {
            name: _ServerRuntimeState(srv, self.clock)
            for name, srv in validated.mcp_servers.items()
            if validated.enabled and srv.enabled
        }
        if validated.enabled and validated.web_search.enabled:
            from .web_search import WebSearchClient

            self._web_search_client = WebSearchClient(validated.web_search, opener=self._http_opener, env=self._env)
        else:
            self._web_search_client = None

    @property
    def web_search_enabled(self) -> bool:
        with self._lock:
            return bool(not self._closed and self.config.enabled and self.config.web_search.enabled)

    @property
    def web_search_config(self) -> WebSearchConfig:
        with self._lock:
            return self.config.web_search

    def _create_transport(self, server_cfg: MCPServerConfig) -> Any:
        if self._transport_factory is not None:
            return self._transport_factory(server_cfg)
        resolved = server_cfg.resolve(self._env)
        if resolved.transport == "streamable_http":
            return StreamableHTTPMCPTransport(
                resolved.url,
                headers=resolved.headers,
                timeout=resolved.timeout_seconds,
                opener=self._http_opener,
            )
        return StdioMCPTransport(
            resolved.command,
            resolved.args,
            env=resolved.env,
            cwd=resolved.cwd,
            log_dir=self.log_dir,
            timeout=resolved.timeout_seconds,
            base_env=self._env,
        )

    def _connect_and_handshake_locked(self, state: _ServerRuntimeState) -> Any:
        if state.transport is not None and state.transport.is_alive():
            return state.transport
        if state.transport is not None:
            try:
                state.transport.close()
            except Exception:  # noqa: BLE001
                pass
            state.transport = None

        transport = self._create_transport(state.config)
        try:
            init_res = transport.request(
                "initialize",
                {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "wxbot-helper", "version": "1.0.0"},
                },
                timeout=state.config.timeout_seconds,
            )
            if not isinstance(init_res, Mapping):
                raise MCPError("MCP initialize 响应格式无效")
            transport.notify("notifications/initialized", {}, timeout=state.config.timeout_seconds)
        except BaseException:
            try:
                transport.close()
            except Exception:  # noqa: BLE001
                pass
            raise

        state.transport = transport
        return transport

    def _discover_server_tools_locked(
        self,
        state: _ServerRuntimeState,
        occupied_names: set[str],
    ) -> tuple[MCPToolDescriptor, ...]:
        if state.tools is not None:
            for desc in state.tools:
                occupied_names.add(desc.name)
            return state.tools

        if not state.breaker.allow_request():
            return ()

        srv = state.config
        try:
            transport = self._connect_and_handshake_locked(state)
            raw_tools: list[Mapping[str, Any]] = []
            cursor: str | None = None
            seen_cursors: set[str] = set()
            for _ in range(MAX_TOOLS_LIST_PAGES):
                params = {"cursor": cursor} if cursor else {}
                page = transport.request("tools/list", params, timeout=srv.timeout_seconds)
                if not isinstance(page, Mapping) or not isinstance(page.get("tools"), list):
                    raise MCPError("MCP tools/list 响应格式无效")
                for item in page["tools"]:
                    if isinstance(item, Mapping):
                        raw_tools.append(item)
                next_cursor = page.get("nextCursor")
                if not isinstance(next_cursor, str) or not next_cursor or next_cursor in seen_cursors:
                    break
                seen_cursors.add(next_cursor)
                cursor = next_cursor

            descriptors: list[MCPToolDescriptor] = []
            by_name: dict[str, MCPToolDescriptor] = {}
            for raw_tool in raw_tools:
                orig_name = raw_tool.get("name")
                if not isinstance(orig_name, str) or not orig_name.strip():
                    continue
                orig_name = orig_name.strip()
                if not is_mcp_tool_included(orig_name, srv):
                    continue
                read_only = classify_mcp_tool_read_only(raw_tool, srv)
                if not read_only and not srv.allow_write:
                    continue

                exposed_name = make_mcp_tool_name(srv.name, orig_name)
                if exposed_name in RESERVED_TOOL_NAMES or exposed_name in occupied_names or exposed_name in by_name:
                    logger.warning("跳过冲突或保留的 MCP 工具名: %s", exposed_name)
                    continue

                raw_desc = str(raw_tool.get("description") or "")
                for pattern, label in _DESCRIPTION_INJECTION_PATTERNS:
                    if pattern.search(raw_desc):
                        logger.warning(
                            "MCP Server %s 工具 %s 描述命中可疑提示词特征 (%s)",
                            srv.name,
                            orig_name,
                            label,
                        )
                        break

                clean_desc = sanitize_external_text(raw_desc).strip()
                if not clean_desc:
                    clean_desc = f"来自 {sanitize_external_text(srv.name)} 的 MCP 工具 {sanitize_external_text(orig_name)}"
                clean_desc = clean_desc[:1000]

                norm_schema, rename_map = normalize_mcp_schema(
                    raw_tool.get("inputSchema"),
                    reserved_params=RESERVED_PARAMETERS,
                )
                is_write = not read_only
                priv_only = True if is_write else (srv.access != "public")
                desc = MCPToolDescriptor(
                    name=exposed_name,
                    server_name=srv.name,
                    original_name=orig_name,
                    description=clean_desc,
                    parameters_schema=norm_schema,
                    rename_map=MappingProxyType(rename_map),
                    read_only=read_only,
                    write=is_write,
                    privileged_only=priv_only,
                    access=srv.access,
                    allow_in_group=srv.allow_in_group,
                    timeout_seconds=srv.timeout_seconds,
                    max_result_chars=srv.max_result_chars,
                )
                descriptors.append(desc)
                by_name[exposed_name] = desc
                occupied_names.add(exposed_name)

            state.breaker.record_success()
            state.tools = tuple(descriptors)
            state.tool_by_name = by_name
            return state.tools
        except Exception as exc:  # noqa: BLE001
            state.breaker.record_failure()
            if state.transport is not None:
                try:
                    state.transport.close()
                except Exception:  # noqa: BLE001
                    pass
                state.transport = None
            logger.warning(
                "MCP Server %s 懒连接或加载工具失败，已隔离跳过: %s",
                srv.name,
                sanitize_external_text(str(exc)) or type(exc).__name__,
            )
            return ()

    def list_tool_descriptors(self) -> tuple[MCPToolDescriptor, ...]:
        """按需懒连接各启用的 MCP Server 并返回隔离后的工具描述符元组。"""
        with self._lock:
            if self._closed or not self.config.enabled:
                return ()
            states = list(self._states.values())

        occupied: set[str] = set(RESERVED_TOOL_NAMES)
        all_descriptors: list[MCPToolDescriptor] = []
        for state in states:
            with state.lock:
                all_descriptors.extend(self._discover_server_tools_locked(state, occupied))
        return tuple(all_descriptors)

    def _find_tool(self, tool_name: str) -> tuple[_ServerRuntimeState, MCPToolDescriptor] | None:
        with self._lock:
            if self._closed or not self.config.enabled:
                return None
            states = list(self._states.values())

        for state in states:
            with state.lock:
                desc = state.tool_by_name.get(tool_name)
                if desc is not None:
                    return state, desc

        # 若尚未发现工具，触发一次按需发现后再查
        self.list_tool_descriptors()
        for state in states:
            with state.lock:
                desc = state.tool_by_name.get(tool_name)
                if desc is not None:
                    return state, desc
        return None

    def call_tool(
        self,
        tool_name: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """调用指定 MCP 工具，处理参数还原、只读复核、熔断拦截、输出脱敏与长度截断。"""
        found = self._find_tool(tool_name)
        if found is None:
            return {"ok": False, "error": f"未知或未启用的 MCP 工具 {sanitize_external_text(str(tool_name))}"}

        state, desc = found
        srv = state.config
        if desc.write and not srv.allow_write:
            return {"ok": False, "error": "该 MCP 工具未开启写入权限"}

        with state.lock:
            if not state.breaker.allow_request():
                return {"ok": False, "error": "外部工具暂时不可用（服务熔断冷却中），请勿重试"}

            remote_args = restore_mcp_arguments(
                arguments,
                desc.rename_map,
                reserved_params=RESERVED_PARAMETERS,
            )
            eff_timeout = (
                srv.timeout_seconds
                if timeout is None
                else max(0.05, min(float(timeout), srv.timeout_seconds))
            )

            try:
                transport = self._connect_and_handshake_locked(state)
                raw_res = transport.request(
                    "tools/call",
                    {"name": desc.original_name, "arguments": remote_args},
                    timeout=eff_timeout,
                )
            except Exception as exc:  # noqa: BLE001
                state.breaker.record_failure()
                if state.transport is not None and not state.transport.is_alive():
                    try:
                        state.transport.close()
                    except Exception:  # noqa: BLE001
                        pass
                    state.transport = None
                logger.warning(
                    "MCP 工具 %s 调用失败: %s",
                    desc.name,
                    sanitize_external_text(str(exc)) or type(exc).__name__,
                )
                return {"ok": False, "error": "外部工具暂时不可用或超时，请勿重试"}

            state.breaker.record_success()

        raw_text = _extract_mcp_call_text(raw_res)
        sanitized = sanitize_external_text(raw_text)
        truncated = len(sanitized) > desc.max_result_chars
        if truncated:
            sanitized = sanitized[: desc.max_result_chars]

        is_error = isinstance(raw_res, Mapping) and raw_res.get("isError") is True
        if is_error:
            payload: dict[str, Any] = {"ok": False, "error": sanitized or "MCP 工具执行返回错误"}
        else:
            payload = {"ok": True, "data": sanitized}
        if truncated:
            payload["truncated"] = True
        return payload

    def search_web(
        self,
        query: str,
        *,
        limit: int | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """调用内置只读网页搜索客户端。"""
        with self._lock:
            client = self._web_search_client if not self._closed and self.web_search_enabled else None
        if client is None:
            return {"ok": False, "error": "网页搜索未启用"}
        return client.search(query, limit=limit, timeout=timeout)

    def _close_states(self, states: list[_ServerRuntimeState]) -> None:
        for state in states:
            with state.lock:
                transport = state.transport
                state.transport = None
                if transport is not None:
                    try:
                        transport.close()
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("关闭 MCP Server %s 传输时忽略异常: %s", state.config.name, type(exc).__name__)

    def reload(self, new_config: MCPConfig) -> None:
        """先安全关闭旧 MCP 会话与子进程树，再加载新配置（零子进程立即启动）。"""
        if not isinstance(new_config, MCPConfig):
            raise MCPConfigError("new_config 必须是 MCPConfig 实例")
        validated = new_config.validate_active(self._env)
        with self._lock:
            old_states = list(self._states.values())
            self._states = {}
            self._web_search_client = None
        self._close_states(old_states)
        with self._lock:
            self._apply_config_locked(validated)

    def close(self) -> None:
        """关闭所有活动 MCP 连接与子进程树；幂等且绝不向外抛异常阻塞退出。"""
        with self._lock:
            if self._closed and not self._states:
                return
            self._closed = True
            states = list(self._states.values())
            self._states = {}
            self._web_search_client = None
        self._close_states(states)
