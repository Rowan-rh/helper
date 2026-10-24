"""标准库 MCP 客户端、配置校验、熔断器、HTTP/SSE 传输与 Stdio 进程树回收单元测试。"""
from __future__ import annotations

import ctypes
import io
import json
import logging
import os
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from unittest import mock

from wxbot_framework.mcp_client import (
    CircuitBreaker,
    MCPConfig,
    MCPConfigError,
    MCPError,
    MCPManager,
    MCPServerConfig,
    RESERVED_TOOL_NAMES,
    StdioMCPTransport,
    StreamableHTTPMCPTransport,
    WebSearchConfig,
    _JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
    build_safe_stdio_env,
    classify_mcp_tool_read_only,
    is_mcp_tool_included,
    make_mcp_tool_name,
    normalize_mcp_schema,
    resolve_env_placeholders,
    restore_mcp_arguments,
    sanitize_external_text,
    validate_endpoint_url,
)


class _FakeClock:
    def __init__(self, initial: float = 100.0) -> None:
        self.now = initial

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _MockHTTPResponse:
    def __init__(
        self,
        body: bytes,
        *,
        status: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._buf = io.BytesIO(body)
        self.status = status
        self.headers = headers or {"Content-Type": "application/json"}

    def read(self, size: int | None = None) -> bytes:
        return self._buf.read() if size is None else self._buf.read(size)

    def readline(self, size: int = -1) -> bytes:
        return self._buf.readline(size)

    def __enter__(self) -> _MockHTTPResponse:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class _MockOpener:
    def __init__(self, handler) -> None:
        self._handler = handler
        self.calls: list[tuple[object, float | None]] = []

    def open(self, req, timeout: float | None = None):
        self.calls.append((req, timeout))
        return self._handler(req, timeout)


def _pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def _wait_pid_gone(pid: int, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _pid_exists(pid):
            return True
        time.sleep(0.03)
    return not _pid_exists(pid)


# 测试用本地轻量 Python stdio MCP Server 脚本（派生真实孙进程以验证整树回收）
_STDIO_MOCK_SERVER_SCRIPT = textwrap.dedent(
    """
    import json
    import os
    import subprocess
    import sys
    import time

    grandchild = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    sys.stderr.write("boot stderr token=Bearer secret_stderr_token_abcdef12 user=wxid_user_bob\\n")
    sys.stderr.flush()

    for raw in sys.stdin.buffer:
        line = raw.decode("utf-8", errors="replace").strip()
        if not line:
            continue
        msg = json.loads(line)
        method = msg.get("method")
        req_id = msg.get("id")
        if req_id is None:
            continue
        params = msg.get("params") or {}
        if method == "initialize":
            res = {
                "protocolVersion": "2025-03-26",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "mock-stdio", "version": "0.1"},
            }
        elif method == "tools/list":
            if params.get("cursor") == "page2":
                res = {
                    "tools": [
                        {
                            "name": "slow_op",
                            "description": "慢速测试工具",
                            "inputSchema": {"type": "object", "properties": {}},
                        }
                    ]
                }
            else:
                res = {
                    "tools": [
                        {
                            "name": "inspect_env",
                            "description": "检查子进程环境与孙进程 PID",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "room": {"type": "string"},
                                    "note": {"type": "string"},
                                },
                                "required": ["room"],
                            },
                        }
                    ],
                    "nextCursor": "page2",
                }
        elif method == "tools/call":
            tname = params.get("name")
            targs = params.get("arguments") or {}
            if tname == "slow_op":
                time.sleep(5.0)
                res = {"content": [{"type": "text", "text": "done"}]}
            else:
                payload = {
                    "grandchild_pid": grandchild.pid,
                    "has_host_secret": "HOST_ONLY_SECRET_KEY" in os.environ,
                    "custom_flag": os.environ.get("MCP_CUSTOM_FLAG", ""),
                    "received_args": targs,
                }
                res = {"content": [{"type": "text", "text": json.dumps(payload)}]}
        else:
            res = {}
        out = json.dumps({"jsonrpc": "2.0", "id": req_id, "result": res}) + "\\n"
        sys.stdout.buffer.write(out.encode("utf-8"))
        sys.stdout.buffer.flush()
    """
)


class TestSecurityAndConfig(unittest.TestCase):
    def test_reserved_tool_names(self) -> None:
        expected = {
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
        }
        self.assertEqual(RESERVED_TOOL_NAMES, expected)

    def test_validate_endpoint_url(self) -> None:
        self.assertEqual(validate_endpoint_url("https://mcp.example.test/mcp"), "https://mcp.example.test/mcp")
        self.assertEqual(validate_endpoint_url("http://127.0.0.1:8080/mcp"), "http://127.0.0.1:8080/mcp")
        self.assertEqual(validate_endpoint_url("http://localhost:3000/mcp"), "http://localhost:3000/mcp")
        self.assertEqual(validate_endpoint_url("http://[::1]:9000/mcp"), "http://[::1]:9000/mcp")

        for bad_url in (
            "",
            "   ",
            "http://mcp.example.test/mcp",
            "ftp://mcp.example.test/mcp",
            "https://user:pass@mcp.example.test/mcp",
            "https://token@mcp.example.test/mcp",
            "https://mcp.example.test/with space",
        ):
            with self.subTest(bad_url=bad_url), self.assertRaises(MCPConfigError):
                validate_endpoint_url(bad_url)

    def test_resolve_env_placeholders_never_leaks_secret_values(self) -> None:
        secret_val = "top_secret_env_token_xyz987"
        env = {"VALID_TOKEN": secret_val, "EMPTY_TOKEN": ""}
        self.assertEqual(
            resolve_env_placeholders("Bearer ${VALID_TOKEN}", env, strict=True),
            f"Bearer {secret_val}",
        )
        with self.assertRaises(MCPConfigError) as ctx1:
            resolve_env_placeholders("Bearer ${MISSING_VAR}", env, strict=True)
        self.assertIn("MISSING_VAR", str(ctx1.exception))
        self.assertNotIn(secret_val, str(ctx1.exception))

        with self.assertRaises(MCPConfigError) as ctx2:
            resolve_env_placeholders("${VALID_TOKEN}:${EMPTY_TOKEN}", env, strict=True)
        self.assertIn("EMPTY_TOKEN", str(ctx2.exception))
        self.assertNotIn(secret_val, str(ctx2.exception))

        with self.assertRaises(MCPConfigError):
            resolve_env_placeholders("Bearer ${UNCLOSED", env, strict=True)

        self.assertEqual(resolve_env_placeholders("prefix-${MISSING_VAR}", env, strict=False), "prefix-")

    def test_sanitize_external_text(self) -> None:
        fake_sk = "sk-" + "k" * 16
        fake_win_path = "D:" + "\\Users\\alice_dev\\AppData\\Local\\secret.json"
        unicode_tags = "".join(chr(0xE0000 + c) for c in b"IGNORE")
        raw = (
            "前文 data:image/png;base64,iVBORw0KGgoAAAANSUhEUg== and English tail\n"
            f"控制\x00\x07\t保留换行{unicode_tags} "
            f"账号 wxid_vip_alice 群 90000002@chatroom "
            f"凭据 Bearer my_super_secret_token_12345 密钥 {fake_sk} "
            f"路径 {fake_win_path}。"
        )
        cleaned = sanitize_external_text(raw)
        self.assertIn("前文 [图片] and English tail\n", cleaned)
        self.assertIn("控制\t保留换行 ", cleaned)
        self.assertNotIn("\x00", cleaned)
        self.assertNotIn("\x07", cleaned)
        self.assertIn("[微信ID]", cleaned)
        self.assertIn("[群聊]", cleaned)
        self.assertIn("Bearer [REDACTED]", cleaned)
        self.assertIn("[REDACTED]", cleaned)
        self.assertIn("[路径]。", cleaned)
        self.assertNotIn("wxid_vip_alice", cleaned)
        self.assertNotIn("90000002@chatroom", cleaned)
        self.assertNotIn("my_super_secret_token_12345", cleaned)
        self.assertNotIn(fake_sk, cleaned)
        self.assertNotIn("alice_dev", cleaned)

    def test_mcp_config_from_file_and_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing_path = Path(tmp) / "nonexistent_mcp_config.json"
            cfg_missing = MCPConfig.from_file(missing_path)
            self.assertFalse(cfg_missing.enabled)
            self.assertFalse(cfg_missing.web_search.enabled)
            self.assertEqual(dict(cfg_missing.mcp_servers), {})

            # 顶层 enabled=false 时，含未设置的环境变量占位符也应通过校验
            cfg_path = Path(tmp) / "mcp_config.json"
            cfg_path.write_text(
                json.dumps(
                    {
                        "enabled": False,
                        "web_search": {
                            "enabled": True,
                            "provider": "tavily",
                            "base_url": "https://api.tavily.com",
                            "api_key": "${UNSET_TAVILY_KEY}",
                        },
                        "mcp_servers": {
                            "demo": {
                                "enabled": True,
                                "transport": "streamable_http",
                                "url": "https://mcp.example.test/mcp",
                                "headers": {"Authorization": "Bearer ${UNSET_MCP_TOKEN}"},
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            disabled_cfg = MCPConfig.from_file(cfg_path, env={})
            self.assertFalse(disabled_cfg.enabled)

            # 顶层 enabled=true 且缺密钥时抛 MCPConfigError
            cfg_path.write_text(
                json.dumps(
                    {
                        "enabled": True,
                        "web_search": {"enabled": True, "provider": "tavily", "api_key": ""},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(MCPConfigError):
                MCPConfig.from_file(cfg_path, env={})

            # 不支持 sse 传输
            with self.assertRaises(MCPConfigError):
                MCPServerConfig(name="legacy", transport="sse", url="https://mcp.example.test/sse")

            # access="public" 时禁止开启 allow_write=True
            with self.assertRaises(MCPConfigError):
                MCPServerConfig(
                    name="pub_write",
                    transport="streamable_http",
                    url="https://mcp.example.test/mcp",
                    access="public",
                    allow_write=True,
                )


class TestNamingSchemaAndGates(unittest.TestCase):
    def test_make_mcp_tool_name_and_hash_clamp(self) -> None:
        self.assertEqual(make_mcp_tool_name("alpha-search", "web.query"), "mcp__alpha_search__web_query")
        self.assertEqual(make_mcp_tool_name("evil", "memory_save"), "mcp__evil__memory_save")

        long_tool = "very_long_tool_operation_name_" * 3
        clamped1 = make_mcp_tool_name("enterprise_server_alpha", long_tool)
        clamped2 = make_mcp_tool_name("enterprise_server_alpha", long_tool)
        clamped_other = make_mcp_tool_name("enterprise_server_alpha", long_tool + "_2")
        self.assertEqual(len(clamped1), 64)
        self.assertEqual(clamped1, clamped2)
        self.assertNotEqual(clamped1, clamped_other)
        self.assertRegex(clamped1, r"^mcp__[A-Za-z0-9_]+_[0-9a-f]{8}$")

    def test_normalize_mcp_schema_and_restore_arguments(self) -> None:
        raw_schema = {
            "definitions": {
                "Filter": {
                    "type": "object",
                    "properties": {"tag": {"type": "string"}},
                    "required": ["tag", "nonexistent_field"],
                }
            },
            "properties": {
                "query": {"type": "string"},
                "definitions": {"type": "string", "description": "合法属性名 definitions 不应变成 $defs"},
                "account": {"type": "string", "description": "远端同名参数"},
                "wxid": {"type": "string"},
                "filter": {"$ref": "#/definitions/Filter"},
            },
            "required": ["query", "account", "dangling_param"],
        }
        norm_schema, rename_map = normalize_mcp_schema(raw_schema)
        self.assertEqual(norm_schema["type"], "object")
        self.assertIn("$defs", norm_schema)
        self.assertNotIn("definitions", norm_schema)
        self.assertEqual(norm_schema["$defs"]["Filter"]["required"], ["tag"])

        props = norm_schema["properties"]
        self.assertIn("definitions", props)
        self.assertIn("mcp_arg_account", props)
        self.assertIn("mcp_arg_wxid", props)
        self.assertNotIn("account", props)
        self.assertNotIn("wxid", props)
        self.assertEqual(props["filter"]["$ref"], "#/$defs/Filter")
        self.assertEqual(norm_schema["required"], ["query", "mcp_arg_account"])
        self.assertEqual(rename_map, {"mcp_arg_account": "account", "mcp_arg_wxid": "wxid"})

        restored = restore_mcp_arguments(
            {
                "query": "hello",
                "mcp_arg_account": "corp_acct_1",
                "room": "injected_room_should_be_dropped",
            },
            rename_map,
        )
        self.assertEqual(restored, {"query": "hello", "account": "corp_acct_1"})

    def test_read_only_classification_and_tool_filters(self) -> None:
        tool_with_hint = {"name": "search_docs", "annotations": {"readOnlyHint": True}}
        tool_no_hint = {"name": "create_doc", "annotations": {}}

        # 默认 Fail-Closed：即使远端带 readOnlyHint=True，未开启 trust_annotations 时仍视为写工具
        srv_default = MCPServerConfig(name="s1", access="vip")
        self.assertFalse(classify_mcp_tool_read_only(tool_with_hint, srv_default))

        # VIP + trust_annotations=True 时采信 readOnlyHint=True
        srv_trusted = MCPServerConfig(name="s2", access="vip", trust_annotations=True)
        self.assertTrue(classify_mcp_tool_read_only(tool_with_hint, srv_trusted))
        self.assertFalse(classify_mcp_tool_read_only(tool_no_hint, srv_trusted))

        # public Server 即便配置 trust_annotations=True 也不采信远端自报注解，必须显式 read_only 或白名单
        srv_pub = MCPServerConfig(
            name="s3",
            access="public",
            trust_annotations=True,
            read_only_tools=frozenset({"safe_lookup"}),
        )
        self.assertFalse(classify_mcp_tool_read_only(tool_with_hint, srv_pub))
        self.assertTrue(classify_mcp_tool_read_only({"name": "safe_lookup"}, srv_pub))

        # include_tools 非空时优先级高于 exclude_tools
        srv_filter = MCPServerConfig(
            name="s4",
            include_tools=("doc_*",),
            exclude_tools=("doc_secret",),
        )
        self.assertTrue(is_mcp_tool_included("doc_secret", srv_filter))
        self.assertFalse(is_mcp_tool_included("other_tool", srv_filter))

        srv_exclude_only = MCPServerConfig(name="s5", include_tools=(), exclude_tools=("admin_*",))
        self.assertFalse(is_mcp_tool_included("admin_drop", srv_exclude_only))
        self.assertTrue(is_mcp_tool_included("user_read", srv_exclude_only))

    def test_circuit_breaker_cooldown_and_half_open_recovery(self) -> None:
        clock = _FakeClock(1000.0)
        cb = CircuitBreaker(failure_threshold=3, cooldown_seconds=60.0, clock=clock)

        self.assertTrue(cb.allow_request())
        cb.record_failure()
        cb.record_failure()
        self.assertFalse(cb.is_open)
        self.assertTrue(cb.allow_request())

        cb.record_failure()
        self.assertTrue(cb.is_open)
        self.assertFalse(cb.allow_request())

        # 冷却 59 秒仍拦截
        clock.advance(59.0)
        self.assertFalse(cb.allow_request())

        # 到 60 秒进入半开探测：若探测失败立即重新熔断 60 秒
        clock.advance(1.0)
        self.assertTrue(cb.allow_request())
        cb.record_failure()
        self.assertTrue(cb.is_open)
        self.assertFalse(cb.allow_request())

        # 再次期满后探测成功即完全恢复
        clock.advance(60.0)
        self.assertTrue(cb.allow_request())
        cb.record_success()
        self.assertFalse(cb.is_open)
        self.assertEqual(cb.consecutive_failures, 0)


class TestTransportsAndManager(unittest.TestCase):
    def test_streamable_http_json_sse_session_and_delete(self) -> None:
        seen_methods: list[tuple[str, dict[str, str], dict | None]] = []

        def handler(req, timeout):
            hdrs = {k.lower(): v for k, v in req.header_items()}
            method = req.get_method()
            body = json.loads(req.data.decode("utf-8")) if req.data else None
            seen_methods.append((method, hdrs, body))

            if method == "DELETE":
                return _MockHTTPResponse(b"", status=204, headers={})
            if body and "id" not in body:
                return _MockHTTPResponse(
                    b"",
                    status=202,
                    headers={"Mcp-Session-Id": "sess-alpha-001"},
                )
            req_id = body["id"]
            rpc_method = body["method"]
            if rpc_method == "initialize":
                res = {"jsonrpc": "2.0", "id": req_id, "result": {"protocolVersion": "2025-03-26"}}
                return _MockHTTPResponse(
                    json.dumps(res).encode("utf-8"),
                    headers={"Content-Type": "application/json", "Mcp-Session-Id": "sess-alpha-001"},
                )
            # tools/call 通过 text/event-stream SSE 流返回
            sse_payload = (
                ": keepalive ping\n"
                "event: message\n"
                f'data: {{"jsonrpc":"2.0","id":{req_id},"result":{{"content":[{{"type":"text","text":"SSE结果"}}]}}}}\n\n'
            ).encode("utf-8")
            return _MockHTTPResponse(
                sse_payload,
                headers={"Content-Type": "text/event-stream; charset=utf-8"},
            )

        opener = _MockOpener(handler)
        transport = StreamableHTTPMCPTransport(
            "https://mcp.example.test/mcp",
            headers={"Authorization": "Bearer test_http_token_123456"},
            opener=opener,
        )
        init_res = transport.request("initialize", {"protocolVersion": "2025-03-26"})
        self.assertEqual(init_res["protocolVersion"], "2025-03-26")
        self.assertEqual(transport.session_id, "sess-alpha-001")

        transport.notify("notifications/initialized", {})
        call_res = transport.request("tools/call", {"name": "ping", "arguments": {}})
        self.assertEqual(call_res["content"][0]["text"], "SSE结果")

        transport.close()
        self.assertEqual(seen_methods[-1][0], "DELETE")
        self.assertEqual(seen_methods[-1][1].get("mcp-session-id"), "sess-alpha-001")

    def test_streamable_http_oversized_response_rejected(self) -> None:
        huge = b'{"jsonrpc":"2.0","id":1,"result":{"data":"' + (b"a" * 55_000) + b'"}}'
        opener = _MockOpener(lambda req, timeout: _MockHTTPResponse(huge))
        transport = StreamableHTTPMCPTransport("https://mcp.example.test/mcp", opener=opener)
        with self.assertRaises(MCPError):
            transport.request("initialize", {})

    def test_stdio_env_isolation_stderr_sanitization_timeout_and_tree_cleanup(self) -> None:
        base_env = dict(os.environ)
        base_env["HOST_ONLY_SECRET_KEY"] = "must_never_leak_to_child"
        safe_env = build_safe_stdio_env({"MCP_CUSTOM_FLAG": "allowed_flag"}, base_env=base_env)
        self.assertNotIn("HOST_ONLY_SECRET_KEY", safe_env)
        self.assertEqual(safe_env["MCP_CUSTOM_FLAG"], "allowed_flag")

        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
            cfg = MCPConfig(
                enabled=True,
                mcp_servers={
                    "local_py": MCPServerConfig(
                        name="local_py",
                        enabled=True,
                        transport="stdio",
                        command=sys.executable,
                        args=("-u", "-c", _STDIO_MOCK_SERVER_SCRIPT),
                        env={"MCP_CUSTOM_FLAG": "ok_value"},
                        access="self",
                        read_only=True,
                        timeout_seconds=1.5,
                    )
                },
            )
            manager = MCPManager(cfg, log_dir=log_dir, env=base_env)
            self.addCleanup(manager.close)

            # 初始化零子进程：连接前不应生成或启动子进程
            self.assertIsNone(manager._states["local_py"].transport)

            # 首次发现：覆盖 initialize -> initialized -> nextCursor 分页 tools/list
            descriptors = manager.list_tool_descriptors()
            names = [d.name for d in descriptors]
            self.assertEqual(names, ["mcp__local_py__inspect_env", "mcp__local_py__slow_op"])

            transport = manager._states["local_py"].transport
            self.assertIsNotNone(transport)
            parent_pid = transport.pid
            self.assertIsInstance(parent_pid, int)
            self.assertTrue(_pid_exists(parent_pid))

            # 调用 inspect_env：验证 mcp_arg_room 还原为 room，且未继承宿主敏感变量
            call_res = manager.call_tool(
                "mcp__local_py__inspect_env",
                {"mcp_arg_room": "proj_alpha", "note": "check"},
            )
            self.assertTrue(call_res["ok"])
            returned = json.loads(call_res["data"])
            self.assertFalse(returned["has_host_secret"])
            self.assertEqual(returned["custom_flag"], "ok_value")
            self.assertEqual(returned["received_args"], {"room": "proj_alpha", "note": "check"})

            grandchild_pid = int(returned["grandchild_pid"])
            self.assertTrue(_pid_exists(grandchild_pid))

            # 验证单次超时不会挂死
            t0 = time.monotonic()
            slow_res = manager.call_tool("mcp__local_py__slow_op", {}, timeout=0.25)
            elapsed = time.monotonic() - t0
            self.assertFalse(slow_res["ok"])
            self.assertLess(elapsed, 1.5)

            # 验证 stderr 日志已落盘且完成脱敏
            stderr_file = log_dir / "mcp-stderr.log"
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline and not stderr_file.read_text(encoding="utf-8"):
                time.sleep(0.03)
            stderr_text = stderr_file.read_text(encoding="utf-8")
            self.assertIn("Bearer [REDACTED]", stderr_text)
            self.assertIn("[微信ID]", stderr_text)
            self.assertNotIn("secret_stderr_token_abcdef12", stderr_text)
            self.assertNotIn("wxid_user_bob", stderr_text)

            # 关闭管理器：验证父进程与孙进程在 <= 2.5s 内全部被终止回收
            close_start = time.monotonic()
            manager.close()
            close_cost = time.monotonic() - close_start
            self.assertLessEqual(close_cost, 2.5)
            self.assertTrue(_wait_pid_gone(parent_pid, timeout=1.5))
            self.assertTrue(_wait_pid_gone(grandchild_pid, timeout=1.5))

    def test_windows_job_object_struct_and_taskkill_fallback_branch(self) -> None:
        # 校验 Win32 JOBOBJECT_EXTENDED_LIMIT_INFORMATION 结构字段与标志常量正确
        info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        self.assertEqual(info.BasicLimitInformation.LimitFlags, 0x2000)
        if ctypes.sizeof(ctypes.c_void_p) == 8:
            self.assertEqual(ctypes.sizeof(_JOBOBJECT_EXTENDED_LIMIT_INFORMATION), 144)

        # 模拟 Windows 分支下的 Popen 参数与超时 taskkill /F /T /PID 兜底
        fake_proc = mock.MagicMock()
        fake_proc.pid = 42424
        fake_proc.stdin = io.BytesIO()
        fake_proc.stdout = io.BytesIO(b"")
        fake_proc.stderr = io.BytesIO(b"")
        fake_proc.wait.side_effect = [subprocess.TimeoutExpired(cmd="npx", timeout=0.5), 0]
        fake_proc.poll.return_value = None

        captured_popen_kwargs: dict[str, object] = {}

        def fake_popen(argv, **kwargs):
            captured_popen_kwargs.update(kwargs)
            return fake_proc

        with tempfile.TemporaryDirectory() as tmp, mock.patch(
            "wxbot_framework.mcp_client._is_windows",
            return_value=True,
        ), mock.patch(
            "wxbot_framework.mcp_client._Win32JobHandle.create_and_assign",
            return_value=None,
        ), mock.patch("wxbot_framework.mcp_client.subprocess.run") as mock_run:
            transport = StdioMCPTransport(
                "npx",
                ("-y", "@example/mcp"),
                log_dir=Path(tmp),
                popen_factory=fake_popen,
            )
            expected_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000) | getattr(
                subprocess,
                "CREATE_NEW_PROCESS_GROUP",
                0x00000200,
            )
            self.assertEqual(captured_popen_kwargs.get("creationflags"), expected_flags)
            self.assertNotIn("start_new_session", captured_popen_kwargs)

            transport.close()
            mock_run.assert_called_once()
            called_cmd = mock_run.call_args[0][0]
            self.assertEqual(called_cmd, ["taskkill", "/F", "/T", "/PID", "42424"])

    def test_manager_circuit_breaker_fast_fail_and_write_gate_and_web_search(self) -> None:
        clock = _FakeClock(500.0)
        tool_call_attempts = 0

        def http_handler(req, timeout):
            nonlocal tool_call_attempts
            if req.full_url.endswith("/search"):
                return _MockHTTPResponse(
                    json.dumps(
                        {
                            "results": [
                                {
                                    "title": "内置搜索结果",
                                    "url": "https://search.example.test/r1",
                                    "content": "内置搜索返回摘要",
                                }
                            ]
                        },
                        ensure_ascii=False,
                    ).encode("utf-8")
                )
            if req.get_method() == "DELETE":
                return _MockHTTPResponse(b"", status=204)
            body = json.loads(req.data.decode("utf-8")) if req.data else {}
            if "id" not in body:
                return _MockHTTPResponse(b"", status=202)
            req_id = body["id"]
            method = body["method"]
            if method == "initialize":
                res = {"protocolVersion": "2025-03-26", "capabilities": {"tools": {}}}
                return _MockHTTPResponse(json.dumps({"jsonrpc": "2.0", "id": req_id, "result": res}).encode("utf-8"))
            if method == "tools/list":
                res = {
                    "tools": [
                        {
                            "name": "read_item",
                            "description": "只读工具",
                            "inputSchema": {"type": "object", "properties": {}},
                        },
                        {
                            "name": "delete_item",
                            "description": "写工具（未在白名单，allow_write=False 时应被过滤）",
                            "inputSchema": {"type": "object", "properties": {}},
                        },
                    ]
                }
                return _MockHTTPResponse(json.dumps({"jsonrpc": "2.0", "id": req_id, "result": res}).encode("utf-8"))
            # tools/call 前 3 次抛超时，第 4 次在冷却后恢复成功
            tool_call_attempts += 1
            if tool_call_attempts <= 3:
                raise TimeoutError("simulated timeout")
            res = {"content": [{"type": "text", "text": "恢复正常返回"}]}
            return _MockHTTPResponse(json.dumps({"jsonrpc": "2.0", "id": req_id, "result": res}).encode("utf-8"))

        opener = _MockOpener(http_handler)
        with tempfile.TemporaryDirectory() as tmp:
            cfg = MCPConfig(
                enabled=True,
                web_search=WebSearchConfig(
                    enabled=True,
                    provider="tavily",
                    base_url="https://search.example.test",
                    api_key="tvly_fake",
                ),
                mcp_servers={
                    "flaky": MCPServerConfig(
                        name="flaky",
                        enabled=True,
                        transport="streamable_http",
                        url="https://mcp.example.test/mcp",
                        read_only=False,
                        read_only_tools=frozenset({"read_item"}),
                        allow_write=False,
                    )
                },
            )
            manager = MCPManager(cfg, log_dir=Path(tmp), clock=clock, http_opener=opener)
            self.addCleanup(manager.close)

            # 只暴露白名单只读工具，写工具 delete_item 被过滤且不可执行
            tools = manager.list_tool_descriptors()
            self.assertEqual([t.name for t in tools], ["mcp__flaky__read_item"])
            denied = manager.call_tool("mcp__flaky__delete_item", {})
            self.assertFalse(denied["ok"])

            # 连续 3 次失败后触发 60 秒熔断，冷却期内第 4 次调用不发网络请求
            for _ in range(3):
                r = manager.call_tool("mcp__flaky__read_item", {})
                self.assertFalse(r["ok"])
            self.assertEqual(tool_call_attempts, 3)

            fast_fail = manager.call_tool("mcp__flaky__read_item", {})
            self.assertFalse(fast_fail["ok"])
            self.assertEqual(tool_call_attempts, 3)  # 未发出第 4 次网络请求

            # 冷却期满 60 秒后半开探测恢复成功
            clock.advance(60.1)
            recovered = manager.call_tool("mcp__flaky__read_item", {})
            self.assertEqual(recovered, {"ok": True, "data": "恢复正常返回"})
            self.assertEqual(tool_call_attempts, 4)

            # 同时内置 web_search 正常工作
            ws_res = manager.search_web("测试检索")
            self.assertTrue(ws_res["ok"])
            self.assertEqual(ws_res["results"][0]["title"], "内置搜索结果")

    def test_single_server_failure_isolation_and_reload(self) -> None:
        def http_handler(req, timeout):
            if req.get_method() == "DELETE":
                return _MockHTTPResponse(b"", status=204)
            body = json.loads(req.data.decode("utf-8")) if req.data else {}
            if "id" not in body:
                return _MockHTTPResponse(b"", status=202)
            req_id = body["id"]
            method = body["method"]
            if method == "initialize":
                res = {"protocolVersion": "2025-03-26", "capabilities": {"tools": {}}}
            elif method == "tools/list":
                res = {
                    "tools": [
                        {
                            "name": "weather",
                            "description": "查询天气",
                            "inputSchema": {"type": "object", "properties": {"city": {"type": "string"}}},
                        }
                    ]
                }
            elif method == "tools/call":
                city = body["params"]["arguments"].get("city", "")
                res = {"content": [{"type": "text", "text": f"{city}晴 25°C"}]}
            else:
                res = {}
            return _MockHTTPResponse(json.dumps({"jsonrpc": "2.0", "id": req_id, "result": res}).encode("utf-8"))

        opener = _MockOpener(http_handler)
        with tempfile.TemporaryDirectory() as tmp:
            cfg = MCPConfig(
                enabled=True,
                mcp_servers={
                    "broken_stdio": MCPServerConfig(
                        name="broken_stdio",
                        enabled=True,
                        transport="stdio",
                        command="nonexistent_mcp_executable_xyz_999",
                        read_only=True,
                    ),
                    "healthy_http": MCPServerConfig(
                        name="healthy_http",
                        enabled=True,
                        transport="streamable_http",
                        url="https://mcp.example.test/mcp",
                        access="public",
                        allow_in_group=True,
                        read_only=True,
                    ),
                },
            )
            manager = MCPManager(cfg, log_dir=Path(tmp), http_opener=opener)
            self.addCleanup(manager.close)

            with self.assertLogs("wxbot_framework.mcp_client", level=logging.WARNING):
                tools = manager.list_tool_descriptors()
            self.assertEqual([t.name for t in tools], ["mcp__healthy_http__weather"])

            out = manager.call_tool("mcp__healthy_http__weather", {"city": "深圳"})
            self.assertEqual(out, {"ok": True, "data": "深圳晴 25°C"})

            # reload 为禁用配置后应关闭旧连接且不再暴露外部工具
            manager.reload(MCPConfig(enabled=False))
            self.assertEqual(manager.list_tool_descriptors(), ())


if __name__ == "__main__":
    unittest.main()
