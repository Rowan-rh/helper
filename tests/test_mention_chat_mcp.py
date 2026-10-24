"""tests/test_mention_chat_mcp.py

验证 MCP 阶段 3：
1. init_config.py 生成 mcp_config.json 及已存在时不覆盖；
2. check_publication.py 模板检查与对 mcp_config 结构化安全扫描（拦截非空明文密钥与开启态模板，且绝不输出密钥内容）；
3. mention_chat 插件接入 MCPManager：
   - 群聊、私聊（VIP）、文件传输助手（/ai）三入口端到端 Agent 工具调用与动态权限过滤；
   - 未配置模型（ctx.llm 抛错）时优先抛错，不触发任何 MCP 操作。
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from plugins.mention_chat import Plugin as MentionChatPlugin
from tools import check_publication, init_config
from wxbot_framework.agent import (
    AgentConfig,
    SessionKey,
)
from wxbot_framework.agent_tools import create_agent_tool_registry
from wxbot_framework.llm import ChatResult, LLMError
from wxbot_framework.mcp_client import (
    MCPConfig,
    MCPToolDescriptor,
    WebSearchConfig,
)
from wxbot_framework.members import MemberStore
from wxbot_framework.sdk import Message, PluginContext, ServiceFacade

ROOT = Path(__file__).resolve().parent.parent

ACCOUNT = "wxid_robot_main"
ROOM_ID = "12345678@chatroom"
VIP_USER = "wxid_vip_boss"
NORMAL_USER = "wxid_normal_user"


class MockBackend:
    session_id = "test_session_mcp"

    def __init__(self):
        self.sent = []

    def send_text(self, to_wxid, content, *, expected_session=None, _mentions=(), delivery_guard=None):
        if delivery_guard is not None:
            with delivery_guard():
                self.sent.append((to_wxid, content, ()))
        else:
            self.sent.append((to_wxid, content, ()))
        return True

    def send_at_text(self, room_wxid, content, at_list, *, expected_session=None, display_names=None, delivery_guard=None):
        if delivery_guard is not None:
            with delivery_guard():
                self.sent.append((room_wxid, content, tuple(at_list)))
        else:
            self.sent.append((room_wxid, content, tuple(at_list)))
        return True

    def stop(self):
        return {"dll_unloaded": True}


class FakeLLM:
    def __init__(self, *responses, enable_tools: bool = True):
        self.responses = list(responses)
        self.requests = []
        self.enable_tools = enable_tools

    def complete_chat(self, messages, *, tools=None, tool_choice="auto"):
        self.requests.append({"messages": messages, "tools": tools})
        if not self.responses:
            return ChatResult("默认模型回复")
        item = self.responses.pop(0)
        if callable(item):
            return item(messages, tools)
        if isinstance(item, Exception):
            raise item
        return item


def _tool_reply(*calls, content: str = "") -> ChatResult:
    message = {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {
                "id": item["id"],
                "type": "function",
                "function": {
                    "name": item["name"],
                    "arguments": json.dumps(item.get("arguments") or {}, ensure_ascii=False),
                },
            }
            for item in calls
        ],
    }
    return ChatResult(content=content, tool_calls=list(calls), finish_reason="tool_calls", message=message)


def _text_reply(text: str) -> ChatResult:
    return ChatResult(content=text, finish_reason="stop", message={"role": "assistant", "content": text})


class FakeMCPManager:
    """用于测试的假 MCPManager，提供预设的搜索与外部工具描述。"""

    def __init__(self):
        self.web_search_enabled = True
        self.web_search_config = WebSearchConfig(
            enabled=True,
            provider="tavily",
            base_url="https://api.tavily.com",
            timeout_seconds=5.0,
            max_results=3,
            max_result_chars=1000,
        )
        self.search_calls = []
        self.tool_calls = []

    def search_web(self, query: str, limit: int | None = None, timeout: float | None = None) -> dict:
        self.search_calls.append({"query": query, "limit": limit, "timeout": timeout})
        return {
            "query": query,
            "provider": "tavily",
            "results": [
                {
                    "title": f"搜索结果: {query}",
                    "url": "https://example.com/result",
                    "snippet": f"关于 {query} 的公开信息摘要",
                }
            ],
        }

    def list_tool_descriptors(self) -> tuple[MCPToolDescriptor, ...]:
        return (
            MCPToolDescriptor(
                name="mcp__public_info__fetch_data",
                server_name="public_info",
                original_name="fetch_data",
                description="公开信息查询",
                parameters_schema={"type": "object", "properties": {"topic": {"type": "string"}}},
                rename_map={},
                read_only=True,
                write=False,
                privileged_only=False,
                access="public",
                allow_in_group=True,
                timeout_seconds=5.0,
                max_result_chars=2000,
            ),
            MCPToolDescriptor(
                name="mcp__vip_reports__get_report",
                server_name="vip_reports",
                original_name="get_report",
                description="VIP 报告查询（仅 VIP 可在群聊或私聊调用）",
                parameters_schema={"type": "object", "properties": {"report_id": {"type": "string"}}},
                rename_map={},
                read_only=True,
                write=False,
                privileged_only=True,
                access="vip",
                allow_in_group=True,
                timeout_seconds=5.0,
                max_result_chars=2000,
            ),
            MCPToolDescriptor(
                name="mcp__private_note__query_notes",
                server_name="private_note",
                original_name="query_notes",
                description="私聊笔记（禁止在群聊暴露）",
                parameters_schema={"type": "object", "properties": {"keyword": {"type": "string"}}},
                rename_map={},
                read_only=True,
                write=False,
                privileged_only=True,
                access="vip",
                allow_in_group=False,
                timeout_seconds=5.0,
                max_result_chars=2000,
            ),
            MCPToolDescriptor(
                name="mcp__admin_ops__system_diag",
                server_name="admin_ops",
                original_name="system_diag",
                description="系统自检诊断（仅主号本人可用）",
                parameters_schema={"type": "object", "properties": {"check_all": {"type": "boolean"}}},
                rename_map={},
                read_only=True,
                write=False,
                privileged_only=True,
                access="self",
                allow_in_group=False,
                timeout_seconds=5.0,
                max_result_chars=2000,
            ),
        )

    def call_tool(self, name: str, arguments: dict, timeout: float | None = None) -> dict:
        self.tool_calls.append({"name": name, "arguments": arguments, "timeout": timeout})
        return {"status": "ok", "called_tool": name, "args": arguments}


class TestInitConfigAndCheckPublication(unittest.TestCase):
    """验证 tools/init_config.py 与 tools/check_publication.py。"""

    def test_init_config_creates_mcp_config_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            # 复制各 .example.json 到临时目录
            for name in ("bot_config", "members_config", "llm_config", "mcp_config"):
                src = ROOT / f"{name}.example.json"
                (temp_path / f"{name}.example.json").write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

            created = init_config.initialize(temp_path)
            self.assertIn("mcp_config.json", created)
            self.assertIn("bot_config.json", created)
            self.assertIn("members_config.json", created)
            self.assertIn("llm_config.json", created)

            # 验证生成的 mcp_config.json 内容与模板一致
            generated_data = json.loads((temp_path / "mcp_config.json").read_text(encoding="utf-8"))
            self.assertFalse(generated_data["enabled"])
            self.assertFalse(generated_data["web_search"]["enabled"])

            # 再次运行不覆盖已存在的文件
            created_again = init_config.initialize(temp_path)
            self.assertEqual(created_again, [])

    def test_check_publication_clean_on_repo(self):
        """当前仓库中的模板与配置文件应通过发布检查（忽略其他并发单测生成的临时 .runtime 目录）。"""
        res = check_publication.inspect(ROOT)
        non_runtime_findings = [
            f for f in res["findings"]
            if f.get("kind") != "runtime_or_legacy_directory"
        ]
        self.assertEqual(non_runtime_findings, [], f"发布检查未通过: {non_runtime_findings}")

        # 在完全干净的目录中使用仓库公开模板，验证整体 passed 为 True
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            for name in ("bot_config", "members_config", "llm_config", "mcp_config"):
                (temp_path / f"{name}.example.json").write_text(
                    (ROOT / f"{name}.example.json").read_text(encoding="utf-8"), encoding="utf-8"
                )
            clean_res = check_publication.inspect(temp_path)
            self.assertTrue(clean_res["passed"], f"干净环境发布检查应通过: {clean_res['findings']}")

    def test_check_publication_missing_mcp_config_template(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            for name in ("bot_config", "members_config", "llm_config"):
                (temp_path / f"{name}.example.json").write_text("{}", encoding="utf-8")
            res = check_publication.inspect(temp_path)
            self.assertFalse(res["passed"])
            kinds = [f["kind"] for f in res["findings"] if f["file"] == "mcp_config.example.json"]
            self.assertIn("missing_config_template", kinds)

    def test_check_publication_template_enabled_detected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            for name in ("bot_config", "members_config", "llm_config"):
                (temp_path / f"{name}.example.json").write_text("{}", encoding="utf-8")

            # 1. 顶层 enabled 为 true
            tpl = temp_path / "mcp_config.example.json"
            tpl.write_text(json.dumps({"enabled": True, "web_search": {"enabled": False}}), encoding="utf-8")
            res = check_publication.inspect(temp_path)
            self.assertIn("enabled_in_config_template", [f["kind"] for f in res["findings"]])

            # 2. web_search.enabled 为 true
            tpl.write_text(json.dumps({"enabled": False, "web_search": {"enabled": True}}), encoding="utf-8")
            res = check_publication.inspect(temp_path)
            self.assertIn("enabled_in_config_template", [f["kind"] for f in res["findings"]])

            # 3. 任一 server enabled 为 true
            tpl.write_text(json.dumps({
                "enabled": False,
                "web_search": {"enabled": False},
                "mcp_servers": {"srv1": {"enabled": True}},
            }), encoding="utf-8")
            res = check_publication.inspect(temp_path)
            self.assertIn("enabled_in_config_template", [f["kind"] for f in res["findings"]])

    def test_check_publication_secret_scanning_and_no_leak(self):
        secret_literal = "my_custom_secret_token_never_leak"
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            for name in ("bot_config", "members_config", "llm_config", "mcp_config"):
                (temp_path / f"{name}.example.json").write_text(
                    (ROOT / f"{name}.example.json").read_text(encoding="utf-8"), encoding="utf-8"
                )

            # 写入含明文密钥的 mcp_config.json
            cfg = temp_path / "mcp_config.json"
            cfg.write_text(json.dumps({
                "enabled": False,
                "web_search": {"enabled": False, "api_key": "my_plaintext_tavily_key"},
                "mcp_servers": {
                    "srv1": {
                        "enabled": False,
                        "headers": {"Authorization": f"Bearer {secret_literal}"},
                        "env": {"API_KEY": "plaintext_api_key_123"},
                    }
                }
            }), encoding="utf-8")

            res = check_publication.inspect(temp_path)
            self.assertFalse(res["passed"])
            findings = [f for f in res["findings"] if f["file"] == "mcp_config.json"]
            kinds = {f["kind"] for f in findings}
            self.assertIn("nonempty_api_key", kinds)
            self.assertIn("nonempty_mcp_secret", kinds)

            # 绝不输出任何密钥原文
            report_str = json.dumps(res, ensure_ascii=False)
            self.assertNotIn(secret_literal, report_str)
            self.assertNotIn("my_plaintext_tavily_key", report_str)
            self.assertNotIn("plaintext_api_key_123", report_str)

    def test_check_publication_allows_valid_env_placeholders(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            for name in ("bot_config", "members_config", "llm_config", "mcp_config"):
                (temp_path / f"{name}.example.json").write_text(
                    (ROOT / f"{name}.example.json").read_text(encoding="utf-8"), encoding="utf-8"
                )

            cfg = temp_path / "mcp_config.json"
            cfg.write_text(json.dumps({
                "enabled": False,
                "web_search": {"enabled": False, "api_key": "${TAVILY_API_KEY}"},
                "mcp_servers": {
                    "srv1": {
                        "enabled": False,
                        "headers": {"Authorization": "Bearer ${REMOTE_TOKEN}"},
                        "env": {"API_KEY": "${LOCAL_KEY}"},
                    }
                }
            }), encoding="utf-8")

            res = check_publication.inspect(temp_path)
            mcp_findings = [f for f in res["findings"] if f["file"] == "mcp_config.json"]
            self.assertEqual(mcp_findings, [])


class TestMentionChatMCPEndToEnd(unittest.TestCase):
    """验证 mention_chat 插件接入 MCP 后的三入口端到端 Agent 与权限裁剪。"""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.runtime_dir = self.root / ".runtime"
        self.runtime_dir.mkdir(parents=True, exist_ok=True)

        self.config_file = self.root / "members_config.json"
        config_data = {
            "database_file": ".runtime/member_data.sqlite3",
            "history_enabled": True,
            "daily_clear": True,
            "agent_vip_wxids": [VIP_USER],
            "groups": {
                ROOM_ID: [
                    {"key": "vip", "name": "VIP老板", "display_name": "VIP老板", "wxid": VIP_USER},
                    {"key": "normal", "name": "普通成员", "display_name": "普通同学", "wxid": NORMAL_USER},
                ],
            },
        }
        self.config_file.write_text(json.dumps(config_data, ensure_ascii=False), encoding="utf-8")

        self.member_store = MemberStore(self.config_file, ACCOUNT)
        self.member_store.initialize()

        self.backend = MockBackend()
        self.service = ServiceFacade(text_delay={"enabled": False})
        self.service.bind(self.backend)

        self.mcp_manager = FakeMCPManager()
        self.llm = FakeLLM()

        self.ctx = PluginContext(
            "mention_chat",
            {"agent": {"total_budget": 10.0, "max_turns": 10}},
            self.service,
            [
                "text.receive", "text.send", "friend.text.receive", "friend.text.send",
                "group.text.receive", "group.text.send", "group.text.mention",
            ],
            member_store=self.member_store,
            mcp_manager=self.mcp_manager,
        )
        self.ctx._llm = self.llm

        self.plugin = MentionChatPlugin()
        self.plugin.on_load(self.ctx)

    def tearDown(self):
        self.plugin.on_unload(self.ctx)
        self.service.close()
        self.member_store.close()
        self.temp_dir.cleanup()

    def record_and_make_group_at(self, text, *, sender=NORMAL_USER, room=ROOM_ID, sequence=10):
        msg = Message(
            text=f"@机器人\u2005{text}",
            sender_id=sender,
            chat_id=room,
            is_group=True,
            session_id="test_session_mcp",
            sequence=sequence,
            data={
                "bot_generated": False,
                "is_at_self": True,
                "at_members": (ACCOUNT,),
            },
        )
        self.member_store.record(msg)
        return msg

    def test_group_chat_normal_user_calls_web_search(self):
        """普通成员在群聊中艾特机器人，可调用 web_search 等 public 工具，被裁剪掉 vip/self 工具。"""
        # 第一轮：LLM 发起 web_search
        call1 = {"id": "call_ws_1", "name": "web_search", "arguments": {"query": "helper 项目"}}
        self.llm.responses = [
            _tool_reply(call1),
            _text_reply("根据搜索，helper 是一个微信文本机器人框架。"),
        ]

        msg = self.record_and_make_group_at("帮我查一下 helper 项目", sender=NORMAL_USER)
        self.plugin.on_message(msg, self.ctx)

        # 验证 LLM 收到的工具列表被正确裁剪
        tools_offered = self.llm.requests[0]["tools"]
        offered_names = [t["function"]["name"] for t in tools_offered]
        self.assertIn("web_search", offered_names)
        self.assertIn("mcp__public_info__fetch_data", offered_names)
        self.assertNotIn("mcp__vip_reports__get_report", offered_names)  # VIP 专属
        self.assertNotIn("mcp__private_note__query_notes", offered_names)  # 群聊禁用
        self.assertNotIn("mcp__admin_ops__system_diag", offered_names)  # self 专属

        # 验证 search_web 被执行
        self.assertEqual(len(self.mcp_manager.search_calls), 1)
        self.assertEqual(self.mcp_manager.search_calls[0]["query"], "helper 项目")

        # 验证回复发送到群里且带 @普通同学
        self.assertEqual(len(self.backend.sent), 1)
        room, content, at_list = self.backend.sent[0]
        self.assertEqual(room, ROOM_ID)
        self.assertIn("根据搜索，helper 是一个微信文本机器人框架。", content)
        self.assertEqual(at_list, (NORMAL_USER,))

    def test_group_chat_vip_user_can_access_vip_tools(self):
        """VIP 成员在群聊中艾特机器人，可访问 access='vip' 且 allow_in_group=True 的工具。"""
        call1 = {"id": "call_vip_1", "name": "mcp__vip_reports__get_report", "arguments": {"report_id": "R001"}}
        self.llm.responses = [
            _tool_reply(call1),
            _text_reply("报告 R001 内容：一切正常。"),
        ]

        msg = self.record_and_make_group_at("查一下报告 R001", sender=VIP_USER)
        self.plugin.on_message(msg, self.ctx)

        tools_offered = self.llm.requests[0]["tools"]
        offered_names = [t["function"]["name"] for t in tools_offered]
        self.assertIn("mcp__vip_reports__get_report", offered_names)
        self.assertNotIn("mcp__private_note__query_notes", offered_names)  # allow_in_group 为 False
        self.assertNotIn("mcp__admin_ops__system_diag", offered_names)  # 仍非 self

        self.assertEqual(len(self.mcp_manager.tool_calls), 1)
        self.assertEqual(self.mcp_manager.tool_calls[0]["name"], "mcp__vip_reports__get_report")
        self.assertIn("报告 R001 内容：一切正常。", self.backend.sent[0][1])

    def test_private_chat_vip_can_access_allow_in_group_false_tools(self):
        """私聊中 VIP 成员可访问 allow_in_group=False 但 access='vip' 的工具。"""
        call1 = {"id": "call_priv_1", "name": "mcp__private_note__query_notes", "arguments": {"keyword": "会议"}}
        self.llm.responses = [
            _tool_reply(call1),
            _text_reply("为您找到私人笔记：下午两点例会。"),
        ]

        msg = Message(
            text="查下会议笔记",
            sender_id=VIP_USER,
            chat_id=VIP_USER,
            is_group=False,
            data={},
        )
        self.plugin.on_message(msg, self.ctx)

        tools_offered = self.llm.requests[0]["tools"]
        offered_names = [t["function"]["name"] for t in tools_offered]
        self.assertIn("mcp__private_note__query_notes", offered_names)
        self.assertNotIn("mcp__admin_ops__system_diag", offered_names)

        self.assertEqual(len(self.mcp_manager.tool_calls), 1)
        self.assertEqual(self.mcp_manager.tool_calls[0]["name"], "mcp__private_note__query_notes")
        self.assertEqual(self.backend.sent[0][0], VIP_USER)
        self.assertIn("为您找到私人笔记：下午两点例会。", self.backend.sent[0][1])
        self.assertEqual(self.backend.sent[0][2], ())  # 私聊不带 @

    def test_private_chat_non_vip_ignored(self):
        """私聊非 VIP 消息静默忽略，不调用 LLM。"""
        msg = Message(
            text="你好",
            sender_id=NORMAL_USER,
            chat_id=NORMAL_USER,
            is_group=False,
            data={},
        )
        self.plugin.on_message(msg, self.ctx)
        self.assertEqual(len(self.llm.requests), 0)
        self.assertEqual(len(self.backend.sent), 0)

    def test_filehelper_agent_can_access_self_tools(self):
        """文件传输助手入口（主号本人发送 /ai）可访问 access='self' 的管理工具。"""
        call1 = {"id": "call_self_1", "name": "mcp__admin_ops__system_diag", "arguments": {"check_all": True}}
        self.llm.responses = [
            _tool_reply(call1),
            _text_reply("自检诊断完成：各模块状态良好。"),
        ]

        msg = Message(
            text="/ai 执行系统自检",
            sender_id=ACCOUNT,
            chat_id="filehelper",
            is_group=False,
            data={},
        )
        self.plugin.on_message(msg, self.ctx)

        tools_offered = self.llm.requests[0]["tools"]
        offered_names = [t["function"]["name"] for t in tools_offered]
        self.assertIn("mcp__admin_ops__system_diag", offered_names)

        self.assertEqual(len(self.mcp_manager.tool_calls), 1)
        self.assertEqual(self.mcp_manager.tool_calls[0]["name"], "mcp__admin_ops__system_diag")
        self.assertEqual(self.backend.sent[0][0], "filehelper")
        self.assertIn("自检诊断完成：各模块状态良好。", self.backend.sent[0][1])

    def test_llm_config_error_blocks_on_load_prior_to_mcp(self):
        """llm_config 缺少 api_key 时，ctx.llm 抛错且优先打断，不进行后续初始化。"""
        plugin = MentionChatPlugin()

        class BrokenContext:
            def __init__(self, members):
                self.members = members
                self.config = {}
                self.logger = mock.MagicMock()

            @property
            def llm(self):
                raise ValueError("请在大模型配置文件中填写 api_key")

            @property
            def mcp(self):
                raise AssertionError("不应访问 mcp")

        broken_ctx = BrokenContext(self.member_store)

        with self.assertRaises(ValueError):
            plugin.on_load(broken_ctx)


if __name__ == "__main__":
    unittest.main()
