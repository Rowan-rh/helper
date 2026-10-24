"""第二期：Agent 提醒工具（reminder_create / reminder_list / reminder_cancel）与协同路由测试。

全部使用假大模型（FakeLLM）、假后端（FakeBackend）、假时钟与虚构 ID，不连接真实微信、不调用付费模型。
"""
from __future__ import annotations

import json
import logging
import tempfile
import unittest
import uuid
from datetime import datetime, timezone
from pathlib import Path

from plugins.mention_chat import Plugin as MentionChatPlugin
from plugins.reminder import Plugin as ReminderPlugin
from plugins.reminder.parser import NO_REPEAT_TIP, ONLY_SELF_TIP, SUPPLEMENT_TIME_TIP, USAGE_TIP
from wxbot_framework.agent import (
    RESERVED_PARAMETERS,
    AgentConfig,
    ToolContext,
    ToolExecutor,
    build_system_prompt,
)
from wxbot_framework.agent_tools import (
    REMINDER_GROUP_ONLY_ERROR,
    create_agent_tool_registry,
    create_agent_tool_specs,
    create_reminder_tool_specs,
    format_beijing_now,
)
from wxbot_framework.llm import ChatResult
from wxbot_framework.members import BEIJING, MemberStore
from wxbot_framework.reminders import ReminderScheduler, ReminderStore, prepare_reminder
from wxbot_framework.sdk import Message, PluginContext, ServiceFacade

ROBOT = "wxid_bot_demo"
ROOM = "100001@chatroom"
OTHER_ROOM = "100002@chatroom"
ALICE = "wxid_alice_demo"      # 普通非 VIP 群成员
BOB = "wxid_bob_demo"          # 普通非 VIP 群成员
VIP_USER = "wxid_vip_demo"     # VIP 成员
AT = chr(0x2005)
AT_ROBOT = f"@机器人{AT} "

# 基准时间：北京时间 2026-10-24 15:00:00（星期六）
BASE_NOW = datetime(2026, 10, 24, 15, 0, 0, tzinfo=BEIJING).timestamp()


def make_tool_call(name: str, arguments: dict, call_id: str = "call_1") -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
        "name": name,
        "arguments": arguments,
    }


def tool_chat_result(*calls: dict, content: str = "") -> ChatResult:
    message = {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {"id": c["id"], "type": "function", "function": c["function"]}
            for c in calls
        ],
    }
    return ChatResult(content=content, tool_calls=list(calls), finish_reason="tool_calls", message=message)


def text_chat_result(text: str) -> ChatResult:
    return ChatResult(content=text, finish_reason="stop", message={"role": "assistant", "content": text})


class FakeBackend:
    def __init__(self):
        self.session_id = uuid.uuid4().hex
        self.sent = []  # (target, content, at_tuple)

    def send_text(self, target, content):
        self.sent.append((target, content, ()))
        return True

    def send_at_text(self, room, content, at_list):
        self.sent.append((room, content, tuple(at_list)))
        return True

    def stop(self):
        return {"dll_unloaded": True}


class FakeLLM:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []
        self.enable_tools = True

    def complete_chat(self, messages, *, tools=None, tool_choice="auto"):
        self.requests.append({
            "messages": json.loads(json.dumps(messages, ensure_ascii=False)),
            "tools": json.loads(json.dumps(tools, ensure_ascii=False)) if tools is not None else None,
        })
        if not self.responses:
            return text_chat_result("默认回复")
        item = self.responses.pop(0)
        if callable(item):
            return item(messages, tools)
        if isinstance(item, BaseException):
            raise item
        return item


class ReminderAgentToolsTestBase(unittest.TestCase):
    def setUp(self):
        previous = logging.root.manager.disable
        logging.disable(logging.CRITICAL)
        self.addCleanup(logging.disable, previous)

        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name).resolve()
        (self.root / ".runtime").mkdir(parents=True, exist_ok=True)

        self.members_path = self.root / "members_config.json"
        self.members_path.write_text(
            json.dumps(
                {
                    "database_file": ".runtime/member_data.sqlite3",
                    "history_enabled": True,
                    "daily_clear": True,
                    "agent_vip_wxids": [VIP_USER],
                    "groups": {ROOM: [], OTHER_ROOM: []},
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        self.members = MemberStore(self.members_path, ROBOT)
        self.addCleanup(self.members.close)
        self.members.initialize()
        self.members.register("阿狸", ROOM, ALICE)
        self.members.remember_wechat_name(BOB, "大壮")

        self.reminder_store = ReminderStore.default(ROBOT, self.root)
        self.addCleanup(self.reminder_store.close)

        self.backend = FakeBackend()
        self.service = ServiceFacade(text_delay={"enabled": False})
        self.service.bind(self.backend)
        self.addCleanup(self.service.close)

        self.now = BASE_NOW
        self.active_plugins = {"reminder": True, "mention_chat": True}
        self.llm = FakeLLM()

        caps = (
            "text.receive",
            "text.send",
            "friend.text.receive",
            "friend.text.send",
            "group.text.receive",
            "group.text.send",
            "group.text.mention",
        )
        self.reminder_ctx = PluginContext(
            "reminder",
            {},
            self.service,
            caps,
            member_store=self.members,
            reminder_store=self.reminder_store,
            plugin_active_check=lambda name: self.active_plugins.get(name, False),
        )
        self.mention_ctx = PluginContext(
            "mention_chat",
            {"agent": {"total_budget": 10.0, "max_turns": 10}},
            self.service,
            caps,
            member_store=self.members,
            reminder_store=self.reminder_store,
            plugin_active_check=lambda name: self.active_plugins.get(name, False),
        )
        self.mention_ctx._llm = self.llm

        self.reminder_plugin = ReminderPlugin(clock=lambda: self.now)
        self.reminder_plugin.on_load(self.reminder_ctx)
        self.addCleanup(self.reminder_plugin.on_unload, self.reminder_ctx)

        self.mention_plugin = MentionChatPlugin(clock=lambda: self.now)
        self.mention_plugin.on_load(self.mention_ctx)
        self.addCleanup(self.mention_plugin.on_unload, self.mention_ctx)

        self.scheduler = ReminderScheduler(
            self.reminder_store,
            self.members,
            lambda room, content, at_list, display_names: self.service.send_at_text(
                room, content, at_list, display_names=display_names
            ),
            is_active=lambda: self.active_plugins.get("reminder", False),
            ready=lambda: True,
            clock=lambda: self.now,
        )

    def dispatch_group(
        self,
        text: str,
        *,
        sender: str = ALICE,
        room: str = ROOM,
        at_self: bool = True,
        at_members: tuple | None = None,
        sequence: int = 1,
    ):
        """模拟 runtime._emit 的完整记录、意图标记与双插件广播流程。"""
        if at_members is None:
            at_members = (ROBOT,) if at_self else ()
        msg = Message(
            text=text,
            chat_id=room,
            sender_id=sender,
            is_group=True,
            is_self=(sender == ROBOT),
            sequence=sequence,
            session_id=self.backend.session_id,
            data={
                "dispatch_to_plugins": True,
                "bot_generated": False,
                "is_at_self": at_self,
                "at_members": at_members,
            },
        )
        self.members.record(msg)
        if self.active_plugins.get("reminder"):
            msg = prepare_reminder(
                msg,
                self.members,
                agent_fallback=self.active_plugins.get("mention_chat", False),
                now=self.now,
            )
        start = len(self.backend.sent)
        if self.active_plugins.get("mention_chat"):
            self.mention_plugin.on_message(msg, self.mention_ctx)
        if self.active_plugins.get("reminder"):
            self.reminder_plugin.on_message(msg, self.reminder_ctx)
        return msg, self.backend.sent[start:]


class ReminderToolSpecAndPermissionTests(ReminderAgentToolsTestBase):
    def test_factory_defaults_and_no_reserved_parameters(self):
        # 默认不开启提醒工具，保持 6 工具断言不变
        self.assertEqual(len(create_agent_tool_specs()), 6)
        self.assertEqual(len(create_agent_tool_registry()), 6)

        # 显式开启后为 9 工具
        specs_with_reminders = create_agent_tool_specs(include_reminders=True)
        self.assertEqual(len(specs_with_reminders), 9)
        reminder_specs = create_reminder_tool_specs()
        self.assertEqual(
            [s.name for s in reminder_specs],
            ["reminder_create", "reminder_list", "reminder_cancel"],
        )

        for spec in reminder_specs:
            props = spec.parameters_schema.get("properties", {})
            self.assertFalse(RESERVED_PARAMETERS.intersection(props))
            self.assertNotIn("creator", props)
            self.assertIs(spec.parameters_schema.get("additionalProperties"), False)

        # memory_* 保持 privileged_only=True；reminder_create/cancel 为 write=True, privileged_only=False
        reg = create_agent_tool_registry(include_reminders=True)
        self.assertTrue(reg.get("memory_save").privileged_only)
        self.assertTrue(reg.get("memory_delete").privileged_only)
        self.assertTrue(reg.get("reminder_create").write)
        self.assertFalse(reg.get("reminder_create").privileged_only)
        self.assertFalse(reg.get("reminder_list").write)
        self.assertTrue(reg.get("reminder_cancel").write)
        self.assertFalse(reg.get("reminder_cancel").privileged_only)

    def test_mention_chat_gracefully_disables_reminder_tools_without_store(self):
        ctx_no_rem = PluginContext(
            "mention_chat",
            {},
            self.service,
            ("text.receive", "text.send", "group.text.receive", "group.text.send", "group.text.mention"),
            member_store=self.members,
            reminder_store=None,
        )
        ctx_no_rem._llm = self.llm
        plugin = MentionChatPlugin(clock=lambda: self.now)
        plugin.on_load(ctx_no_rem)
        self.addCleanup(plugin.on_unload, ctx_no_rem)
        self.assertIsNone(plugin.reminder_store)
        self.assertEqual(len(plugin.tool_registry), 6)

    def test_non_vip_can_use_reminder_write_tools_but_counted_in_max_write_calls(self):
        reg = create_agent_tool_registry(include_reminders=True)
        ctx = ToolContext(
            account=ROBOT,
            room=ROOM,
            sender_id=ALICE,
            is_vip=False,
            is_self=False,
            is_group=True,
            store=self.members,
            extras={
                "reminder_store": self.reminder_store,
                "now_fn": lambda: self.now,
            },
        )
        executor = ToolExecutor(reg, ctx, AgentConfig(max_write_calls=2))

        # 普通群友调用 memory_save 仍被拦截为无权限
        mem_res = json.loads(
            executor.execute({"id": "m1", "name": "memory_save", "arguments": {"action": "add", "content": "越权"}})
        )
        self.assertEqual(mem_res, {"ok": False, "error": "无权限"})
        self.assertEqual(executor.write_calls, 0)

        # 第 1 次本人提醒写操作：成功
        r1 = json.loads(
            executor.execute(
                {"id": "r1", "name": "reminder_create", "arguments": {"content": "喝水", "delay_minutes": 10}}
            )
        )
        self.assertTrue(r1["ok"])
        self.assertEqual(executor.write_calls, 1)

        # 第 2 次本人提醒写操作（取消）：成功
        r2 = json.loads(
            executor.execute({"id": "r2", "name": "reminder_cancel", "arguments": {"number": 1}})
        )
        self.assertTrue(r2["ok"])
        self.assertEqual(executor.write_calls, 2)

        # 第 3 次本人提醒写操作：达到单轮 max_write_calls=2 上限被拦截
        r3 = json.loads(
            executor.execute(
                {"id": "r3", "name": "reminder_create", "arguments": {"content": "再喝水", "delay_minutes": 15}}
            )
        )
        self.assertEqual(r3, {"ok": False, "error": "本次对话的写入次数已达上限"})

    def test_system_prompt_injects_beijing_now_only_when_provided(self):
        plain = build_system_prompt()
        self.assertNotIn("当前北京时间", plain)

        now_str = format_beijing_now(BASE_NOW)
        self.assertEqual(now_str, "当前北京时间：2026-10-24 15:00（星期六）")
        with_now = build_system_prompt(now_text=now_str, persona="简短说话")
        self.assertIn("当前北京时间：2026-10-24 15:00（星期六）", with_now)
        self.assertLess(with_now.index("当前北京时间"), with_now.index("当前人格"))


class ColloquialCreateAndSchedulerDeliveryTests(ReminderAgentToolsTestBase):
    def test_colloquial_create_and_scheduler_tick_at_creator(self):
        # 1. “一刻钟后提醒我关火”（含“提醒”，但规则落入 USAGE_TIP -> 走 Agent 兜底）
        self.llm.responses = [
            tool_chat_result(make_tool_call("reminder_create", {"content": "关火", "delay_minutes": 15}, "c1")),
            text_chat_result("好嘞！已设置提醒 [#1]：2026-10-24 15:15 提醒你「关火」。"),
        ]
        msg1, sent1 = self.dispatch_group(AT_ROBOT + "一刻钟后提醒我关火", sender=ALICE, sequence=1)
        self.assertFalse(msg1.data.get("reminder_intent"))
        self.assertEqual(len(self.llm.requests), 2)
        # 验证系统提示词注入了当前北京时间与星期
        sys_prompt = self.llm.requests[0]["messages"][0]["content"]
        self.assertIn("当前北京时间：2026-10-24 15:00（星期六）", sys_prompt)
        # 只回复 1 条，绝不双回
        self.assertEqual(len(sent1), 1)
        self.assertIn("[#1]", sent1[0][1])
        self.assertIn("2026-10-24 15:15", sent1[0][1])

        # 2. “半小时后叫我拿快递”（不含“提醒”词 -> 原本就走 Agent）
        self.llm.responses = [
            tool_chat_result(make_tool_call("reminder_create", {"content": "拿快递", "delay_minutes": 30}, "c2")),
            text_chat_result("没问题，已设置提醒 [#2]：2026-10-24 15:30 提醒你「拿快递」。"),
        ]
        _, sent2 = self.dispatch_group(AT_ROBOT + "半小时后叫我拿快递", sender=ALICE, sequence=2)
        self.assertEqual(len(sent2), 1)
        self.assertIn("[#2]", sent2[0][1])

        # 3. “下周五15:00提醒我交材料”（跨周绝对时间 due_at="2026-10-30 15:00"）
        self.llm.responses = [
            tool_chat_result(
                make_tool_call("reminder_create", {"content": "交材料", "due_at": "2026-10-30 15:00"}, "c3")
            ),
            text_chat_result("已设置提醒 [#3]：2026-10-30 15:00 提醒你「交材料」。"),
        ]
        msg3, sent3 = self.dispatch_group(AT_ROBOT + "下周五15:00提醒我交材料", sender=BOB, sequence=3)
        self.assertFalse(msg3.data.get("reminder_intent"))
        self.assertEqual(len(sent3), 1)
        self.assertEqual(len(self.reminder_store.list_pending(ROOM, BOB)), 1)

        # 4. 推进时钟到 15:15:05，验证 ReminderScheduler.tick 准点 @ 创建者本人（绑定称呼“阿狸”）
        self.now = BASE_NOW + 15 * 60 + 5
        before_tick = len(self.backend.sent)
        self.assertEqual(self.scheduler.tick(), 1)
        fired = self.backend.sent[before_tick:]
        self.assertEqual(len(fired), 1)
        self.assertEqual(fired[0][0], ROOM)
        self.assertEqual(fired[0][2], (ALICE,))
        self.assertEqual(fired[0][1], f"@阿狸{AT}\n⏰ 提醒时间到：关火（设定时间 15:15）")

        # 推进到下周五 2026-10-30 15:00:10，Bob 的提醒准点触发（回退微信名“大壮”），Alice 的 #2 超宽限期静默作废
        self.now = datetime(2026, 10, 30, 15, 0, 10, tzinfo=BEIJING).timestamp()
        before_tick2 = len(self.backend.sent)
        self.assertEqual(self.scheduler.tick(), 1)
        fired2 = self.backend.sent[before_tick2:]
        self.assertEqual(len(fired2), 1)
        self.assertEqual(fired2[0][2], (BOB,))
        self.assertEqual(fired2[0][1], f"@大壮{AT}\n⏰ 提醒时间到：交材料（设定时间 15:00）")

    def test_iso_timezone_due_at_parsing(self):
        reg = create_agent_tool_registry(include_reminders=True)
        ctx = ToolContext(
            account=ROBOT,
            room=ROOM,
            sender_id=ALICE,
            is_group=True,
            extras={"reminder_store": self.reminder_store, "now_fn": lambda: self.now},
        )
        executor = ToolExecutor(reg, ctx, AgentConfig())

        # UTC 08:00Z == 北京时间 16:00
        res = json.loads(
            executor.execute(
                {
                    "id": "iso1",
                    "name": "reminder_create",
                    "arguments": {"content": "跨时区会议", "due_at": "2026-10-24T08:00:00Z"},
                }
            )
        )
        self.assertTrue(res["ok"])
        self.assertEqual(res["due_at_beijing"], "2026-10-24 16:00")


class MultiTurnAndListCancelTests(ReminderAgentToolsTestBase):
    def test_multi_turn_follow_up_and_colloquial_list_and_cancel(self):
        # 第 1 轮：用户只说事项未给时间（落入 Agent 反问）
        self.llm.responses = [text_chat_result("你想让我几点或者多久后提醒你拿奶茶外卖呢？")]
        msg_turn1, sent_t1 = self.dispatch_group(AT_ROBOT + "提醒我一下刚才说的拿奶茶外卖", sender=ALICE, sequence=1)
        self.assertFalse(msg_turn1.data.get("reminder_intent"))
        self.assertEqual(len(sent_t1), 1)

        # 第 2 轮：用户只补充“20分钟后吧”，模型凭借第 1 轮历史调用 reminder_create
        def verify_history_and_create(messages, tools):
            roles = [m["role"] for m in messages]
            self.assertEqual(roles, ["system", "user", "assistant", "user"])
            self.assertIn("拿奶茶外卖", messages[1]["content"])
            return tool_chat_result(
                make_tool_call("reminder_create", {"content": "拿奶茶外卖", "delay_minutes": 20}, "c_t2")
            )

        self.llm.responses = [
            verify_history_and_create,
            text_chat_result("好的，已设置提醒 [#1]：2026-10-24 15:20 提醒你「拿奶茶外卖」。"),
        ]
        _, sent_t2 = self.dispatch_group(AT_ROBOT + "20分钟后吧", sender=ALICE, sequence=2)
        self.assertEqual(len(sent_t2), 1)
        self.assertEqual(len(self.reminder_store.list_pending(ROOM, ALICE)), 1)

        # 同时给 Bob 在同群创建一条提醒，验证口语 list / cancel 严格只见本人
        self.reminder_store.create(ROOM, BOB, self.now + 1800, "大壮的私密事项", now=self.now)

        # 口语查看：“帮我看看我设了哪些提醒”
        captured_list_tool_output = []

        def check_list_result(messages, tools):
            tool_msg = next(m for m in messages if m["role"] == "tool")
            captured_list_tool_output.append(json.loads(tool_msg["content"]))
            return text_chat_result("你当前有 1 条提醒：#1 15:20 拿奶茶外卖。")

        self.llm.responses = [
            tool_chat_result(make_tool_call("reminder_list", {}, "c_list")),
            check_list_result,
        ]
        _, sent_list = self.dispatch_group(AT_ROBOT + "帮我看看我设了哪些提醒", sender=ALICE, sequence=3)
        self.assertEqual(len(sent_list), 1)
        self.assertEqual(len(captured_list_tool_output), 1)
        list_payload = captured_list_tool_output[0]
        self.assertEqual(
            list_payload,
            {
                "ok": True,
                "reminders": [{"number": 1, "due_at_beijing": "2026-10-24 15:20", "content": "拿奶茶外卖"}],
                "count": 1,
                "limit": 5,
            },
        )

        # 口语取消：“把拿外卖那个提醒取消掉”
        self.llm.responses = [
            tool_chat_result(make_tool_call("reminder_cancel", {"number": 1}, "c_cancel")),
            text_chat_result("已为你取消提醒 [#1]（2026-10-24 15:20 拿奶茶外卖）。"),
        ]
        _, sent_cancel = self.dispatch_group(AT_ROBOT + "把拿外卖那个提醒取消掉", sender=ALICE, sequence=4)
        self.assertEqual(len(sent_cancel), 1)
        self.assertEqual(self.reminder_store.list_pending(ROOM, ALICE), [])
        # Bob 的提醒不受影响
        self.assertEqual(len(self.reminder_store.list_pending(ROOM, BOB)), 1)


class SecurityIsolationAndPolicyTests(ReminderAgentToolsTestBase):
    def test_forged_parameters_and_cross_member_room_isolation(self):
        # Alice 在群 A 建一条 #1，群 B 建一条 #1
        self.reminder_store.create(ROOM, ALICE, self.now + 600, "群A事项", now=self.now)
        self.reminder_store.create(OTHER_ROOM, ALICE, self.now + 900, "群B事项", now=self.now)

        reg = create_agent_tool_registry(include_reminders=True)
        bob_ctx = ToolContext(
            account=ROBOT,
            room=ROOM,
            sender_id=BOB,
            is_vip=True,  # 即便是 VIP 也绝不能越权改他人/他群提醒
            is_group=True,
            extras={"reminder_store": self.reminder_store, "now_fn": lambda: self.now},
        )
        executor = ToolExecutor(reg, bob_ctx, AgentConfig(max_write_calls=5))

        # 1. 模型伪造传入 sender_id / room / creator / account 等参数 -> 全部被 schema 拦截
        for bad_arg in (
            {"number": 1, "sender_id": ALICE},
            {"number": 1, "room": OTHER_ROOM},
            {"number": 1, "creator": ALICE},
            {"number": 1, "account": "wxid_other"},
        ):
            res = json.loads(executor.execute({"id": "bad", "name": "reminder_cancel", "arguments": bad_arg}))
            self.assertFalse(res["ok"])
            self.assertIn("不支持的参数", res["error"])

        # 2. Bob 正常调用 reminder_cancel(number=1) -> 只能查自己，无法取消 Alice 的 #1
        res_bob_cancel = json.loads(
            executor.execute({"id": "b1", "name": "reminder_cancel", "arguments": {"number": 1}})
        )
        self.assertFalse(res_bob_cancel["ok"])
        self.assertIn("未找到你的待触发提醒 #1", res_bob_cancel["error"])
        self.assertEqual(len(self.reminder_store.list_pending(ROOM, ALICE)), 1)

        # 3. Bob 调用 cancel_all=True -> 不会清掉同群 Alice 的提醒
        res_bob_all = json.loads(
            executor.execute({"id": "b2", "name": "reminder_cancel", "arguments": {"cancel_all": True}})
        )
        self.assertTrue(res_bob_all["ok"])
        self.assertEqual(res_bob_all["cancelled_count"], 0)
        self.assertEqual(len(self.reminder_store.list_pending(ROOM, ALICE)), 1)

        # 4. Alice 在群 A 调用 cancel_all=True -> 只清群 A，不影响群 B
        alice_room_a = ToolContext(
            account=ROBOT,
            room=ROOM,
            sender_id=ALICE,
            is_group=True,
            extras={"reminder_store": self.reminder_store, "now_fn": lambda: self.now},
        )
        exec_alice = ToolExecutor(reg, alice_room_a, AgentConfig())
        res_alice_all = json.loads(
            exec_alice.execute({"id": "a1", "name": "reminder_cancel", "arguments": {"cancel_all": True}})
        )
        self.assertEqual(res_alice_all["cancelled_count"], 1)
        self.assertEqual(self.reminder_store.list_pending(ROOM, ALICE), [])
        self.assertEqual(len(self.reminder_store.list_pending(OTHER_ROOM, ALICE)), 1)

    def test_sanitization_of_ids_in_tool_outputs(self):
        reg = create_agent_tool_registry(include_reminders=True)
        ctx = ToolContext(
            account=ROBOT,
            room=ROOM,
            sender_id=ALICE,
            is_group=True,
            extras={"reminder_store": self.reminder_store, "now_fn": lambda: self.now},
        )
        executor = ToolExecutor(reg, ctx, AgentConfig())
        create_raw = executor.execute(
            {
                "id": "s1",
                "name": "reminder_create",
                "arguments": {"content": "联系 wxid_secret_99 在 88888@chatroom", "delay_minutes": 10},
            }
        )
        list_raw = executor.execute({"id": "s2", "name": "reminder_list", "arguments": {}})
        for output in (create_raw, list_raw):
            self.assertNotIn("wxid_secret_99", output)
            self.assertNotIn("88888@chatroom", output)
            self.assertIn("[微信ID]", output)
            self.assertIn("[群聊]", output)

    def test_policy_rejections_other_mentions_periodic_past_date_only_and_quotas(self):
        # 1. 不含“提醒”但同时 @ 了其他群友 -> ToolContext.has_other_mentions=True 拦截
        captured_errors = []

        def capture_and_reply(messages, tools):
            tool_msg = next(m for m in messages if m["role"] == "tool")
            captured_errors.append(json.loads(tool_msg["content"]))
            return text_chat_result("抱歉，只能为你自己设置提醒。")

        self.llm.responses = [
            tool_chat_result(make_tool_call("reminder_create", {"content": "开会", "delay_minutes": 15}, "o1")),
            capture_and_reply,
        ]
        _, sent = self.dispatch_group(
            f"@机器人{AT}@大壮{AT} 半小时后喊他开会",
            sender=ALICE,
            at_members=(ROBOT, BOB),
            sequence=1,
        )
        self.assertEqual(len(sent), 1)
        self.assertEqual(captured_errors[-1], {"ok": False, "error": ONLY_SELF_TIP})
        self.assertEqual(self.reminder_store.list_pending(ROOM, ALICE), [])
        self.assertEqual(self.reminder_store.list_pending(ROOM, BOB), [])

        # 2. 直接测试工具层的周期词、仅日期无时刻、已过时间、过短/超长、参数冲突与配额上限
        reg = create_agent_tool_registry(include_reminders=True)
        ctx = ToolContext(
            account=ROBOT,
            room=ROOM,
            sender_id=ALICE,
            is_group=True,
            extras={"reminder_store": self.reminder_store, "now_fn": lambda: self.now},
        )
        executor = ToolExecutor(reg, ctx, AgentConfig(max_write_calls=10))

        # 周期词拒绝
        res_periodic = json.loads(
            executor.execute(
                {"id": "p1", "name": "reminder_create", "arguments": {"content": "每天打卡", "delay_minutes": 10}}
            )
        )
        self.assertEqual(res_periodic, {"ok": False, "error": NO_REPEAT_TIP})

        # 仅日期无时刻拒绝（禁止静默当成 00:00）
        res_date_only = json.loads(
            executor.execute(
                {"id": "p2", "name": "reminder_create", "arguments": {"content": "交周报", "due_at": "2026-10-25"}}
            )
        )
        self.assertEqual(res_date_only, {"ok": False, "error": SUPPLEMENT_TIME_TIP})

        # 已过时间拒绝（当前 15:00，目标 14:00）
        res_past = json.loads(
            executor.execute(
                {
                    "id": "p3",
                    "name": "reminder_create",
                    "arguments": {"content": "开会", "due_at": "2026-10-24 14:00"},
                }
            )
        )
        self.assertFalse(res_past["ok"])
        self.assertIn("目标时间（2026-10-24 14:00）已过去", res_past["error"])

        # 过短（0 分钟）与超过 30 天
        res_zero = json.loads(
            executor.execute(
                {"id": "p4", "name": "reminder_create", "arguments": {"content": "喝水", "delay_minutes": 0}}
            )
        )
        self.assertIn("至少在 1 分钟之后", res_zero["error"])

        res_far = json.loads(
            executor.execute(
                {
                    "id": "p5",
                    "name": "reminder_create",
                    "arguments": {"content": "续费", "delay_minutes": 31 * 24 * 60},
                }
            )
        )
        self.assertIn("最长支持未来 30 天内", res_far["error"])

        # 二选一冲突与缺失
        res_both = json.loads(
            executor.execute(
                {
                    "id": "p6",
                    "name": "reminder_create",
                    "arguments": {"content": "喝水", "delay_minutes": 10, "due_at": "2026-10-24 16:00"},
                }
            )
        )
        self.assertFalse(res_both["ok"])
        res_neither = json.loads(
            executor.execute({"id": "p7", "name": "reminder_create", "arguments": {"content": "喝水"}})
        )
        self.assertFalse(res_neither["ok"])

        # 个人配额上限 5/5
        for idx in range(1, 6):
            self.reminder_store.create(ROOM, ALICE, self.now + idx * 300, f"事项{idx}", now=self.now)
        res_member_limit = json.loads(
            executor.execute(
                {"id": "p8", "name": "reminder_create", "arguments": {"content": "第6条", "delay_minutes": 20}}
            )
        )
        self.assertEqual(
            res_member_limit,
            {"ok": False, "error": "你在本群的待触发提醒已达上限（5/5），请先取消部分提醒后再试。"},
        )

        # 全群配额上限 30/30
        for u in range(1, 6):
            for m in range(1, 6):
                self.reminder_store.create(ROOM, f"wxid_extra_{u}", self.now + 600 + m * 60, f"占{m}", now=self.now)
        bob_ctx = ToolContext(
            account=ROBOT,
            room=ROOM,
            sender_id=BOB,
            is_group=True,
            extras={"reminder_store": self.reminder_store, "now_fn": lambda: self.now},
        )
        res_room_limit = json.loads(
            ToolExecutor(reg, bob_ctx, AgentConfig()).execute(
                {"id": "p9", "name": "reminder_create", "arguments": {"content": "第31条", "delay_minutes": 10}}
            )
        )
        self.assertEqual(
            res_room_limit,
            {"ok": False, "error": "本群待触发提醒总数已达上限（30/30），请稍后再试。"},
        )

    def test_private_vip_and_filehelper_rejected_and_inactive_reminder_plugin_rejected(self):
        # 1. 私聊 VIP 尝试调用 reminder_create / list / cancel
        seen_private = []

        def inspect_private(messages, tools):
            for m in messages:
                if m["role"] == "tool":
                    seen_private.append(json.loads(m["content"]))
            return text_chat_result("私聊不支持提醒功能哦。")

        self.llm.responses = [
            tool_chat_result(
                make_tool_call("reminder_create", {"content": "喝水", "delay_minutes": 10}, "pv1"),
                make_tool_call("reminder_list", {}, "pv2"),
                make_tool_call("reminder_cancel", {"cancel_all": True}, "pv3"),
            ),
            inspect_private,
        ]
        vip_msg = Message(
            text="10分钟后提醒我喝水",
            chat_id=VIP_USER,
            sender_id=VIP_USER,
            is_group=False,
            session_id=self.backend.session_id,
            data={"bot_generated": False},
        )
        self.mention_plugin.on_message(vip_msg, self.mention_ctx)
        self.assertEqual(
            seen_private,
            [{"ok": False, "error": REMINDER_GROUP_ONLY_ERROR}] * 3,
        )

        # 2. 文件传输助手 /ai 尝试调用 reminder_create
        seen_fh = []

        def inspect_fh(messages, tools):
            tool_msg = next(m for m in messages if m["role"] == "tool")
            seen_fh.append(json.loads(tool_msg["content"]))
            return text_chat_result("文件助手不支持群提醒。")

        self.llm.responses = [
            tool_chat_result(make_tool_call("reminder_create", {"content": "开会", "delay_minutes": 10}, "fh1")),
            inspect_fh,
        ]
        fh_msg = Message(
            text="/ai 10分钟后提醒我开会",
            chat_id="filehelper",
            sender_id=ROBOT,
            is_group=False,
            is_self=True,
            session_id=self.backend.session_id,
            data={"bot_generated": False},
        )
        self.mention_plugin.on_message(fh_msg, self.mention_ctx)
        self.assertEqual(seen_fh, [{"ok": False, "error": REMINDER_GROUP_ONLY_ERROR}])

        # 3. 群聊中若 reminder 插件已被 disable 停用，reminder_create 拒绝创建
        self.active_plugins["reminder"] = False
        seen_disabled = []

        def inspect_disabled(messages, tools):
            tool_msg = next(m for m in messages if m["role"] == "tool")
            seen_disabled.append(json.loads(tool_msg["content"]))
            return text_chat_result("提醒插件当前已停用，无法设置提醒。")

        self.llm.responses = [
            tool_chat_result(make_tool_call("reminder_create", {"content": "喝水", "delay_minutes": 10}, "dis1")),
            inspect_disabled,
        ]
        self.dispatch_group(AT_ROBOT + "一刻钟后提醒我喝水", sender=ALICE, sequence=10)
        self.assertEqual(len(seen_disabled), 1)
        self.assertFalse(seen_disabled[0]["ok"])
        self.assertIn("提醒插件已停用", seen_disabled[0]["error"])
        self.assertEqual(self.reminder_store.list_pending(ROOM, ALICE), [])


class PersonaTruncationAndRoutingCoexistenceTests(ReminderAgentToolsTestBase):
    def test_zuichou_persona_does_not_truncate_reminder_tool_replies(self):
        self.members.set_personality(ROOM, "嘴臭")

        # 1. 调用了 reminder_create 的回复：超过 20 字且含 [#1] 与 2026-10-24，绝不截断、不抹掉 # 和 -
        full_confirmation = "行了记上了！已设置提醒 [#1]：2026-10-24 15:15 提醒你「关火」，别把锅烧穿了！"
        self.llm.responses = [
            tool_chat_result(make_tool_call("reminder_create", {"content": "关火", "delay_minutes": 15}, "z1")),
            text_chat_result(full_confirmation),
        ]
        _, sent_rem = self.dispatch_group(AT_ROBOT + "一刻钟后提醒我关火", sender=ALICE, sequence=1)
        self.assertEqual(len(sent_rem), 1)
        self.assertEqual(sent_rem[0][1], f"@阿狸{AT}\n{full_confirmation}")

        # 2. 对照：未调用 reminder_* 工具的普通闲聊，仍然严格按嘴臭人格截断至 <= 20 字
        self.llm.responses = [
            text_chat_result("这是一段超过二十个汉字的普通嘴臭闲聊回复应该被严格截断掉后面的尾巴")
        ]
        _, sent_chat = self.dispatch_group(AT_ROBOT + "今天吃什么好呢", sender=ALICE, sequence=2)
        self.assertEqual(len(sent_chat), 1)
        body = sent_chat[0][1].split("\n", 1)[1]
        self.assertLessEqual(len(body), 20)

    def test_rule_first_zero_llm_overhead_and_no_mention_chat_fallback(self):
        # 1. mention_chat 开启时，规则可解析的 "@机器人 5分钟后提醒我喝水" 仍走规则：0 次 LLM、只回 1 条
        msg_rule, sent_rule = self.dispatch_group(AT_ROBOT + "5分钟后提醒我喝水", sender=ALICE, sequence=1)
        self.assertTrue(msg_rule.data.get("reminder_intent"))
        self.assertEqual(len(self.llm.requests), 0)
        self.assertEqual(len(sent_rule), 1)
        self.assertIn("已设置提醒 [#1]：2026-10-24 15:05 提醒你「喝水」", sent_rule[0][1])

        # 2. mention_chat 开启时，命中明确拒绝策略（周期词、已过时间、仅日期、替他人）仍走规则：0 次 LLM、只回 1 条
        for idx, rejected_text in enumerate(
            (
                AT_ROBOT + "每天08:00提醒我打卡",
                AT_ROBOT + "今天14:00提醒我开会",
                AT_ROBOT + "明天提醒我交周报",
                AT_ROBOT + "5分钟后提醒大家开会",
            ),
            start=2,
        ):
            msg_rej, sent_rej = self.dispatch_group(rejected_text, sender=ALICE, sequence=idx)
            self.assertTrue(msg_rej.data.get("reminder_intent"), rejected_text)
            self.assertEqual(len(self.llm.requests), 0)
            self.assertEqual(len(sent_rej), 1)

        # 3. 斜杠命令 "/提醒 一刻钟后 关火" 即使规则无法解析，也打 reminder_intent=True，0 次 LLM、返回 USAGE_TIP
        msg_slash, sent_slash = self.dispatch_group("/提醒 一刻钟后 关火", sender=ALICE, at_self=False, sequence=10)
        self.assertTrue(msg_slash.data.get("reminder_intent"))
        self.assertEqual(len(self.llm.requests), 0)
        self.assertEqual(len(sent_slash), 1)
        self.assertEqual(sent_slash[0][1], USAGE_TIP)

        # 4. mention_chat 未启用（或无模型未激活）时，"@机器人 一刻钟后提醒我关火" 保持回复 USAGE_TIP 不变
        self.active_plugins["mention_chat"] = False
        msg_no_agent, sent_no_agent = self.dispatch_group(AT_ROBOT + "一刻钟后提醒我关火", sender=ALICE, sequence=11)
        self.assertTrue(msg_no_agent.data.get("reminder_intent"))
        self.assertEqual(len(self.llm.requests), 0)
        self.assertEqual(len(sent_no_agent), 1)
        self.assertEqual(sent_no_agent[0][1], USAGE_TIP)


if __name__ == "__main__":
    unittest.main()
