"""Agent 入口全链路验收测试：覆盖 AC-2、AC-7 与 AC-8 及各项命令与隔离。

- AC-2：多群与多账号隔离、/清空、跨天，且长期记忆保留；/重置对话 单人重置。
- AC-7：私聊 VIP 触发（回复无 @）、非 VIP 忽略。
- AC-8：文件传输助手 /ai 触发（剥离前缀、回复无 @）、非 /ai 忽略、bot_generated 排除防死循环。
- AC-9：语气词轮不占窗口、超长回复按 1023 字节切分。
- 命令鉴权与安全：/删除记忆3 无空格不触发写操作。
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from plugins.mention_chat import Plugin as MentionChatPlugin
from wxbot_framework.agent import (
    AgentConfig,
    SessionKey,
)
from wxbot_framework.agent_tools import create_agent_tool_registry
from wxbot_framework.llm import ChatResult
from wxbot_framework.members import MemberStore
from wxbot_framework.memory import GroupMemoryStore
from wxbot_framework.sdk import Message, PluginContext, ServiceFacade

ACCOUNT = "wxid_robot_main"
ACCOUNT_OTHER = "wxid_robot_other"
ROOM_1 = "11111@chatroom"
ROOM_2 = "22222@chatroom"
VIP_USER = "wxid_vip_boss"
NORMAL_USER = "wxid_normal_user"
OTHER_USER = "wxid_other_user"


class MockBackend:
    session_id = "test_session_1"

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
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []
        self.enable_tools = True

    def complete_chat(self, messages, *, tools=None, tool_choice="auto"):
        self.requests.append({"messages": messages, "tools": tools})
        if not self.responses:
            return ChatResult("默认假模型回复")
        item = self.responses.pop(0)
        if callable(item):
            return item(messages, tools)
        if isinstance(item, Exception):
            raise item
        return item


class AgentEntryTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.runtime_dir = self.root / ".runtime"
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.runtime_dir / "member_data.sqlite3"

        # 配置文件
        self.config_file = self.root / "members_config.json"
        config_data = {
            "database_file": ".runtime/member_data.sqlite3",
            "history_enabled": True,
            "daily_clear": True,
            "agent_vip_wxids": [VIP_USER],
            "groups": {
                ROOM_1: [
                    {"key": "vip", "name": "VIP老板", "display_name": "VIP老板", "wxid": VIP_USER},
                    {"key": "alice", "name": "小明", "display_name": "小明同学", "wxid": NORMAL_USER},
                    {"key": "bob", "name": "小李", "display_name": "小李同学", "wxid": OTHER_USER},
                ],
                ROOM_2: [
                    {"key": "vip2", "name": "VIP老板", "display_name": "VIP老板", "wxid": VIP_USER},
                    {"key": "charlie", "name": "张三", "display_name": "张三同学", "wxid": NORMAL_USER},
                ],
            },
        }
        self.config_file.write_text(json.dumps(config_data, ensure_ascii=False), encoding="utf-8")

        self.member_store = MemberStore(self.config_file, ACCOUNT)
        self.member_store.initialize()

        self.backend = MockBackend()
        self.service = ServiceFacade(text_delay={"enabled": False})
        self.service.bind(self.backend)

        self.llm = FakeLLM()
        self.ctx = PluginContext(
            "mention_chat",
            {"agent": {"total_budget": 10.0, "max_turns": 10}},
            self.service,
            ["text.receive", "text.send", "friend.text.receive", "friend.text.send",
             "group.text.receive", "group.text.send", "group.text.mention"],
            member_store=self.member_store,
        )
        self.ctx._llm = self.llm

        self.plugin = MentionChatPlugin()
        self.plugin.on_load(self.ctx)

    def tearDown(self):
        self.plugin.on_unload(self.ctx)
        self.service.close()
        self.member_store.close()
        self.temp_dir.cleanup()

    def record_and_make_group_at(self, text, *, sender=NORMAL_USER, room=ROOM_1, sequence=10):
        # 先记录到聊天缓存，以便 conversation_snapshot 获取
        msg = Message(
            text=f"@机器人\u2005{text}",
            chat_id=room,
            sender_id=sender,
            is_group=True,
            session_id="test_session_1",
            sequence=sequence,
            data={
                "bot_generated": False,
                "is_at_self": True,
                "at_members": (ACCOUNT,),
            },
        )
        self.member_store.record(msg)
        return msg

    def test_ac2_multi_room_and_account_isolation_and_clear_and_rollover(self):
        """AC-2：多群与多账号隔离、/清空、跨天，且长期记忆保留；/重置对话 单人重置。"""
        # 1. 在 Room 1 和 Room 2 中分别发生艾特对话
        self.llm.responses = [
            ChatResult("你好小明，我是群1机器人"),
            ChatResult("你好张三，我是群2机器人"),
        ]
        msg1 = self.record_and_make_group_at("我是小明，我在群1", room=ROOM_1, sequence=1)
        self.plugin.on_message(msg1, self.ctx)
        self.assertEqual(len(self.backend.sent), 1)
        self.assertIn("你好小明，我是群1机器人", self.backend.sent[-1][1])
        self.assertEqual(self.backend.sent[-1][0], ROOM_1)

        msg2 = self.record_and_make_group_at("我是张三，我在群2", room=ROOM_2, sequence=2)
        self.plugin.on_message(msg2, self.ctx)
        self.assertEqual(len(self.backend.sent), 2)
        self.assertIn("你好张三，我是群2机器人", self.backend.sent[-1][1])
        self.assertEqual(self.backend.sent[-1][0], ROOM_2)

        # 从 store 检查上下文
        snap1 = self.member_store.conversation_snapshot(msg1)
        snap2 = self.member_store.conversation_snapshot(msg2)
        history_r1 = self.plugin.conversation_store.history(
            SessionKey.group(ACCOUNT, ROOM_1, NORMAL_USER), (snap1.day, snap1.generation)
        )
        history_r2 = self.plugin.conversation_store.history(
            SessionKey.group(ACCOUNT, ROOM_2, NORMAL_USER), (snap2.day, snap2.generation)
        )
        self.assertEqual(len(history_r1), 1)
        self.assertEqual(len(history_r2), 1)
        self.assertEqual(history_r1[0].user, "我是小明，我在群1")
        self.assertEqual(history_r2[0].user, "我是张三，我在群2")

        # 2. 长期记忆写入与跨群隔离
        self.plugin.memory_store.add(ACCOUNT, ROOM_1, "群1周五聚餐")
        self.plugin.memory_store.add(ACCOUNT, ROOM_2, "群2禁止发广告")
        self.assertEqual(len(self.plugin.memory_store.list_memories(ACCOUNT, ROOM_1)), 1)
        self.assertEqual(len(self.plugin.memory_store.list_memories(ACCOUNT, ROOM_2)), 1)
        self.assertEqual(self.plugin.memory_store.list_memories(ACCOUNT, ROOM_1)[0].content, "群1周五聚餐")
        self.assertEqual(self.plugin.memory_store.list_memories(ACCOUNT, ROOM_2)[0].content, "群2禁止发广告")

        # 3. 在 Room 1 执行 /清空
        clear_msg = Message("/清空", ROOM_1, NORMAL_USER, is_group=True, session_id="test_session_1")
        self.plugin.on_message(clear_msg, self.ctx)
        self.member_store.clear(ROOM_1)

        # 验证 Room 1 多轮上下文已被清空，但长期记忆依然保留
        history_r1_after = self.plugin.conversation_store.load(
            SessionKey.group(ACCOUNT, ROOM_1, NORMAL_USER), (snap1.day, snap1.generation + 1)
        ).turns
        self.assertEqual(len(history_r1_after), 0)
        # 长期记忆不受影响
        self.assertEqual(len(self.plugin.memory_store.list_memories(ACCOUNT, ROOM_1)), 1)
        self.assertEqual(self.plugin.memory_store.list_memories(ACCOUNT, ROOM_1)[0].content, "群1周五聚餐")

        # Room 2 的多轮上下文和长期记忆均不受影响
        history_r2_after = self.plugin.conversation_store.history(
            SessionKey.group(ACCOUNT, ROOM_2, NORMAL_USER), (snap2.day, snap2.generation)
        )
        self.assertEqual(len(history_r2_after), 1)
        self.assertEqual(len(self.plugin.memory_store.list_memories(ACCOUNT, ROOM_2)), 1)

        # 4. 模拟跨天（day rollover）
        self.plugin._active_day = "2020-01-01"
        self.plugin._check_rollover()
        # 跨天后，多轮对话上下文已全部释放，长期记忆依然完整
        self.assertEqual(len(self.plugin.conversation_store), 0)
        self.assertEqual(len(self.plugin.memory_store.list_memories(ACCOUNT, ROOM_1)), 1)
        self.assertEqual(len(self.plugin.memory_store.list_memories(ACCOUNT, ROOM_2)), 1)

        # 5. /重置对话（单人重置）
        # 让 NORMAL_USER 和 OTHER_USER 在 Room 2 分别建立上下文
        self.llm.responses = [ChatResult("回复小明"), ChatResult("回复小李")]
        m_u1 = self.record_and_make_group_at("小明新消息", sender=NORMAL_USER, room=ROOM_2, sequence=11)
        m_u2 = self.record_and_make_group_at("小李新消息", sender=OTHER_USER, room=ROOM_2, sequence=12)
        self.plugin.on_message(m_u1, self.ctx)
        self.plugin.on_message(m_u2, self.ctx)

        snap_u1 = self.member_store.conversation_snapshot(m_u1)
        snap_u2 = self.member_store.conversation_snapshot(m_u2)
        self.assertEqual(len(self.plugin.conversation_store.history(
            SessionKey.group(ACCOUNT, ROOM_2, NORMAL_USER), (snap_u1.day, snap_u1.generation)
        )), 1)
        self.assertEqual(len(self.plugin.conversation_store.history(
            SessionKey.group(ACCOUNT, ROOM_2, OTHER_USER), (snap_u2.day, snap_u2.generation)
        )), 1)

        # NORMAL_USER 发送 /重置对话
        reset_msg = Message("/重置对话", ROOM_2, NORMAL_USER, is_group=True, session_id="test_session_1")
        self.plugin.on_message(reset_msg, self.ctx)
        self.assertIn("已重置你与我的对话上下文", self.backend.sent[-1][1])

        # 验证 NORMAL_USER 上下文已重置，OTHER_USER 上下文完好保留
        self.assertEqual(len(self.plugin.conversation_store.history(
            SessionKey.group(ACCOUNT, ROOM_2, NORMAL_USER), (snap_u1.day, snap_u1.generation)
        )), 0)
        self.assertEqual(len(self.plugin.conversation_store.history(
            SessionKey.group(ACCOUNT, ROOM_2, OTHER_USER), (snap_u2.day, snap_u2.generation)
        )), 1)

    def test_ac7_private_vip_triggers_and_non_vip_ignored(self):
        """AC-7：私聊 VIP 触发、非 VIP 忽略。"""
        # 1. 非 VIP 发送私聊消息：被完全忽略
        non_vip_msg = Message("你好，可以认识一下吗", NORMAL_USER, NORMAL_USER, is_group=False, session_id="test_session_1")
        self.plugin.on_message(non_vip_msg, self.ctx)
        self.assertEqual(len(self.backend.sent), 0)
        self.assertEqual(len(self.llm.requests), 0)

        # 2. VIP 发送私聊消息：正常触发 Agent 且回复不带 @
        self.llm.responses = [ChatResult("老板好，随时为您服务！")]
        vip_msg = Message("今天有哪些待办事项？", VIP_USER, VIP_USER, is_group=False, session_id="test_session_1")
        self.plugin.on_message(vip_msg, self.ctx)
        self.assertEqual(len(self.backend.sent), 1)
        to_wxid, content, at_list = self.backend.sent[0]
        self.assertEqual(to_wxid, VIP_USER)
        self.assertEqual(content, "老板好，随时为您服务！")
        self.assertNotIn("@", content)
        self.assertEqual(at_list, ())

        # 3. VIP 在私聊中管理私聊专属记忆
        vip_mem_save = Message("/记忆", VIP_USER, VIP_USER, is_group=False, session_id="test_session_1")
        self.plugin.on_message(vip_mem_save, self.ctx)
        self.assertIn("本群暂无长期记忆", self.backend.sent[-1][1])

        # 非 VIP 发送记忆命令被静默忽略
        sent_count_before = len(self.backend.sent)
        non_vip_cmd = Message("/记忆", NORMAL_USER, NORMAL_USER, is_group=False, session_id="test_session_1")
        self.plugin.on_message(non_vip_cmd, self.ctx)
        self.assertEqual(len(self.backend.sent), sent_count_before)

    def test_ac8_filehelper_ai_trigger_and_no_self_loop(self):
        """AC-8：文件传输助手 /ai 触发且不会自触发循环。"""
        # 1. 主号本人发送不带 /ai 前缀的消息（如备忘录）：被忽略
        note_msg = Message(
            "备忘录：下午 3 点开会",
            "filehelper",
            ACCOUNT,
            is_group=False,
            session_id="test_session_1",
            is_self=True,
            data={"bot_generated": False},
        )
        self.plugin.on_message(note_msg, self.ctx)
        self.assertEqual(len(self.backend.sent), 0)
        self.assertEqual(len(self.llm.requests), 0)

        # 2. 主号本人发送带 /ai 前缀的消息：触发 Agent 循环，剥离前缀
        self.llm.responses = [ChatResult("根据计算结果，1 加 1 等于 2。")]
        ai_msg = Message(
            "/ai 算一下 1+1 等于几",
            "filehelper",
            ACCOUNT,
            is_group=False,
            session_id="test_session_1",
            is_self=True,
            data={"bot_generated": False},
        )
        self.plugin.on_message(ai_msg, self.ctx)
        self.assertEqual(len(self.backend.sent), 1)
        to_wxid, content, at_list = self.backend.sent[0]
        self.assertEqual(to_wxid, "filehelper")
        self.assertEqual(content, "根据计算结果，1 加 1 等于 2。")
        self.assertNotIn("@", content)

        # 检查发给大模型的问题已剥离 /ai 前缀
        sent_messages = self.llm.requests[-1]["messages"]
        user_message = next(m for m in sent_messages if m["role"] == "user")
        self.assertEqual(user_message["content"], "算一下 1+1 等于几")

        # 3. bot_generated 回显消息：即使带 /ai 前缀，也严格忽略，杜绝自发自收死循环
        loop_msg = Message(
            "/ai 回显的自己生成的消息",
            "filehelper",
            ACCOUNT,
            is_group=False,
            session_id="test_session_1",
            is_self=True,
            data={"bot_generated": True},
        )
        self.plugin.on_message(loop_msg, self.ctx)
        self.assertEqual(len(self.backend.sent), 1)  # 没有新增发送

        # 4. 伪造非本人发送到 filehelper：忽略
        fake_msg = Message(
            "/ai 伪造提问",
            "filehelper",
            NORMAL_USER,
            is_group=False,
            session_id="test_session_1",
            is_self=False,
            data={"bot_generated": False},
        )
        self.plugin.on_message(fake_msg, self.ctx)
        self.assertEqual(len(self.backend.sent), 1)

    def test_delete_memory_without_space_does_not_trigger_write(self):
        """/删除记忆3（没有空格）会被当作普通对话，不会误触发删除或写入操作。"""
        self.plugin.memory_store.add(ACCOUNT, ROOM_1, "第一条记忆")
        self.plugin.memory_store.add(ACCOUNT, ROOM_1, "第二条记忆")
        self.plugin.memory_store.add(ACCOUNT, ROOM_1, "第三条记忆")
        self.assertEqual(len(self.plugin.memory_store.list_memories(ACCOUNT, ROOM_1)), 3)

        # VIP 发送 /删除记忆3（无空格）：不会触发删除记忆命令
        cmd_msg = Message("/删除记忆3", ROOM_1, VIP_USER, is_group=True, session_id="test_session_1")
        self.plugin.on_message(cmd_msg, self.ctx)
        # 3 条记忆依然完好存在
        self.assertEqual(len(self.plugin.memory_store.list_memories(ACCOUNT, ROOM_1)), 3)

    def test_long_reply_byte_limit_split(self):
        """长回复超过 1023 字节按 UTF-8 字符边界切分。"""
        long_text = "这是一段很长的测试回复内容。" * 80
        self.llm.responses = [ChatResult(long_text)]
        msg = self.record_and_make_group_at("请长篇大论", room=ROOM_1, sequence=99)
        self.plugin.on_message(msg, self.ctx)

        # 验证发送了多个切片，且每个切片都在 1023 字节限制内
        self.assertGreater(len(self.backend.sent), 1)
        for target, text, at_list in self.backend.sent:
            self.assertEqual(target, ROOM_1)
            self.assertLessEqual(len(text.encode("utf-8")), 1023)
            self.assertEqual(at_list, (NORMAL_USER,))

    def test_agent_tool_calling_in_mention_chat_and_private(self):
        """Agent 在群聊艾特中成功调用工具，且私聊中群类只读工具返回非群会话。"""
        # 1. VIP 在群聊艾特中让模型调用 memory_save
        self.llm.responses = [
            ChatResult(
                content="",
                tool_calls=[{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "memory_save", "arguments": json.dumps({"action": "add", "content": "每周二技术分享"})},
                    "name": "memory_save",
                    "arguments": {"action": "add", "content": "每周二技术分享"},
                }],
            ),
            ChatResult("已为您记住：每周二技术分享。"),
        ]
        vip_msg = self.record_and_make_group_at("记住：每周二技术分享", sender=VIP_USER, room=ROOM_1, sequence=101)
        self.plugin.on_message(vip_msg, self.ctx)
        self.assertIn("已为您记住", self.backend.sent[-1][1])
        # 验证长期记忆已被写入
        mems = self.plugin.memory_store.list_memories(ACCOUNT, ROOM_1)
        self.assertTrue(any("每周二技术分享" in m.content for m in mems))

        # 2. 私聊 VIP 触发时，若模型尝试调用 get_group_context，工具返回非群会话
        tool_executed = []
        def check_tool_reply(messages, tools):
            tool_msg = next((m for m in messages if m.get("role") == "tool"), None)
            if tool_msg:
                tool_executed.append(json.loads(tool_msg["content"]))
                return ChatResult("私聊无法查看群聊上下文")
            return ChatResult(
                content="",
                tool_calls=[{
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "get_group_context", "arguments": json.dumps({"limit": 5})},
                    "name": "get_group_context",
                    "arguments": {"limit": 5},
                }],
            )

        self.llm.responses = [check_tool_reply, check_tool_reply]
        p_msg = Message("帮我看看群聊说了什么", VIP_USER, VIP_USER, is_group=False, session_id="test_session_1")
        self.plugin.on_message(p_msg, self.ctx)
        self.assertEqual(len(tool_executed), 1)
        self.assertEqual(tool_executed[0], {"ok": False, "error": "非群会话"})
        self.assertIn("私聊无法查看群聊上下文", self.backend.sent[-1][1])

    def test_mention_request_cancels_when_personality_switched_during_call(self):
        """请求进行中切换人格，guard 失效，旧人格回复被丢弃且不发送。"""
        self.member_store.set_personality(ROOM_1, "嘴臭")

        def fake_complete(messages, tools=None):
            # 模型请求期间外部切换人格
            self.member_store.set_personality(ROOM_1, "卑微")
            return ChatResult("旧人格回复内容")

        self.llm.responses = [fake_complete]
        msg = self.record_and_make_group_at("请求中切换人格", room=ROOM_1, sequence=110)
        self.plugin.on_message(msg, self.ctx)

        # 验证没有发出任何消息
        self.assertEqual(len(self.backend.sent), 0)

    def test_mention_request_cancels_when_clear_during_call(self):
        """请求进行中群聊记录被清空，guard 失效，回复被丢弃且不发送。"""
        def fake_complete(messages, tools=None):
            # 模型请求期间外部执行群清空
            self.member_store.clear(ROOM_1)
            return ChatResult("清空前的回复内容")

        self.llm.responses = [fake_complete]
        msg = self.record_and_make_group_at("请求中清空记录", room=ROOM_1, sequence=111)
        self.plugin.on_message(msg, self.ctx)

        # 验证没有发出任何消息
        self.assertEqual(len(self.backend.sent), 0)

    def test_group_clear_command_with_arguments_does_not_clear_conversation(self):
        """群聊收到 /清空 xxx（带参数）时不清除多轮对话，只有单独的 /清空 才会清理。"""
        self.llm.responses = [ChatResult("第一轮回复")]
        msg1 = self.record_and_make_group_at("第一轮消息", room=ROOM_1, sequence=112)
        self.plugin.on_message(msg1, self.ctx)
        snap = self.member_store.conversation_snapshot(msg1)
        key = SessionKey.group(ACCOUNT, ROOM_1, NORMAL_USER)
        history = self.plugin.conversation_store.history(key, (snap.day, snap.generation))
        self.assertEqual(len(history), 1)

        # 发送带参数的 /清空 123：不应清空会话
        arg_clear = Message("/清空 123", ROOM_1, NORMAL_USER, is_group=True, session_id="test_session_1")
        self.plugin.on_message(arg_clear, self.ctx)
        self.assertEqual(len(self.plugin.conversation_store.history(key, (snap.day, snap.generation))), 1)

        # 发送标准 /清空：应清空该群会话
        exact_clear = Message("/清空", ROOM_1, NORMAL_USER, is_group=True, session_id="test_session_1")
        self.plugin.on_message(exact_clear, self.ctx)
        self.assertEqual(len(self.plugin.conversation_store.history(key, (snap.day, snap.generation))), 0)

    def test_empty_or_markdown_slop_reply_handles_safely(self):
        """空回复或经人格清洗后为空时安全捕获并降级提示，不崩溃。"""
        self.llm.responses = [ChatResult("")]
        msg = self.record_and_make_group_at("空回复测试", room=ROOM_1, sequence=113)
        self.plugin.on_message(msg, self.ctx)
        self.assertIn("暂时无法回复", self.backend.sent[-1][1])
