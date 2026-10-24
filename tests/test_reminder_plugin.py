"""reminder 插件、时间解析器与 mention_chat 协同测试。

覆盖：
1. 解析器表驱动测试（相对时间、当天时刻、跨天日期、中文数字、周期拦截、仅日期拦截、已过时间拦截、范围拦截、替他人拦截）；
2. 插件命令测试（创建、查看、取消单条、取消全部、配额上限、多群隔离、他人权限隔离）；
3. 与 mention_chat 共存不双回测试。
"""
from __future__ import annotations

import json
import logging
import tempfile
import time
import unittest
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from unittest.mock import Mock

from plugins.mention_chat import Plugin as MentionChatPlugin
from plugins.reminder import Plugin as ReminderPlugin
from plugins.reminder.parser import (
    ParsedReminder,
    ReminderError,
    ReminderParseError,
    ReminderTimeError,
    format_error,
    parse,
)
from wxbot_framework.members import BEIJING, MemberStore
from wxbot_framework.reminders import ReminderLimit, ReminderStore, prepare_reminder
from wxbot_framework.sdk import Message

ROOT = Path(__file__).resolve().parent.parent
SCRATCH = ROOT / ".runtime/tests"
ROOM, OTHER_ROOM = "100001@chatroom", "100002@chatroom"
ROBOT = "wxid_bot_demo"
ALICE, BOB = "wxid_alice_demo", "wxid_bob_demo"
# 固定测试基准时间：北京时间 2026-10-24 15:00:00
BASE_NOW = datetime(2026, 10, 24, 15, 0, 0, tzinfo=BEIJING)


class MockContext:
    def __init__(self, members, reminders):
        self.members = members
        self.reminders = reminders
        self.logger = logging.getLogger("MockContext")
        self.replies = []

    def reply(self, message, text, **kwargs):
        self.replies.append((message.chat_id, message.sender_id, text))
        return True


def make_group_message(text: str, sender: str = ALICE, room: str = ROOM,
                       is_at_self: bool = False, at_members: tuple = (),
                       reminder_intent: bool = False) -> Message:
    data = {
        "dispatch_to_plugins": True,
        "is_at_self": is_at_self,
        "at_members": at_members,
    }
    if reminder_intent:
        data["reminder_intent"] = True
    return Message(
        type=1,
        chat_id=room,
        sender_id=sender,
        text=text,
        is_group=True,
        data=MappingProxyType(data),
    )


class ReminderParserTestCase(unittest.TestCase):
    def test_relative_time_parsing(self):
        cases = [
            ("10分钟后提醒我关火", 10 * 60, "关火"),
            ("十分钟后喝水", 10 * 60, "喝水"),
            ("半小时后开会", 30 * 60, "开会"),
            ("半个小时后开会", 30 * 60, "开会"),
            ("1小时后出发", 3600, "出发"),
            ("一小时后出发", 3600, "出发"),
            ("一个小时后出发", 3600, "出发"),
            ("1小时30分钟后开会", 90 * 60, "开会"),
            ("一个半小时后开会", 90 * 60, "开会"),
            ("1小时半后开会", 90 * 60, "开会"),
            ("1天后交材料", 86400, "交材料"),
            ("三天后开会", 3 * 86400, "开会"),
        ]
        for text, expected_delta, expected_content in cases:
            with self.subTest(text=text):
                res = parse(text, BASE_NOW)
                self.assertEqual(res.content, expected_content)
                self.assertEqual(res.due_at, BASE_NOW.timestamp() + expected_delta)

    def test_absolute_time_parsing(self):
        cases = [
            ("15:30 喝水", datetime(2026, 10, 24, 15, 30, 0, tzinfo=BEIJING), "喝水"),
            ("20:30 提交周报", datetime(2026, 10, 24, 20, 30, 0, tzinfo=BEIJING), "提交周报"),
            ("下午3点半 开会", datetime(2026, 10, 24, 15, 30, 0, tzinfo=BEIJING), "开会"),
            ("下午4点 开会", datetime(2026, 10, 24, 16, 0, 0, tzinfo=BEIJING), "开会"),
            ("下午4点一起开会", datetime(2026, 10, 24, 16, 0, 0, tzinfo=BEIJING), "一起开会"),
            ("明天9点一定交报告", datetime(2026, 10, 25, 9, 0, 0, tzinfo=BEIJING), "一定交报告"),
            ("明天9点30 开会", datetime(2026, 10, 25, 9, 30, 0, tzinfo=BEIJING), "开会"),
            ("明天9点30分 开会", datetime(2026, 10, 25, 9, 30, 0, tzinfo=BEIJING), "开会"),
            ("明天9点提醒我检查每一个文件", datetime(2026, 10, 25, 9, 0, 0, tzinfo=BEIJING), "检查每一个文件"),
            ("晚上12点 睡觉", datetime(2026, 10, 25, 0, 0, 0, tzinfo=BEIJING), "睡觉"),
            ("今晚12点 睡觉", datetime(2026, 10, 25, 0, 0, 0, tzinfo=BEIJING), "睡觉"),
            ("半夜12点 睡觉", datetime(2026, 10, 25, 0, 0, 0, tzinfo=BEIJING), "睡觉"),
            ("今晚20:30 提交周报", datetime(2026, 10, 24, 20, 30, 0, tzinfo=BEIJING), "提交周报"),
            ("今晚8点半 提交周报", datetime(2026, 10, 24, 20, 30, 0, tzinfo=BEIJING), "提交周报"),
            ("明天09:30 开会", datetime(2026, 10, 25, 9, 30, 0, tzinfo=BEIJING), "开会"),
            ("明天上午9点半 参加晨会", datetime(2026, 10, 25, 9, 30, 0, tzinfo=BEIJING), "参加晨会"),
            ("后天下午2点 讨论", datetime(2026, 10, 26, 14, 0, 0, tzinfo=BEIJING), "讨论"),
            ("大后天晚上8点 聚餐", datetime(2026, 10, 27, 20, 0, 0, tzinfo=BEIJING), "聚餐"),
            ("10月25日 09:30 提交", datetime(2026, 10, 25, 9, 30, 0, tzinfo=BEIJING), "提交"),
            ("2026-10-25 15:30 评审", datetime(2026, 10, 25, 15, 30, 0, tzinfo=BEIJING), "评审"),
        ]
        for text, expected_dt, expected_content in cases:
            with self.subTest(text=text):
                res = parse(text, BASE_NOW)
                self.assertEqual(res.content, expected_content)
                self.assertEqual(res.due_at, expected_dt.timestamp())

    def test_command_structures(self):
        # 结构 A: <时间>提醒我<事项>
        r1 = parse("@机器人 10分钟后提醒我关火", BASE_NOW)
        self.assertEqual(r1.content, "关火")
        self.assertEqual(r1.due_at, BASE_NOW.timestamp() + 600)

        # 结构 B: 提醒我<时间><事项>
        r2 = parse("@机器人 提醒我10分钟后关火", BASE_NOW)
        self.assertEqual(r2.content, "关火")
        self.assertEqual(r2.due_at, BASE_NOW.timestamp() + 600)

        # 结构 C: 斜杠命令 /提醒 <时间> <事项>
        r3 = parse("/提醒 10分钟后 关火", BASE_NOW)
        self.assertEqual(r3.content, "关火")
        self.assertEqual(r3.due_at, BASE_NOW.timestamp() + 600)

        r4 = parse("/提醒 15:30 开周会", BASE_NOW)
        self.assertEqual(r4.content, "开周会")
        self.assertEqual(r4.due_at, datetime(2026, 10, 24, 15, 30, 0, tzinfo=BEIJING).timestamp())

    def test_rejection_rules(self):
        # 1. 周期词
        with self.assertRaises(ReminderParseError) as cm:
            parse("@机器人 每天08:00提醒我打卡", BASE_NOW)
        self.assertIn("暂不支持重复提醒", str(cm.exception))

        # 2. 仅日期无具体时刻
        with self.assertRaises(ReminderParseError) as cm:
            parse("@机器人 明天提醒我交报告", BASE_NOW)
        self.assertIn("请补充具体提醒时刻", str(cm.exception))

        with self.assertRaises(ReminderParseError) as cm:
            parse("/提醒 明天 交报告", BASE_NOW)
        self.assertIn("请补充具体提醒时刻", str(cm.exception))

        # 3. 目标时间已过去（不顺延）
        with self.assertRaises(ReminderTimeError) as cm:
            parse("@机器人 今天14:00提醒我开会", BASE_NOW)
        self.assertEqual(cm.exception.kind, "past")
        msg = format_error(cm.exception)
        self.assertIn("目标时间（2026-10-24 14:00）已过去", msg)

        # 4. 时间太近（< 1分钟）
        with self.assertRaises(ReminderTimeError) as cm:
            parse("@机器人 30秒后提醒我关火", BASE_NOW)
        self.assertEqual(cm.exception.kind, "too_soon")
        self.assertIn("至少在 1 分钟之后", format_error(cm.exception))

        # 5. 时间太远（> 30天）
        with self.assertRaises(ReminderTimeError) as cm:
            parse("@机器人 31天后提醒我开会", BASE_NOW)
        self.assertEqual(cm.exception.kind, "too_far")
        self.assertIn("最长支持未来 30 天内", format_error(cm.exception))

        # 6. 替他人设置拦截
        with self.assertRaises(ReminderParseError) as cm:
            parse("@机器人 @大壮 5分钟后提醒他开会", BASE_NOW, has_other_mentions=True)
        self.assertIn("仅支持为自己设置提醒", str(cm.exception))

        with self.assertRaises(ReminderParseError) as cm:
            parse("@机器人 5分钟后提醒他开会", BASE_NOW, has_other_mentions=False)
        self.assertIn("仅支持为自己设置提醒", str(cm.exception))

        with self.assertRaises(ReminderParseError) as cm:
            parse("@机器人 提醒大家明天09:30开会", BASE_NOW)
        self.assertIn("仅支持为自己设置提醒", str(cm.exception))

        # 7. 事项缺失或超长
        with self.assertRaises(ReminderParseError) as cm:
            parse("@机器人 10分钟后提醒我", BASE_NOW)
        self.assertIn("未识别到有效的提醒时间或事项", str(cm.exception))

        with self.assertRaises(ReminderParseError) as cm:
            parse("@机器人 提醒我", BASE_NOW)
        self.assertIn("未识别到有效的提醒时间或事项", str(cm.exception))

        with self.assertRaises(ReminderError) as cm:
            parse(f"@机器人 10分钟后提醒我 {'a' * 101}", BASE_NOW)
        self.assertIn("1–100 个字符", format_error(cm.exception))

        with self.assertRaises(ReminderError) as cm:
            parse("@机器人 10分钟后提醒我 关火\n关电闸", BASE_NOW)
        self.assertIn("1–100 个字符", format_error(cm.exception))

        # 8. 非法中文数字异常拦截并转换为用法提示，不泄露内部描述
        with self.assertRaises(ReminderParseError) as cm:
            parse("@机器人 一二三分钟后提醒我喝水", BASE_NOW)
        self.assertIn("未识别到有效的提醒时间或事项", str(cm.exception))


class ReminderPluginTestCase(unittest.TestCase):
    def setUp(self):
        previous = logging.root.manager.disable
        logging.disable(logging.CRITICAL)
        self.addCleanup(logging.disable, previous)
        SCRATCH.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=SCRATCH)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.members_path = self.root / "members.json"
        self.members_path.write_text(json.dumps({"database_file": ".runtime/member_data.sqlite3",
                                                 "groups": {ROOM: [], OTHER_ROOM: []}}), encoding="utf-8")
        self.members = MemberStore(self.members_path, ROBOT)
        self.store = ReminderStore(self.root / ".runtime/reminders.sqlite3", ROBOT, self.root)
        self.addCleanup(self.store.close)
        self.ctx = MockContext(self.members, self.store)
        self.plugin = ReminderPlugin(clock=lambda: BASE_NOW.timestamp())
        self.plugin.on_load(self.ctx)

    def test_create_and_view_and_cancel_lifecycle(self):
        # 1. AC-01: 相对时间创建提醒
        msg = make_group_message("@机器人 10分钟后提醒我关火", sender=ALICE, room=ROOM,
                                 is_at_self=True, at_members=(ROBOT,), reminder_intent=True)
        self.plugin.on_message(msg, self.ctx)
        self.assertEqual(len(self.ctx.replies), 1)
        _, _, reply_text = self.ctx.replies[-1]
        self.assertIn("已设置提醒 [#1]", reply_text)
        self.assertIn("2026-10-24 15:10", reply_text)
        self.assertIn("关火", reply_text)

        # 2. 查看提醒列表
        list_msg = make_group_message("/我的提醒", sender=ALICE, room=ROOM, reminder_intent=True)
        self.plugin.on_message(list_msg, self.ctx)
        _, _, reply_text = self.ctx.replies[-1]
        self.assertIn("你在本群的待触发提醒（1/5）：", reply_text)
        self.assertIn("#1 2026-10-24 15:10 关火", reply_text)

        # 3. 他人隔离：Bob 尝试取消 Alice 的 #1
        bob_cancel = make_group_message("/取消提醒 1", sender=BOB, room=ROOM, reminder_intent=True)
        self.plugin.on_message(bob_cancel, self.ctx)
        _, _, reply_text = self.ctx.replies[-1]
        self.assertEqual(reply_text, "未找到你的待触发提醒 #1，请发送 /我的提醒 查看。")

        # 4. Alice 成功取消 #1
        alice_cancel = make_group_message("/取消提醒 1", sender=ALICE, room=ROOM, reminder_intent=True)
        self.plugin.on_message(alice_cancel, self.ctx)
        _, _, reply_text = self.ctx.replies[-1]
        self.assertEqual(reply_text, "已取消提醒 [#1]（2026-10-24 15:10 关火）。")

        # 5. 取消后再次查看为空
        self.plugin.on_message(list_msg, self.ctx)
        _, _, reply_text = self.ctx.replies[-1]
        self.assertEqual(reply_text, "你在本群暂无待触发的提醒。")

    def test_cancel_all(self):
        # 创建 2 条提醒
        msg1 = make_group_message("/提醒 10分钟后 喝水", sender=ALICE, room=ROOM, reminder_intent=True)
        msg2 = make_group_message("/提醒 20分钟后 休息", sender=ALICE, room=ROOM, reminder_intent=True)
        self.plugin.on_message(msg1, self.ctx)
        self.plugin.on_message(msg2, self.ctx)

        # 取消全部
        cancel_all_msg = make_group_message("/取消全部提醒", sender=ALICE, room=ROOM, reminder_intent=True)
        self.plugin.on_message(cancel_all_msg, self.ctx)
        _, _, reply_text = self.ctx.replies[-1]
        self.assertEqual(reply_text, "已取消你在本群的全部待触发提醒（共 2 条）。")

        # 再次取消全部显示无待触发提醒
        self.plugin.on_message(cancel_all_msg, self.ctx)
        _, _, reply_text = self.ctx.replies[-1]
        self.assertEqual(reply_text, "你在本群暂无待触发的提醒。")

    def test_cancel_invalid_arguments(self):
        msg_no_arg = make_group_message("/取消提醒", sender=ALICE, room=ROOM, reminder_intent=True)
        self.plugin.on_message(msg_no_arg, self.ctx)
        self.assertEqual(self.ctx.replies[-1][2], "请指定要取消的提醒编号，例如：/取消提醒 1")

        msg_bad_arg = make_group_message("/取消提醒 abc", sender=ALICE, room=ROOM, reminder_intent=True)
        self.plugin.on_message(msg_bad_arg, self.ctx)
        self.assertEqual(self.ctx.replies[-1][2], "请指定有效的提醒编号，例如：/取消提醒 1")

    def test_member_and_room_limits(self):
        # 1. 个人本群上限 5 条 (AC-07)
        for i in range(1, 6):
            msg = make_group_message(f"/提醒 {10 + i}分钟后 事项{i}", sender=ALICE, room=ROOM, reminder_intent=True)
            self.plugin.on_message(msg, self.ctx)
            self.assertIn(f"已设置提醒 [#{i}]", self.ctx.replies[-1][2])

        # 第 6 条超限
        msg6 = make_group_message("/提醒 20分钟后 事项6", sender=ALICE, room=ROOM, reminder_intent=True)
        self.plugin.on_message(msg6, self.ctx)
        self.assertEqual(self.ctx.replies[-1][2], "你在本群的待触发提醒已达上限（5/5），请先取消部分提醒后再试。")

        # 2. 全群上限 30 条
        # 给其他 5 位成员各创建 5 条 (总计 5 + 25 = 30)
        for u in range(1, 6):
            uid = f"wxid_user_{u}"
            for i in range(1, 6):
                self.store.create(ROOM, uid, BASE_NOW.timestamp() + 600 + i * 10, f"test{i}", now=BASE_NOW.timestamp())

        # Bob 尝试创建第 31 条
        msg_bob = make_group_message("/提醒 15分钟后 喝水", sender=BOB, room=ROOM, reminder_intent=True)
        self.plugin.on_message(msg_bob, self.ctx)
        self.assertEqual(self.ctx.replies[-1][2], "本群待触发提醒总数已达上限（30/30），请稍后再试。")

    def test_multi_room_isolation(self):
        # Alice 在群 ROOM 创建提醒
        msg = make_group_message("/提醒 10分钟后 关火", sender=ALICE, room=ROOM, reminder_intent=True)
        self.plugin.on_message(msg, self.ctx)

        # Alice 在群 OTHER_ROOM 查看
        msg_other = make_group_message("/我的提醒", sender=ALICE, room=OTHER_ROOM, reminder_intent=True)
        self.plugin.on_message(msg_other, self.ctx)
        self.assertEqual(self.ctx.replies[-1][2], "你在本群暂无待触发的提醒。")

        # Alice 在群 OTHER_ROOM 取消 #1
        msg_cancel = make_group_message("/取消提醒 1", sender=ALICE, room=OTHER_ROOM, reminder_intent=True)
        self.plugin.on_message(msg_cancel, self.ctx)
        self.assertEqual(self.ctx.replies[-1][2], "未找到你的待触发提醒 #1，请发送 /我的提醒 查看。")

    def test_sending_state_cancel_returns_not_found(self):
        # 模拟提醒已被调度器认领（sending 状态），cancel 返回 None
        reminder = self.store.create(ROOM, ALICE, BASE_NOW.timestamp() + 300, "做饭", now=BASE_NOW.timestamp())
        self.assertTrue(self.store.claim(reminder.id))

        cancel_msg = make_group_message("/取消提醒 1", sender=ALICE, room=ROOM, reminder_intent=True)
        self.plugin.on_message(cancel_msg, self.ctx)
        self.assertEqual(self.ctx.replies[-1][2], "未找到你的待触发提醒 #1，请发送 /我的提醒 查看。")

    def test_mention_chat_coexistence_no_double_reply(self):
        """验证与 mention_chat 共存时，遇到 reminder_intent 直接跳过，不产生双回复 (AC-08)。"""
        mention_chat = MentionChatPlugin()
        mention_chat._active_day = None
        mention_chat.memory_store = None
        mention_chat.members = self.members
        mention_chat.conversation_store = Mock()
        mention_chat._run_group_agent = Mock()

        # 1. 普通艾特消息（无 reminder_intent）：mention_chat 正常处理
        chat_msg = make_group_message("@机器人 你好呀", sender=ALICE, room=ROOM,
                                      is_at_self=True, at_members=(ROBOT,), reminder_intent=False)
        chat_msg = prepare_reminder(chat_msg, self.members)
        self.assertNotIn("reminder_intent", chat_msg.data)

        mention_chat.on_message(chat_msg, self.ctx)
        self.assertTrue(mention_chat._run_group_agent.called)

        # 重置 mock
        mention_chat._run_group_agent.reset_mock()

        # 2. 提醒艾特消息：经过 prepare_reminder 打上 reminder_intent
        remind_msg = make_group_message("@机器人 10分钟后提醒我喝水", sender=ALICE, room=ROOM,
                                        is_at_self=True, at_members=(ROBOT,))
        remind_msg = prepare_reminder(remind_msg, self.members)
        self.assertTrue(remind_msg.data.get("reminder_intent"))

        # mention_chat 遇到 reminder_intent 直接跳过
        mention_chat.on_message(remind_msg, self.ctx)
        self.assertFalse(mention_chat._run_group_agent.called)

        # reminder 插件正常处理并只回复 1 次
        self.plugin.on_message(remind_msg, self.ctx)
        self.assertEqual(len(self.ctx.replies), 1)
        self.assertIn("已设置提醒 [#1]", self.ctx.replies[-1][2])

    def test_invalid_number_plugin_reply(self):
        """验证一二三等非法中文数字在插件层兜底回复用法提示，且不泄露异常内部描述。"""
        msg = make_group_message("@机器人 一二三分钟后提醒我喝水", sender=ALICE, room=ROOM,
                                 is_at_self=True, at_members=(ROBOT,), reminder_intent=True)
        self.plugin.on_message(msg, self.ctx)
        self.assertIn("未识别到有效的提醒时间或事项", self.ctx.replies[-1][2])


if __name__ == "__main__":
    unittest.main()
