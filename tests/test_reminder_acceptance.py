"""NERE-99 提醒功能 AC-01～AC-09 独立验收（假后端 + 假时钟，不连接微信）。

规格来源：父卡 NERE-97 插件与需求专家规格「四、Given / When / Then 验收用例」。
被测基线：develop @ effb4a2（阶段 1 PR #9、阶段 2 PR #10）。

虚构测试基线（与规格一致）：
- 机器人账号 wxid_bot_demo；启用群 A 100001@chatroom、群 B 100002@chatroom；
- wxid_alice_demo 在群 A 绑定称呼「阿狸」，在群 B 未绑定、缓存微信名 Alice；
- wxid_bob_demo 未绑定、缓存微信名「大壮」；
- 基准宿主机北京时间 2026-10-24 15:00:00。

路径说明：插件回复与到点通知均经真实 PluginContext / ServiceFacade 走到假后端，
到点文本因此包含真实拼装的 ``@称呼\\u2005`` 前缀；解析与调度全部使用可注入假时钟。
本文件按框架既有假设验证：真实微信中 @ 标签以 ``\\u2005``（四分之一空格）结尾；
该假设的最终确认需要 Windows 真机原生验证，不在本文件范围。
"""
from __future__ import annotations

import json
import logging
import sqlite3
import tempfile
import unittest
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock

from plugins.mention_chat import Plugin as MentionChatPlugin
from plugins.reminder import Plugin as ReminderPlugin
from wxbot_framework.members import BEIJING, MemberStore
from wxbot_framework.reminders import ReminderScheduler, ReminderStore, prepare_reminder
from wxbot_framework.sdk import Message, PluginContext, ServiceFacade

ROOT = Path(__file__).resolve().parent.parent
SCRATCH = ROOT / ".runtime/tests"
ROOM, OTHER_ROOM = "100001@chatroom", "100002@chatroom"
ROBOT, OTHER_ROBOT = "wxid_bot_demo", "wxid_bot_other"
ALICE, BOB = "wxid_alice_demo", "wxid_bob_demo"
# 北京时间 2026-10-24 15:00:00（数据库统一存 UTC 时间戳）。
NOW = datetime(2026, 10, 24, 7, 0, 0, tzinfo=timezone.utc).timestamp()
# 真实微信 @ 标签结尾符（框架统一假设，四分之一空格），用 chr(0x2005) 显式表达。
AT = chr(0x2005)
# 用户侧艾特机器人样例："@机器人" + AT + 空格 + 正文。
AT_ROBOT = "@机器人" + AT + " "


def beijing(*args):
    return datetime(*args, tzinfo=BEIJING).timestamp()


class _RecordCollector(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


class FakeRaw:
    def __init__(self, self_wxid):
        self.self_wxid = self_wxid


class FakeBackend:
    """假原生后端：只记录发送内容。"""

    def __init__(self):
        self.session_id = uuid.uuid4().hex
        self.raw = FakeRaw(ROBOT)
        self.sent = []  # (chat_id, text, at_tuple)；at_tuple 为空表示普通文本。

    def send_text(self, chat_id, text):
        self.sent.append((chat_id, text, ()))
        return True

    def send_at_text(self, room, text, at_list):
        self.sent.append((room, text, tuple(at_list)))
        return True

    def stop(self):
        return {"dll_unloaded": True}


class AcceptanceTestCase(unittest.TestCase):
    """共享验收台架：真实插件 + 真实存储/调度器 + 真实 ServiceFacade + 假后端与假时钟。"""

    def setUp(self):
        previous = logging.root.manager.disable
        logging.disable(logging.CRITICAL)
        self.addCleanup(logging.disable, previous)
        SCRATCH.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=SCRATCH)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.members_path = self.root / "members.json"
        self.members_path.write_text(json.dumps({
            "database_file": ".runtime/member_data.sqlite3",
            "history_enabled": True,
            "daily_clear": True,
            "groups": {ROOM: [], OTHER_ROOM: []},
        }), encoding="utf-8")
        self.members = MemberStore(self.members_path, ROBOT)
        self.addCleanup(self.members.close)
        self.members.initialize()
        # 规格基线：Alice 在群 A 绑定「阿狸」；Bob 与 Alice(群 B) 走缓存微信名。
        self.members.register("阿狸", ROOM, ALICE)
        self.members.remember_wechat_name(BOB, "大壮")
        self.members.remember_wechat_name(ALICE, "Alice")

        self.store = ReminderStore.default(ROBOT, self.root)
        self.addCleanup(self.store.close)
        self.backend = FakeBackend()
        self.service = ServiceFacade(text_delay={"enabled": False})
        self.service.bind(self.backend)
        self.addCleanup(self.service.close)
        self.ctx = PluginContext("reminder", {}, self.service,
                                 ("text.receive", "group.text.receive", "group.text.send"),
                                 member_store=self.members, reminder_store=self.store)

        self.now = NOW
        self.connected = True
        self.plugin = ReminderPlugin(clock=lambda: self.now)
        self.plugin.on_load(self.ctx)
        self.scheduler = ReminderScheduler(self.store, self.members, self._deliver,
                                           is_active=lambda: True,
                                           ready=lambda: self.connected,
                                           clock=lambda: self.now)

    def _deliver(self, room, content, at_list, display_names):
        # 与 runtime.BotApplication._send_reminder 相同：不绑定旧连接会话。
        if not self.connected:
            raise RuntimeError("假后端未连接")
        return self.service.send_at_text(room, content, at_list, display_names=display_names)

    def build(self, text, *, sender=ALICE, room=ROOM, at_self=True, at_members=None):
        """构造群消息并走真实意图标记（prepare_reminder），返回待分发消息。"""
        if at_members is None:
            at_members = (ROBOT,) if at_self else ()
        message = Message(text, room, sender_id=sender, is_group=True,
                          session_id=self.backend.session_id,
                          data={"dispatch_to_plugins": True, "is_at_self": at_self,
                                "at_members": at_members})
        return prepare_reminder(message, self.members)

    def say(self, text, *, sender=ALICE, room=ROOM, at_self=True, at_members=None):
        """模拟一条群消息经意图标记后分发到 reminder 插件，返回新增的普通回复文本列表。"""
        start = len(self.backend.sent)
        message = self.build(text, sender=sender, room=room, at_self=at_self, at_members=at_members)
        self.plugin.on_message(message, self.ctx)
        return [entry[1] for entry in self.backend.sent[start:] if entry[2] == ()]

    @property
    def at_sends(self):
        return [entry for entry in self.backend.sent if entry[2]]


class AC01RelativeTimeOnTimeMentionTests(AcceptanceTestCase):
    def test_ac01_relative_create_and_on_time_at_with_bound_name(self):
        # When：15:00:00 Alice 在群 A 真实艾特机器人设置相对时间提醒。
        replies = self.say(AT_ROBOT + "10分钟后提醒我关火")
        # Then：确认回复含 #1、完整时间与事项。
        self.assertEqual(len(replies), 1)
        self.assertIn("已设置提醒 [#1]", replies[0])
        self.assertIn("2026-10-24 15:10", replies[0])
        self.assertIn("关火", replies[0])

        # Then：15:10 向原群发起真实群艾特（at_list 仅含设置人），前缀 @阿狸，内容含关火。
        self.now = NOW + 600 + 1
        self.assertEqual(self.scheduler.tick(), 1)
        self.assertEqual(len(self.at_sends), 1)
        room, text, ats = self.at_sends[0]
        self.assertEqual((room, ats), (ROOM, (ALICE,)))
        self.assertEqual(text, "@阿狸" + AT + "\n⏰ 提醒时间到：关火（设定时间 15:10）")
        # 发送成功后立即清理，不残留正文。
        self.assertEqual(self.store.list_pending(ROOM, ALICE), [])


class AC02NextDayWechatNameFallbackTests(AcceptanceTestCase):
    def test_ac02_next_day_reminder_survives_clear_and_uses_wechat_name(self):
        # Given：Bob 未绑定，缓存微信名「大壮」。
        # When：Bob 设置跨天口语时间提醒。
        replies = self.say(AT_ROBOT + "提醒我明天上午9点交日报", sender=BOB)
        self.assertEqual(len(replies), 1)
        self.assertIn("已设置提醒 [#1]", replies[0])
        self.assertIn("2026-10-25 09:00", replies[0])
        self.assertIn("交日报", replies[0])

        # When：群内执行 /清空（聊天记录清理路径），并执行跨天每日清理。
        self.members.clear(ROOM)
        self.members.rollover()
        # Then：待触发提醒不被 /清空 或每日清理删除。
        self.assertEqual(len(self.store.list_pending(ROOM, BOB)), 1)

        # Then：次日 09:00 在原群 @ Bob，未绑定回退微信名「大壮」。
        self.now = beijing(2026, 10, 25, 9, 0, 30)
        self.assertEqual(self.scheduler.tick(), 1)
        room, text, ats = self.at_sends[0]
        self.assertEqual((room, ats), (ROOM, (BOB,)))
        self.assertEqual(text, "@大壮" + AT + "\n⏰ 提醒时间到：交日报（设定时间 09:00）")


class AC03ViewCancelAndPermissionTests(AcceptanceTestCase):
    def test_ac03_cancel_loop_and_other_member_isolation(self):
        # Given：Alice 在群 A 有待触发提醒 #1。
        self.assertEqual(len(self.say("/提醒 10分钟后 开会")), 1)

        # When：Bob 尝试取消 Alice 的 #1。
        bob_reply = self.say("/取消提醒 1", sender=BOB)[0]
        # Then：Bob 得到未找到提示，#1 仍保留。
        self.assertEqual(bob_reply, "未找到你的待触发提醒 #1，请发送 /我的提醒 查看。")
        self.assertEqual(len(self.store.list_pending(ROOM, ALICE)), 1)

        # When：Alice 查看，再取消。
        alice_list = self.say("/我的提醒")[0]
        self.assertIn("你在本群的待触发提醒（1/5）", alice_list)
        self.assertIn("#1 2026-10-24 15:10 开会", alice_list)
        alice_cancel = self.say("/取消提醒 1")[0]
        self.assertEqual(alice_cancel, "已取消提醒 [#1]（2026-10-24 15:10 开会）。")

        # Then：取消后到点不再发送。
        self.now = NOW + 600 + 5
        self.assertEqual(self.scheduler.tick(), 0)
        self.assertEqual(self.at_sends, [])


class AC04PastTimeAndParseFailureTests(AcceptanceTestCase):
    def test_ac04_past_time_rejected_without_deferral_and_date_only_rejected(self):
        # When：15:00 发送当天 14:00 的提醒。
        first = self.say(AT_ROBOT + "今天14:00提醒我开会")[0]
        # Then：明确提示时间已过去，不顺延到次日。
        self.assertIn("设置失败：目标时间（2026-10-24 14:00）已过去", first)
        self.assertIn("请重新指定未来的时间", first)

        # When：仅日期无具体时刻。
        second = self.say(AT_ROBOT + "明天提醒我交周报")[0]
        # Then：提示补充具体时刻。
        self.assertIn("请补充具体提醒时刻", second)

        # Then：数据库均不新增记录。
        self.assertEqual(self.store.list_pending(ROOM, ALICE), [])


class AC05NoProxyNoRepeatTests(AcceptanceTestCase):
    def test_ac05_setting_for_others_and_repeat_rejected(self):
        # When：同时 @ 大壮并要求提醒他。
        first = self.say("@机器人" + AT + "@大壮" + AT + " 5分钟后提醒他开会",
                         at_members=(ROBOT, BOB))[0]
        self.assertIn("仅支持为自己设置提醒", first)

        # When：要求每天重复提醒。
        second = self.say(AT_ROBOT + "每天08:00提醒我打卡")[0]
        self.assertIn("暂不支持重复提醒", second)

        # Then：均不落库。
        self.assertEqual(self.store.list_pending(ROOM, ALICE), [])


class AC06MultiRoomMultiAccountTests(AcceptanceTestCase):
    def test_ac06_room_isolation_and_account_isolation(self):
        # Given：Alice 在群 A 有 #1。
        self.assertEqual(len(self.say("/提醒 10分钟后 关火")), 1)

        # When：Alice 在群 B 查看与取消。
        list_reply = self.say("/我的提醒", room=OTHER_ROOM)[0]
        cancel_reply = self.say("/取消提醒 1", room=OTHER_ROOM)[0]
        # Then：群 B 不可见、不可操作群 A 的提醒。
        self.assertEqual(list_reply, "你在本群暂无待触发的提醒。")
        self.assertEqual(cancel_reply, "未找到你的待触发提醒 #1，请发送 /我的提醒 查看。")
        self.assertEqual(len(self.store.list_pending(ROOM, ALICE)), 1)

        # Then：#1 到点只发往原群 A。
        self.now = NOW + 605
        self.assertEqual(self.scheduler.tick(), 1)
        self.assertEqual([entry[0] for entry in self.at_sends], [ROOM])

        # When：配置切换为另一机器人账号 wxid_bot_other。
        other_members = MemberStore(self.members_path, OTHER_ROBOT)
        self.addCleanup(other_members.close)
        other_members.initialize()
        other_store = ReminderStore(self.store.path, OTHER_ROBOT, self.root)
        self.addCleanup(other_store.close)
        # Then：另一账号读不到也触发不了 wxid_bot_demo 的提醒。
        self.assertEqual(other_store.list_pending(ROOM, ALICE), [])
        self.assertEqual(other_store.due(NOW + 3600), [])
        other_scheduler = ReminderScheduler(other_store, other_members, self._deliver,
                                            clock=lambda: self.now)
        self.assertEqual(other_scheduler.tick(now=NOW + 605), 0)
        self.assertEqual(len(self.at_sends), 1)


class AC07QuotaLimitTests(AcceptanceTestCase):
    def test_ac07_member_limit_and_release(self):
        # Given：Alice 在群 A 已有 5 条待触发提醒（编号 1–5）。
        for index in range(1, 6):
            reply = self.say(f"/提醒 {10 + index}分钟后 事项{index}")[0]
            self.assertIn(f"已设置提醒 [#{index}]", reply)

        # When：尝试创建第 6 条。
        # Then：拒绝并提示 5/5。
        sixth = self.say("/提醒 20分钟后 事项6")[0]
        self.assertEqual(sixth, "你在本群的待触发提醒已达上限（5/5），请先取消部分提醒后再试。")

        # When：取消 #3 后再建。
        self.say("/取消提醒 3")
        recreated = self.say("/提醒 25分钟后 事项7")[0]
        # Then：成功并复用最小空闲号 3。
        self.assertIn("已设置提醒 [#3]", recreated)

        # When：#1 到点发出（触发释放名额）。
        self.now = NOW + 11 * 60 + 1
        self.assertEqual(self.scheduler.tick(), 1)
        # Then：可再次创建并复用编号 1。
        after_fire = self.say("/提醒 30分钟后 事项8")[0]
        self.assertIn("已设置提醒 [#1]", after_fire)

    def test_ac07_room_limit_30(self):
        # Given：群内已有 30 条待触发提醒（Alice 5 条 + 其他成员 25 条）。
        for index in range(1, 6):
            self.say(f"/提醒 {10 + index}分钟后 事项{index}")
        for user in range(1, 6):
            for minute in range(1, 6):
                self.store.create(ROOM, f"wxid_member_{user}", NOW + 1200 + minute * 60,
                                  f"占位{minute}", now=NOW)
        # When：Bob 尝试创建第 31 条。
        reply = self.say("/提醒 15分钟后 喝水", sender=BOB)[0]
        # Then：拒绝并提示 30/30。
        self.assertEqual(reply, "本群待触发提醒总数已达上限（30/30），请稍后再试。")


class AC08NoModelAndCoexistenceTests(AcceptanceTestCase):
    def test_ac08_scenario1_no_model_configured(self):
        # Given：场景 1 —— 未配置大模型：PluginContext 未接入 llm_config_path，
        # 提醒解析与调度全程 100% 本地规则，不触碰 ctx.llm。
        self.assertIsNone(self.ctx._llm_config_path)
        # When：正常设置提醒。
        replies = self.say(AT_ROBOT + "5分钟后提醒我喝水")
        self.assertIn("已设置提醒 [#1]", replies[0])
        # Then：到点照常 @ 提醒。
        self.now = NOW + 305
        self.assertEqual(self.scheduler.tick(), 1)
        self.assertIn("提醒时间到：喝水", self.at_sends[0][1])

    def test_ac08_scenario2_mention_chat_enabled_single_reply(self):
        # Given：场景 2 —— 模型已配置且 mention_chat 开启
        # （_run_group_agent 打桩代表模型对话路径，不真调模型）。
        mention = MentionChatPlugin()
        mention._active_day = None
        mention.memory_store = None
        mention.members = self.members
        mention.conversation_store = Mock()
        mention._run_group_agent = Mock()

        # 对照：普通艾特闲聊（不含提醒意图）由 mention_chat 处理。
        chat = self.build(AT_ROBOT + "今天天气怎么样")
        self.assertNotIn("reminder_intent", chat.data)
        mention.on_message(chat, self.ctx)
        self.assertTrue(mention._run_group_agent.called)

        # When：同一条提醒消息按广播顺序先给 mention_chat 再给 reminder。
        mention._run_group_agent.reset_mock()
        start = len(self.backend.sent)
        reminder_msg = self.build(AT_ROBOT + "5分钟后提醒我喝水")
        self.assertTrue(reminder_msg.data.get("reminder_intent"))
        mention.on_message(reminder_msg, self.ctx)
        self.plugin.on_message(reminder_msg, self.ctx)
        # Then：mention_chat 让路，提醒设置结果只回复一次，不双回。
        self.assertFalse(mention._run_group_agent.called)
        replies = [entry[1] for entry in self.backend.sent[start:] if entry[2] == ()]
        self.assertEqual(len(replies), 1)
        self.assertIn("已设置提醒 [#1]", replies[0])


class AC09PersistenceReloadRestartTests(AcceptanceTestCase):
    def member_tables(self):
        # with sqlite3.connect 只管理事务不关闭连接，Windows 下须显式 close 再清理目录。
        db = sqlite3.connect(self.root / ".runtime/member_data.sqlite3")
        try:
            return {row[0] for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
        finally:
            db.close()

    def test_ac09_hot_reload_does_not_double_send(self):
        # Given：#1 原定 15:05。
        self.assertEqual(len(self.say("/提醒 5分钟后 开会")), 1)
        tables_before = self.member_tables()

        # When：控制台 reload —— 插件重建实例，框架调度器不重建、不新增。
        reloaded = ReminderPlugin(clock=lambda: self.now)
        reloaded.on_load(self.ctx)
        # Then：到点只发送一次，重复 tick 不再发送。
        self.now = NOW + 305
        self.assertEqual(self.scheduler.tick(), 1)
        self.assertEqual(self.scheduler.tick(), 0)
        # 即便出现第二个调度器实例同库并发 tick，原子认领也保证不重复 @。
        second = ReminderScheduler(self.store, self.members, self._deliver,
                                   clock=lambda: self.now)
        self.assertEqual(second.tick(), 0)
        self.assertEqual(len(self.at_sends), 1)
        self.assertEqual(self.at_sends[0][1], "@阿狸" + AT + "\n⏰ 提醒时间到：开会（设定时间 15:05）")
        # member_data.sqlite3 未新增表。
        self.assertEqual(self.member_tables(), tables_before)

    def test_ac09_restart_resends_within_grace_and_expires_beyond(self):
        # Given：库中有原定 14:00 的旧提醒（13:59 创建）与原定 15:05 的新提醒；
        # member_data.sqlite3 无新增表。
        self.store.create(ROOM, BOB, NOW - 3600, "过期事项", now=NOW - 3660)
        self.store.create(ROOM, ALICE, NOW + 300, "开会", now=NOW)
        tables_before = self.member_tables()

        # When：停机后至 15:12 重启恢复连接（新存储连接、新调度器）。
        self.store.close()
        store = ReminderStore(self.store.path, ROBOT, self.root)
        self.addCleanup(store.close)
        scheduler = ReminderScheduler(store, self.members, self._deliver,
                                      clock=lambda: self.now)
        self.now = NOW + 720  # 15:12
        self.assertEqual(scheduler.recover(), [])
        self.assertEqual(scheduler.tick(), 1)

        # Then：原定 15:05 的（延迟 7 分钟 ≤ 10 分钟宽限）补发一次 [延迟提醒]。
        room, text, ats = self.at_sends[0]
        self.assertEqual((room, ats), (ROOM, (ALICE,)))
        self.assertEqual(text, "@阿狸" + AT + "\n⏰ [延迟提醒] 开会（原定 15:05）")
        # Then：原定 14:00 的（延迟 72 分钟 > 10 分钟）静默过期清理，不补发。
        self.assertEqual(store.list_pending(ROOM, BOB), [])
        self.assertEqual(store.list_pending(ROOM, ALICE), [])
        self.assertEqual(len(self.at_sends), 1)
        # member_data.sqlite3 未新增表或改动。
        self.assertEqual(self.member_tables(), tables_before)


class ReviewNotesTests(AcceptanceTestCase):
    """父卡审查备注的三项复核：歧义最长匹配、兜底日志不泄密、@ 标签假设。"""

    def test_ambiguous_longest_match_is_visible_in_confirmation(self):
        # 备注 1：歧义写法按最长匹配取分钟。当前行为基线（成功回复中时间与事项
        # 均完整可见，用户可立即发现并取消）；验收判定见 NERE-99 交接评论。
        first = self.say(AT_ROBOT + "提醒我下午5点三十一起吃饭")[0]
        self.assertIn("2026-10-24 17:31", first)
        self.assertIn("「起吃饭」", first)

        second = self.say(AT_ROBOT + "提醒我明天9点5个人开会")[0]
        self.assertIn("2026-10-25 09:05", second)
        self.assertIn("「个人开会」", second)

    def test_plugin_fallback_logs_never_contain_content_or_ids(self):
        # 备注 2：插件兜底分支日志写入 str(err)。捕获 Plugin.reminder 全部日志，
        # 驱动正常创建、各类拒绝回复，以及可达的「存储已关闭」未预期异常路径。
        logging.disable(logging.NOTSET)
        collector = _RecordCollector()
        logger = logging.getLogger("Plugin.reminder")
        previous_level = logger.level
        logger.addHandler(collector)
        logger.setLevel(logging.DEBUG)
        try:
            self.say(AT_ROBOT + "10分钟后提醒我关火")
            self.say(AT_ROBOT + "今天14:00提醒我开会")
            self.say("/取消提醒 9", sender=BOB)
            # 触发创建兜底分支：存储关闭后 create 抛 RuntimeError（静态文案，无正文）。
            self.store.close()
            reply = self.say("/提醒 10分钟后 喝水")[0]
            self.assertEqual(reply, "设置提醒失败，请稍后重试。")
        finally:
            logger.removeHandler(collector)
            logger.setLevel(previous_level)
        text = "\n".join(collector.messages)
        for secret in ("关火", "开会", "喝水", "过期事项", ROOM, OTHER_ROOM, ALICE, BOB,
                       "100001", "100002", "wxid_"):
            self.assertNotIn(secret, text)
        # 兜底分支确实触发且只记录异常类型与静态描述，不含正文与 ID。
        self.assertIn("创建提醒发生未预期异常: RuntimeError: 提醒存储已关闭", text)

    def test_at_label_uses_u2005_on_both_parse_and_send(self):
        # 备注 3：框架统一假设真实微信 @ 标签以 U+2005（四分之一空格）结尾。
        # 收：意图标记能剥离以 U+2005 结尾的 @ 标签；
        # 发：到点文本按 "@称呼" + U+2005 拼装真实艾特标签。
        message = self.build(AT_ROBOT + "10分钟后提醒我关火")
        self.assertTrue(message.data.get("reminder_intent"))
        self.say(AT_ROBOT + "10分钟后提醒我关火")
        self.now = NOW + 605
        self.assertEqual(self.scheduler.tick(), 1)
        room, text, ats = self.at_sends[0]
        self.assertEqual((room, ats), (ROOM, (ALICE,)))
        self.assertTrue(text.startswith("@阿狸" + AT + "\n"))


if __name__ == "__main__":
    unittest.main()
