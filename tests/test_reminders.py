"""提醒存储、调度器、意图标记与运行时接入；不连接微信，使用假时钟与假发送。"""
from __future__ import annotations

import json
import logging
import tempfile
import threading
import time
import unittest
import uuid
from datetime import datetime, timezone
from pathlib import Path

from wxbot_framework.config import ConfigStore
from wxbot_framework.members import BEIJING, MemberStore
from wxbot_framework.reminders import (GRACE_SECONDS, ReminderLimit, ReminderScheduler, ReminderStore,
                                       ReminderTimeError, prepare_reminder, reminder_text)
from wxbot_framework.runtime import BotApplication
from wxbot_framework.sdk import CapabilityError, Message, PluginContext, SendNotAttempted, ServiceFacade
from wxbot_framework.summary import CONTROL

ROOT = Path(__file__).resolve().parent.parent
SCRATCH = ROOT / ".runtime/tests"
ROOM, OTHER_ROOM, OFF_ROOM = "100001@chatroom", "100002@chatroom", "100003@chatroom"
ROBOT, OTHER_ROBOT = "wxid_bot_demo", "wxid_bot_other"
ALICE, BOB = "wxid_alice_demo", "wxid_bob_demo"
# 北京时间 2026-10-24 15:00:00
NOW = datetime(2026, 10, 24, 7, 0, 0, tzinfo=timezone.utc).timestamp()


def wait_for(condition, seconds=3):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.01)
    raise AssertionError("等待异步结果超时")


class Sender:
    def __init__(self):
        self.sent = []
        self.failures = []
        self.lock = threading.Lock()

    def __call__(self, room, content, at_list, display_names):
        if self.failures:
            raise self.failures.pop(0)
        with self.lock:
            self.sent.append((room, content, tuple(at_list), tuple(display_names)))
        return True


class ReminderTestCase(unittest.TestCase):
    def setUp(self):
        previous = logging.root.manager.disable
        logging.disable(logging.CRITICAL)
        self.addCleanup(logging.disable, previous)
        SCRATCH.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=SCRATCH)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.assertTrue(self.root.is_relative_to(SCRATCH.resolve()))
        self.members_path = self.root / "members.json"
        self.members_path.write_text(json.dumps({"database_file": ".runtime/member_data.sqlite3",
                                                 "groups": {ROOM: [], OTHER_ROOM: []}}), encoding="utf-8")
        self.members = self.new_members()
        self.store = self.new_store()
        self.now = NOW
        self.active = True
        self.connected = True
        self.sender = Sender()

    def new_members(self, account=ROBOT):
        members = MemberStore(self.members_path, account)
        self.addCleanup(members.close)
        members.initialize()
        return members

    def new_store(self, account=ROBOT):
        store = ReminderStore.default(account, self.root)
        self.addCleanup(store.close)
        return store

    def scheduler(self, store=None, members=None, sender=None, **options):
        options.setdefault("is_active", lambda: self.active)
        options.setdefault("ready", lambda: self.connected)
        options.setdefault("clock", lambda: self.now)
        return ReminderScheduler(store or self.store, members or self.members, sender or self.sender, **options)

    def create(self, minutes=10, *, room=ROOM, creator=ALICE, content="关火", store=None):
        return (store or self.store).create(room, creator, NOW + minutes * 60, content, now=NOW)


class ReminderStoreTests(ReminderTestCase):
    def test_database_is_under_runtime_and_created_lazily(self):
        self.assertEqual(self.store.path, (self.root / ".runtime/reminders.sqlite3").resolve())
        self.assertFalse(self.store.exists)
        with self.assertRaises(ValueError):
            ReminderStore(self.root / "reminders.sqlite3", ROBOT, self.root)
        self.create()
        self.assertTrue(self.store.path.exists())

    def test_member_limit_and_smallest_free_number(self):
        numbers = [self.create(minutes=10 + i).number for i in range(5)]
        self.assertEqual(numbers, [1, 2, 3, 4, 5])
        with self.assertRaises(ReminderLimit) as caught:
            self.create(minutes=30)
        self.assertEqual((caught.exception.kind, caught.exception.limit), ("member", 5))
        self.assertEqual(self.store.cancel(ROOM, ALICE, 2).number, 2)
        self.assertEqual(self.create(minutes=40).number, 2)
        # 另一个群单独计数和编号。
        self.assertEqual(self.create(room=OTHER_ROOM).number, 1)

    def test_room_limit(self):
        for index in range(6):
            for minute in range(5):
                self.create(minutes=10 + minute, creator=f"wxid_member_{index}")
        with self.assertRaises(ReminderLimit) as caught:
            self.create(creator="wxid_member_late")
        self.assertEqual((caught.exception.kind, caught.exception.limit), ("room", 30))
        self.assertEqual(self.create(room=OTHER_ROOM, creator="wxid_member_late").number, 1)

    def test_time_and_content_validation(self):
        for due, kind in ((NOW - 1, "past"), (NOW, "past"), (NOW + 59, "too_soon"),
                          (NOW + 30 * 86400 + 1, "too_far")):
            with self.assertRaises(ReminderTimeError) as caught:
                self.store.create(ROOM, ALICE, due, "喝水", now=NOW)
            self.assertEqual(caught.exception.kind, kind)
        for content in ("", "   ", "a" * 101, "两\n行", 123):
            with self.assertRaises(ValueError):
                self.store.create(ROOM, ALICE, NOW + 600, content, now=NOW)
        with self.assertRaises(ValueError):
            self.store.create("filehelper", ALICE, NOW + 600, "喝水", now=NOW)
        self.assertEqual(self.store.create(ROOM, ALICE, NOW + 60, "  喝水  ", now=NOW).content, "喝水")
        self.assertEqual(self.store.create(ROOM, ALICE, NOW + 30 * 86400, "a" * 100, now=NOW).number, 2)

    def test_room_member_and_account_isolation(self):
        first = self.create()
        self.assertIsNone(self.store.cancel(ROOM, BOB, first.number))
        self.assertIsNone(self.store.cancel(OTHER_ROOM, ALICE, first.number))
        self.assertEqual(self.store.cancel_all(OTHER_ROOM, ALICE), 0)
        self.assertEqual(self.store.list_pending(OTHER_ROOM, ALICE), [])
        self.assertEqual(self.store.list_pending(ROOM, BOB), [])
        other = self.new_store(OTHER_ROBOT)
        self.assertEqual(other.list_pending(ROOM, ALICE), [])
        self.assertEqual(other.due(NOW + 3600), [])
        self.assertFalse(other.claim(first.id))
        self.assertFalse(other.finish(first.id))
        self.assertIsNone(other.cancel(ROOM, ALICE, first.number))
        self.assertEqual([r.id for r in self.store.list_pending(ROOM, ALICE)], [first.id])
        self.create(minutes=20)
        self.assertEqual(self.store.cancel_all(ROOM, ALICE), 2)
        self.assertEqual(self.store.list_pending(ROOM, ALICE), [])

    def test_claim_is_atomic_and_sending_cannot_be_cancelled(self):
        row = self.create()
        self.assertTrue(self.store.claim(row.id))
        self.assertFalse(self.new_store().claim(row.id))
        self.assertIsNone(self.store.cancel(ROOM, ALICE, row.number))
        self.assertTrue(self.store.release(row.id))
        self.assertTrue(self.store.claim(row.id))
        self.assertTrue(self.store.finish(row.id))
        self.assertEqual(self.store.list_pending(ROOM, ALICE), [])


class ReminderSchedulerTests(ReminderTestCase):
    def test_on_time_sends_fixed_template_to_origin_room_and_creator_then_deletes(self):
        self.members.register("阿狸", ROOM, ALICE)
        row = self.create()
        scheduler = self.scheduler()
        self.now = row.due_at - 1
        self.assertEqual(scheduler.tick(), 0)
        self.now = row.due_at + 1
        self.assertEqual(scheduler.tick(), 1)
        self.assertEqual(self.sender.sent, [(ROOM, "⏰ 提醒时间到：关火（设定时间 15:10）", (ALICE,), ("阿狸",))])
        self.assertEqual(self.store.list_pending(ROOM, ALICE), [])
        self.assertEqual(scheduler.tick(), 0)

    def test_display_name_falls_back_to_wechat_name_then_default(self):
        self.members.remember_wechat_name(BOB, "大壮")
        self.create(creator=BOB)
        self.create(creator="wxid_nobody_demo")
        self.now = NOW + 600
        self.assertEqual(self.scheduler().tick(), 2)
        names = {sent[2][0]: sent[3][0] for sent in self.sender.sent}
        self.assertEqual(names, {BOB: "大壮", "wxid_nobody_demo": "群友"})

    def test_grace_resend_marks_delay_and_stale_ones_expire_silently(self):
        late = self.create(minutes=5, content="开会")
        stale = self.create(minutes=2, creator=BOB, content="旧事")
        # 15:12:01 恢复：15:05 的延迟约 7 分钟（宽限内），15:02 的已超过 10 分钟。
        self.now = stale.due_at + GRACE_SECONDS + 1
        self.assertEqual(self.scheduler().tick(), 1)
        self.assertEqual(self.sender.sent[0][1], "⏰ [延迟提醒] 开会（原定 15:05）")
        self.assertEqual(self.store.list_pending(ROOM, BOB), [])

    def test_other_day_shows_month_and_day(self):
        row = self.create(minutes=18 * 60, content="交日报")  # 北京次日 09:00
        self.assertEqual(reminder_text(row, NOW), "⏰ 提醒时间到：交日报（设定时间 10月25日 09:00）")
        self.assertEqual(reminder_text(row, row.due_at), "⏰ 提醒时间到：交日报（设定时间 09:00）")

    def test_send_not_attempted_releases_within_grace(self):
        row = self.create()
        self.sender.failures.append(SendNotAttempted("微信尚未连接"))
        scheduler = self.scheduler()
        self.now = row.due_at
        self.assertEqual(scheduler.tick(), 0)
        self.assertEqual(self.store.list_pending(ROOM, ALICE)[0].state, "pending")
        self.now += 120
        self.assertEqual(scheduler.tick(), 1)
        self.assertIn("[延迟提醒]", self.sender.sent[0][1])

    def test_send_not_attempted_after_grace_expires(self):
        row = self.create()
        self.sender.failures.append(SendNotAttempted("取消"))
        self.now = row.due_at + 100
        clock = iter([row.due_at + GRACE_SECONDS + 1])
        scheduler = self.scheduler(clock=lambda: next(clock))
        self.assertEqual(scheduler.tick(now=self.now), 0)
        self.assertEqual(self.store.list_pending(ROOM, ALICE), [])

    def test_native_failure_is_not_retried(self):
        row = self.create()
        self.sender.failures.append(OSError("原生发送失败"))
        scheduler = self.scheduler()
        self.now = row.due_at
        self.assertEqual(scheduler.tick(), 0)
        self.assertEqual(self.store.list_pending(ROOM, ALICE), [])
        self.assertEqual(scheduler.tick(), 0)
        self.assertEqual(self.sender.sent, [])

    def test_two_schedulers_ticking_concurrently_send_once(self):
        rows = [self.create(creator=f"wxid_member_{i}") for i in range(5)]
        second_store = self.new_store()
        barrier = threading.Barrier(2)
        schedulers = [self.scheduler(), self.scheduler(store=second_store)]
        results = []

        def run(scheduler):
            barrier.wait()
            results.append(scheduler.tick(now=rows[0].due_at))

        threads = [threading.Thread(target=run, args=(s,)) for s in schedulers]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(5)
        self.assertEqual(sum(results), 5)
        self.assertEqual(sorted(sent[2][0] for sent in self.sender.sent), sorted(r.creator for r in rows))

    def test_restart_abandons_residual_sending(self):
        row = self.create()
        kept = self.create(creator=BOB)
        self.assertTrue(self.store.claim(row.id))
        self.store.close()
        store = self.new_store()
        scheduler = self.scheduler(store=store)
        self.assertEqual([r.id for r in scheduler.recover()], [row.id])
        self.now = row.due_at
        self.assertEqual(scheduler.tick(), 1)
        self.assertEqual([s[2][0] for s in self.sender.sent], [kept.creator])

    def test_recover_does_not_create_database(self):
        self.assertEqual(self.scheduler().recover(), [])
        self.assertFalse(self.store.path.exists())

    def test_room_removed_from_groups_or_allowlist_is_discarded(self):
        self.create()
        self.create(room=OTHER_ROOM)
        self.now = NOW + 600
        self.members.groups.discard(OTHER_ROOM)
        self.assertEqual(self.scheduler(allowlist=["filehelper"]).tick(), 0)
        self.assertEqual(self.sender.sent, [])
        self.assertEqual(self.store.list_pending(ROOM, ALICE), [])
        self.assertEqual(self.store.list_pending(OTHER_ROOM, ALICE), [])

    def test_allowlist_permits_listed_room(self):
        self.create()
        self.now = NOW + 600
        self.assertEqual(self.scheduler(allowlist=[ROOM]).tick(), 1)

    def test_inactive_plugin_or_disconnected_keeps_reminders(self):
        row = self.create()
        scheduler = self.scheduler()
        self.now = row.due_at
        self.active = False
        self.assertEqual(scheduler.tick(), 0)
        self.active, self.connected = True, False
        self.assertEqual(scheduler.tick(), 0)
        self.assertEqual(len(self.store.list_pending(ROOM, ALICE)), 1)
        self.connected = True
        self.now = row.due_at + 120
        self.assertEqual(scheduler.tick(), 1)
        self.assertEqual(self.sender.sent[0][1], "⏰ [延迟提醒] 关火（原定 15:10）")

    def test_inactive_plugin_reenabled_after_grace_expires(self):
        row = self.create()
        scheduler = self.scheduler()
        self.active = False
        self.now = row.due_at + GRACE_SECONDS + 1
        self.assertEqual(scheduler.tick(), 0)
        self.assertEqual(len(self.store.list_pending(ROOM, ALICE)), 1)
        self.active = True
        self.assertEqual(scheduler.tick(), 0)
        self.assertEqual(self.store.list_pending(ROOM, ALICE), [])
        self.assertEqual(self.sender.sent, [])

    def test_ticker_survives_errors_and_restarts_after_death(self):
        scheduler = self.scheduler(interval=0.01)
        calls = []
        original = scheduler.tick

        def flaky(now=None):
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("模拟数据库忙")
            if len(calls) == 2:
                raise SystemExit  # 线程级致命错误，线程退出
            return original(now)

        scheduler.tick = flaky
        scheduler.start()
        self.addCleanup(scheduler.stop)
        wait_for(lambda: len(calls) >= 2 and not scheduler.alive)
        self.assertTrue(scheduler.restart_if_dead())
        wait_for(lambda: len(calls) >= 4)
        self.assertTrue(scheduler.alive)
        self.assertFalse(scheduler.restart_if_dead())
        scheduler.stop()
        self.assertFalse(scheduler.alive)
        self.assertFalse(scheduler.restart_if_dead())

    def test_scheduler_requires_same_account(self):
        with self.assertRaises(ValueError):
            self.scheduler(store=self.new_store(OTHER_ROBOT))

    def test_logs_never_contain_content_or_ids(self):
        logging.disable(logging.NOTSET)
        self.create(content="秘密事项")
        self.create(minutes=1, creator=BOB, content="过期秘密")
        with self.assertLogs("Reminders", level="INFO") as logs:
            # 一条准点发送、一条超期作废、一条原生失败作废。
            self.scheduler().tick(now=NOW + 600 + 1)
            self.sender.failures.append(OSError("x"))
            self.create(minutes=20, creator=BOB, content="失败秘密")
            self.scheduler().tick(now=NOW + 1200)
        text = "\n".join(logs.output)
        for secret in ("秘密", ROOM, ALICE, BOB, ROBOT):
            self.assertNotIn(secret, text)
        self.assertIn("#1", text)


class ServiceAndContextTests(ReminderTestCase):
    def test_service_raises_send_not_attempted_before_native_call(self):
        service = ServiceFacade(text_delay={"enabled": False})
        with self.assertRaises(SendNotAttempted):
            service.send_at_text(ROOM, "x", [ALICE], expected_session=None)
        self.assertTrue(issubclass(SendNotAttempted, RuntimeError))

    def test_plugin_context_reminders(self):
        service = ServiceFacade(text_delay={"enabled": False})
        with self.assertRaises(CapabilityError):
            PluginContext("reminder", {}, service, []).reminders
        ctx = PluginContext("reminder", {}, service, [], member_store=self.members, reminder_store=self.store)
        self.assertIs(ctx.reminders, self.store)

    def test_commands_are_control_words(self):
        self.assertLessEqual({"/提醒", "/我的提醒", "/提醒列表", "/取消提醒", "/取消全部提醒"}, CONTROL)


class PrepareReminderTests(ReminderTestCase):
    def message(self, text, *, room=ROOM, at_self=False, **data):
        if at_self:
            data.update(at_members=(ROBOT,), is_at_self=True)
        return Message(text, room, sender_id=ALICE, is_group=True, session_id="s", data=data)

    def test_marks_commands_and_at_robot_with_reminder_word(self):
        for text, at_self in (("/提醒 10分钟后 喝水", False), ("/我的提醒", False), ("/提醒列表", False),
                              ("/取消提醒 1", False), ("/取消全部提醒", False),
                              ("@机器人\u2005 10分钟后提醒我关火", True), ("@机器人\u2005/提醒 15:00 喝水", True)):
            prepared = prepare_reminder(self.message(text, at_self=at_self), self.members)
            self.assertTrue(prepared.data.get("reminder_intent"), text)

    def test_leaves_mention_chat_and_other_messages_untouched(self):
        cases = [
            self.message("@机器人\u2005你怎么看？", at_self=True),          # 普通艾特闲聊交给 mention_chat
            self.message("10分钟后提醒我关火"),                              # 没有真实艾特机器人
            self.message("@机器人\u2005提醒", at_members=(BOB,), is_at_self=False),
            self.message("@机器人\u2005提醒我", room=OFF_ROOM, at_self=True),  # 群不在 groups
            self.message("/提醒 10分钟后 喝水", bot_generated=True),
            self.message("/提醒 10分钟后 喝水", dispatch_to_plugins=False),
            self.message("@机器人\u2005WXBOT_AT_CHAT_PROBE 提醒", at_self=True),
            self.message("/提醒10分钟后"),
            Message("/提醒 1 x", "wxid_friend", sender_id=ALICE, session_id="s"),
        ]
        for message in cases:
            self.assertIs(prepare_reminder(message, self.members), message, message.text)
        self.assertIs(prepare_reminder(cases[0], None), cases[0])


class MemoryAtBackend:
    def __init__(self):
        self.session_id = uuid.uuid4().hex
        self.pid = 1
        self.last_error = None
        self.sent = []

    def start(self, emit):
        self.emit = emit
        return True

    def send_text(self, chat_id, text):
        self.sent.append((chat_id, text, ()))
        return True

    def send_at_text(self, room, text, at_list):
        self.sent.append((room, text, tuple(at_list)))
        return True

    def stop(self):
        return {"dll_unloaded": True}


RECORDER = """
class Plugin:
    seen = []
    def on_message(self, message, ctx):
        Plugin.seen.append((message.text, bool(message.data.get("reminder_intent")), ctx.reminders.account))
        ctx.send_text("filehelper", "seen")
"""


class RuntimeIntegrationTests(ReminderTestCase):
    def make_app(self, *, enabled=True):
        plugin = self.root / "plugins" / "reminder"
        plugin.mkdir(parents=True)
        (plugin / "__init__.py").write_text(RECORDER, encoding="utf-8")
        config = {"backend": "native_text", "wechat_account": ROBOT, "members_config_file": "members.json",
                  "console": False, "reconnect_seconds": 0.1, "text_send_delay": {"enabled": False},
                  "plugins": {"reminder": {"enabled": enabled}}}
        (self.root / "bot_config.json").write_text(json.dumps(config), encoding="utf-8")
        self.backend = MemoryAtBackend()
        app = BotApplication(self.root, ConfigStore(self.root / "bot_config.json"), backend_factory=lambda: self.backend)
        return app

    def test_runtime_marks_intent_and_scheduler_sends_after_restart(self):
        app = self.make_app()
        thread = threading.Thread(target=app.run)
        thread.start()
        try:
            wait_for(lambda: app._connection_ready and app.plugins.is_active("reminder") and app.scheduler.alive)
            self.backend.emit(Message("/我的提醒", ROOM, sender_id=ALICE, is_group=True,
                                      session_id=self.backend.session_id, data={}))
            wait_for(lambda: any(s[0] == "filehelper" for s in self.backend.sent))
            record = app.plugins.records["reminder"]
            module_plugin = type(record.instance)
            self.assertEqual(module_plugin.seen, [("/我的提醒", True, ROBOT)])
            due = time.time() + 0.3
            app.reminders.create(ROOM, ALICE, due, "喝水", now=due - 61)
            wait_for(lambda: any(s[0] == ROOM for s in self.backend.sent))
            sent = [s for s in self.backend.sent if s[0] == ROOM]
            clock = datetime.fromtimestamp(due, BEIJING).strftime("%H:%M")
            self.assertEqual(sent, [(ROOM, f"@群友\u2005\n⏰ 提醒时间到：喝水（设定时间 {clock}）", (ALICE,))])
            # 调度线程意外退出后由主循环重启。
            original, dead = app.scheduler.tick, threading.Event()

            def fatal(now=None):
                if not dead.is_set():
                    dead.set()
                    raise SystemExit
                return original(now)

            app.scheduler.tick = fatal
            first = app.scheduler._thread
            wait_for(lambda: dead.is_set() and not first.is_alive())
            wait_for(lambda: app.scheduler.alive and app.scheduler._thread is not first)
        finally:
            app.stop_event.set()
            thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertFalse(app.scheduler.alive)

    def test_disabled_plugin_does_not_mark_or_send(self):
        app = self.make_app(enabled=False)
        app.service.bind(self.backend)
        app._connection_ready, app.backend = True, self.backend
        app.plugins.reload()
        try:
            self.assertFalse(app.plugins.is_active("reminder"))
            row = app.reminders.create(ROOM, ALICE, NOW + 600, "喝水", now=NOW)
            self.assertEqual(app.scheduler.tick(now=row.due_at), 0)
            self.assertEqual(self.backend.sent, [])
            self.assertEqual(len(app.reminders.list_pending(ROOM, ALICE)), 1)
        finally:
            app.plugins.shutdown()
            app.service.close()
            app.reminders.close()
            app.members.close()

    def test_ready_requires_matching_account(self):
        app = self.make_app()
        try:
            self.assertFalse(app._reminder_ready())
            app.service.bind(self.backend)
            app._connection_ready, app.backend = True, self.backend
            self.assertTrue(app._reminder_ready())
            self.backend.raw = type("Raw", (), {"self_wxid": OTHER_ROBOT})()
            self.assertFalse(app._reminder_ready())
            self.backend.raw.self_wxid = ROBOT
            self.assertTrue(app._reminder_ready())
        finally:
            app.service.close()
            app.reminders.close()
            app.members.close()


if __name__ == "__main__":
    unittest.main()
