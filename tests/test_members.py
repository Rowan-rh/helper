"""成员绑定、持久记录和插件共用接口验证，不连接微信。"""
import json
import logging
import tempfile
import threading
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, patch

from plugins.member_identity import Plugin
from wxbot_framework.members import MemberError, MemberStore
from wxbot_framework.sdk import Message, PluginContext, ServiceFacade
from wxbot_framework.runtime import BotApplication
from wxbot_framework.config import ConfigStore

ROOT = Path(__file__).resolve().parent.parent
SCRATCH = ROOT / ".runtime/tests"
ROOM = "12345678@chatroom"
ROBOT, SMALL, JING = "wxid_robot", "wxid_small", "wxid_jing"


class MemberTests(unittest.TestCase):
    def setUp(self):
        SCRATCH.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=SCRATCH)
        self.root = Path(self.temporary.name).resolve()
        self.assertTrue(self.root.is_relative_to(SCRATCH.resolve()))
        self.options = {"database_file": "data.sqlite3", "history_enabled": True,
                        "groups": {ROOM: [
                            {"key": "xiaomei", "name": "成员甲", "display_name": "成员甲昵称", "aliases": ["小号"], "wxid": SMALL},
                            {"key": "jing", "name": "成员乙", "display_name": "成员乙群名", "aliases": ["成员乙别名"], "wxid": None},
                            {"key": "yuting", "name": "成员丁", "display_name": "DemoMemberC", "wxid": None}]}}
        self.config_path = self.root / "members.json"
        self.write_options()
        self.stores = []
        self.store = self.create_store()

    def write_options(self):
        self.config_path.write_text(json.dumps(self.options, ensure_ascii=False), encoding="utf-8")

    def create_store(self, account=ROBOT):
        store = MemberStore(self.config_path, account)
        self.stores.append(store)
        return store

    def tearDown(self):
        for store in self.stores:
            store.close()
        self.assertTrue(self.root.is_relative_to(SCRATCH.resolve()))
        self.temporary.cleanup()

    def message(self, text="中文消息 🌸", sender=JING, sequence=1, **kwargs):
        return Message(text, ROOM, sender_id=sender, is_group=True, sequence=sequence, session_id="first", **kwargs)

    def test_configuration_check_does_not_create_database(self):
        self.assertFalse(self.store.path.exists())
        self.store.initialize()
        self.assertTrue(self.store.path.exists())

    def test_aliases_are_resolved_case_insensitively_and_small_is_bound(self):
        self.assertEqual(self.store.resolve("小号").wxid, SMALL)
        self.assertEqual(self.store.resolve("成员乙群名", ROOM).name, "成员乙")
        self.assertEqual(self.store.resolve("demomemberc", ROOM).name, "成员丁")
        self.assertIsNone(self.store.resolve("成员乙").wxid)

    def test_unknown_names_are_not_guessed(self):
        with self.assertRaises(MemberError):
            self.store.resolve("未登记的昵称")
        with self.assertRaises(MemberError):
            self.store.bind("未登记的昵称", ROOM, JING)

    def test_binding_survives_restart_and_null_profile_does_not_clear_it(self):
        self.store.bind("成员乙群名", ROOM, JING)
        self.store.close()
        reopened = self.create_store()
        self.assertEqual(reopened.resolve("成员乙", ROOM).wxid, JING)

    def test_conflicting_binding_does_not_replace_identity(self):
        self.store.bind("成员乙", ROOM, JING)
        self.assertEqual(self.store.bind("成员乙", ROOM, JING).wxid, JING)
        with self.assertRaises(MemberError):
            self.store.bind("成员乙", ROOM, "wxid_other")
        with self.assertRaises(MemberError):
            self.store.bind("成员丁", ROOM, JING)
        self.assertEqual(self.store.resolve("成员乙", ROOM).wxid, JING)

    def test_history_before_binding_becomes_queryable_by_alias(self):
        self.store.record(self.message())
        with self.assertRaises(MemberError):
            self.store.history("成员乙")
        self.store.bind("成员乙", ROOM, JING)
        rows = self.store.history("成员乙别名")
        self.assertEqual([row["text"] for row in rows], ["中文消息 🌸"])
        self.assertEqual(rows[0]["sender_id"], JING)
        self.assertEqual(datetime.fromisoformat(rows[0]["received_at"]).utcoffset(), timedelta(0))

    def test_history_is_isolated_by_robot_account_and_author(self):
        self.store.bind("成员乙", ROOM, JING)
        self.store.record(self.message("成员乙的话"))
        self.store.record(self.message("成员甲的话", sender=SMALL, sequence=2))
        second = self.create_store("wxid_second_robot")
        second.bind("成员乙", ROOM, JING)
        self.assertEqual(second.history("成员乙"), [])
        self.assertEqual([row["text"] for row in self.store.history("成员乙")], ["成员乙的话"])

    def test_private_other_group_and_wrong_account_are_not_archived(self):
        self.assertFalse(self.store.record(Message("私聊", JING, sender_id=JING, session_id="first")))
        self.assertFalse(self.store.record(Message("另一群", "99999@chatroom", sender_id=JING, is_group=True, session_id="first")))
        self.assertFalse(self.store.record(self.message(data={"self_wxid": "wxid_other_robot"})))

    def test_sequence_deduplication_and_new_session(self):
        self.assertTrue(self.store.record(self.message()))
        self.assertFalse(self.store.record(self.message()))
        self.assertTrue(self.store.record(Message("重连消息", ROOM, JING, is_group=True, sequence=1, session_id="second")))
        self.store.bind("成员乙", ROOM, JING)
        self.assertEqual(len(self.store.history("成员乙")), 2)

    def test_limit_and_time_filter_return_chronological_records(self):
        self.store.bind("成员乙", ROOM, JING)
        self.store.record(self.message("第一句", sequence=1))
        since = datetime.now(timezone.utc)
        self.store.record(self.message("第二句", sequence=2))
        self.assertEqual([r["text"] for r in self.store.history("成员乙", limit=1)], ["第二句"])
        self.assertEqual([r["text"] for r in self.store.history("成员乙", since=since)], ["第二句"])
        self.assertEqual([r["text"] for r in self.store.history("成员乙")], ["第一句", "第二句"])
        for limit in (True, 0, 1001):
            with self.assertRaises(MemberError):
                self.store.history("成员乙", limit=limit)
        with self.assertRaises(MemberError):
            self.store.history("成员乙", since=datetime.now())

    def test_disabled_history_keeps_member_directory(self):
        self.options["history_enabled"] = False
        self.write_options()
        disabled = self.create_store()
        self.assertEqual(disabled.resolve("成员甲").wxid, SMALL)
        self.assertFalse(disabled.record(self.message()))

    def test_same_alias_in_two_groups_requires_group(self):
        second_room = "99999@chatroom"
        self.options["groups"][second_room] = self.options["groups"][ROOM]
        self.write_options()
        multi = self.create_store()
        with self.assertRaises(MemberError):
            multi.resolve("成员乙")
        self.assertEqual(multi.resolve("成员乙", second_room).room, second_room)

    def test_configuration_rejects_duplicate_casefolded_aliases(self):
        self.options["groups"][ROOM][1]["aliases"] = ["demomemberc"]
        self.write_options()
        with self.assertRaises(MemberError):
            self.create_store()

    def test_recording_and_lookup_from_different_threads(self):
        self.store.bind("成员乙", ROOM, JING)
        errors = []
        def write(index):
            try:
                self.store.record(self.message(str(index), sequence=index))
                self.store.resolve("成员乙")
            except Exception as error:
                errors.append(error)
        threads = [threading.Thread(target=write, args=(index,)) for index in range(1, 13)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(3)
        self.assertEqual(errors, [])
        self.assertEqual(len(self.store.history("成员乙")), 12)

    def test_identity_plugin_binds_actual_message_author(self):
        ctx = Mock(members=self.store, logger=logging.getLogger("MembersTest"))
        plugin = Plugin()
        plugin.on_load(ctx)
        plugin.on_message(self.message("/绑定 成员乙"), ctx)
        self.assertEqual(self.store.resolve("成员乙").wxid, JING)
        ctx.reply.assert_called_once_with(unittest.mock.ANY, "已绑定：成员乙。")
        plugin.on_message(self.message("/绑定 成员丁"), ctx)
        self.assertIsNone(self.store.resolve("成员丁").wxid)

    def test_context_auto_mention_uses_group_name_instead_of_personal_alias(self):
        self.store.bind("成员乙", ROOM, JING)
        backend = Mock(session_id="first")
        service = ServiceFacade(text_delay={"enabled": False})
        service.bind(backend)
        try:
            ctx = PluginContext("example", {}, service, [], member_store=self.store)
            self.assertEqual(ctx.members.resolve("成员乙").wxid, JING)
            ctx.reply_at(self.message(), "回复")
            backend.send_at_text.assert_called_once_with(ROOM, "@成员乙群名\u2005\n回复", (JING,))
        finally:
            service.close()

    def test_runtime_archives_before_plugin_dispatch(self):
        (self.root / "plugins").mkdir()
        path = self.root / "bot_config.json"
        path.write_text(json.dumps({"backend": "native_text", "wechat_account": ROBOT,
                                    "members_config_file": self.config_path.name}), encoding="utf-8")
        app = BotApplication(self.root, ConfigStore(path))
        try:
            app.members.bind("成员乙", ROOM, JING)
            def verify(message):
                self.assertEqual(app.members.history("成员乙")[0]["text"], message.text)
            with patch.object(app.plugins, "dispatch", side_effect=verify):
                app._emit(self.message())
        finally:
            app.stop()

    def test_free_registration_accepts_new_name_and_links_existing_history(self):
        self.store.record(self.message("绑定前说过的话"))
        member = self.store.register("测试成员", ROOM, JING)
        self.assertEqual(member.wxid, JING)
        self.assertEqual(self.store.history("测试成员")[0]["text"], "绑定前说过的话")

    def test_same_account_can_rename_without_losing_aliases_or_history(self):
        self.store.register("测试成员", ROOM, JING)
        self.store.record(self.message("原有记录"))
        self.store.register("成员丙", ROOM, JING)
        self.assertEqual(self.store.resolve("测试成员").name, "成员丙")
        self.assertEqual(self.store.resolve("成员丙").wxid, JING)
        self.assertEqual(self.store.resolve("成员丙").display_name, "成员丙")
        self.assertEqual(self.store.history("成员丙")[0]["text"], "原有记录")
        self.assertEqual(len([m for m in self.store.list_members() if m.wxid == JING]), 1)

    def test_another_account_cannot_take_existing_nickname(self):
        self.store.register("测试成员", ROOM, JING)
        with self.assertRaises(MemberError):
            self.store.register("测试成员", ROOM, SMALL)
        self.assertEqual(self.store.resolve("测试成员").wxid, JING)

    def test_registered_name_survives_config_seeding_on_restart(self):
        self.store.register("成员乙", ROOM, JING)
        self.store.register("测试成员", ROOM, JING)
        self.store.close()
        reopened = self.create_store()
        self.assertEqual(reopened.resolve("成员乙").name, "测试成员")
        self.assertEqual(reopened.resolve("测试成员").wxid, JING)

    def test_free_registration_still_limits_groups_and_name_shape(self):
        with self.assertRaises(MemberError):
            self.store.register("测试成员", "99999@chatroom", JING)
        for name in ("", "x" * 65, "喵\n喵", "喵\0喵"):
            with self.subTest(name=name), self.assertRaises(MemberError):
                self.store.register(name, ROOM, JING)

    def test_group_binding_overrides_legacy_display_and_survives_restart(self):
        self.store.register("成员甲", ROOM, SMALL)
        self.assertEqual(self.store.display_name(SMALL, ROOM), "成员甲")
        self.store.close()
        self.store = self.create_store()
        self.assertEqual(self.store.display_name(SMALL, ROOM), "成员甲")
        self.assertTrue(self.store.resolve("成员甲", ROOM).self_named)

    def test_wechat_default_and_bindings_are_separate_in_each_group(self):
        other_room = "87654321@chatroom"
        self.options["groups"][other_room] = []
        self.write_options()
        store = self.create_store()
        store.register("成员甲", ROOM, SMALL)
        store.remember_wechat_name(SMALL, "DemoUser")
        message = Message("新群的原话", other_room, SMALL, is_group=True, session_id="new", sequence=1)
        store.record(message)
        self.assertEqual(store.display_name(SMALL, other_room), "DemoUser")
        self.assertEqual(store.list_members(other_room), [])
        self.assertEqual(store.today_snapshot("DemoUser", other_room).member.wxid, SMALL)
        backend = Mock(session_id="new")
        service = ServiceFacade(text_delay={"enabled": False})
        service.bind(backend)
        try:
            ctx = PluginContext("test", {}, service, [], member_store=store)
            ctx.reply_at(message, "默认回复")
            self.assertEqual(backend.send_at_text.call_args.args[1], "@DemoUser\u2005\n默认回复")
            store.register("新群名字", other_room, SMALL)
            ctx.reply_at(message, "已绑定", display_name="旧的硬编码名字")
            self.assertEqual(backend.send_at_text.call_args.args[1], "@新群名字\u2005\n已绑定")
            self.assertEqual(store.display_name(SMALL, ROOM), "成员甲")
            self.assertEqual(store.today_snapshot("新群名字", other_room).messages[0]["text"], "新群的原话")
        finally:
            service.close()

    def test_native_wechat_name_updates_default_without_overwriting_group_binding(self):
        other_room = "87654321@chatroom"
        self.options["groups"][other_room] = []
        self.write_options()
        store = self.create_store()
        store.register("成员甲", ROOM, SMALL)
        for sequence, name in ((1, "DemoUser"), (2, "新的微信名")):
            msg = Message("原文", other_room, SMALL, is_group=True, session_id="new", sequence=sequence,
                          data={"sender_wechat_name": name, "self_wxid": ROBOT})
            store.record(msg)
            self.assertEqual(store.display_name(SMALL, other_room), name)
        self.assertEqual(store.display_name(SMALL, ROOM), "成员甲")
        self.assertEqual(store.today_group_snapshot(other_room).messages[-1]["speaker_name"], "新的微信名")
        self.assertEqual(store.conversation_snapshot(msg).speaker_name, "新的微信名")
        store.clear(other_room)
        self.assertEqual(store.today_snapshot("新的微信名", other_room).messages, ())

    def test_duplicate_default_wechat_names_do_not_mix_member_identities(self):
        self.options["groups"][ROOM] = []
        self.write_options()
        store = self.create_store()
        for index, wxid in enumerate((SMALL, JING), 1):
            store.record(self.message("不同成员", sender=wxid, sequence=index,
                                      data={"sender_wechat_name": "同名"}))
        with self.assertRaisesRegex(MemberError, "多个成员"):
            store.today_snapshot("同名", ROOM)
        store.register("指定成员", ROOM, SMALL)
        self.assertEqual(store.today_snapshot("指定成员", ROOM).member.wxid, SMALL)

    def test_plugin_free_registration_and_usage(self):
        ctx = Mock(members=self.store, logger=logging.getLogger("MembersTest"))
        plugin = Plugin()
        plugin.on_load(ctx)
        plugin.on_message(self.message("/绑定 测试成员"), ctx)
        self.assertEqual(self.store.resolve("测试成员").wxid, JING)
        plugin.on_message(self.message("/绑定"), ctx)
        self.assertEqual(ctx.reply.call_args.args[1], "用法：/绑定 昵称")
        plugin.on_message(self.message(" /绑定\t成员丙  "), ctx)
        self.assertEqual(self.store.resolve("成员丙").wxid, JING)


class AllowAllGroupsTests(unittest.TestCase):
    def setUp(self):
        SCRATCH.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=SCRATCH)
        # addCleanup 后进先出：临时目录最先注册，最后删除，sqlite 句柄一定先关闭。
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.config_path = self.root / "members.json"
        self.stores = []

    def tearDown(self):
        # 递归清理只针对本测试创建、已核对位于工作区的临时目录。
        self.assertTrue(self.root.resolve().is_relative_to(SCRATCH.resolve()))
        # 临时目录删除与成员目录句柄关闭由 addCleanup 统一处理（后进先出）。

    def create_store(self, options, account=ROBOT, now=None):
        self.config_path.write_text(json.dumps(options, ensure_ascii=False), encoding="utf-8")
        store = MemberStore(self.config_path, account, now=now)
        # 句柄关闭必须早于临时目录删除，否则 Windows 上删目录会报 WinError 32。
        self.addCleanup(store.close)
        self.stores.append(store)
        return store

    def test_allow_all_groups_config_validation(self):
        base_options = {
            "database_file": "data.sqlite3",
            "groups": {ROOM: []},
        }
        for invalid_val in ("true", "false", 1, 0, None, [], {}):
            opts = dict(base_options, allow_all_groups=invalid_val)
            with self.assertRaisesRegex(MemberError, "allow_all_groups 必须是布尔值"):
                self.create_store(opts)

        # 默认缺省为 False
        store_default = self.create_store(base_options)
        self.assertFalse(store_default.allow_all_groups)
        self.assertFalse(store_default.groups.allow_all)

        # 显式 True
        store_true = self.create_store(dict(base_options, allow_all_groups=True))
        self.assertTrue(store_true.allow_all_groups)
        self.assertTrue(store_true.groups.allow_all)

    def test_groupset_behavior(self):
        from wxbot_framework.members import GroupSet

        # allow_all=False: 严格集合行为
        gs_false = GroupSet([ROOM], allow_all=False)
        self.assertIn(ROOM, gs_false)
        self.assertNotIn("dynamic_room@chatroom", gs_false)
        self.assertNotIn("wxid_user", gs_false)
        gs_false.discard(ROOM)
        self.assertNotIn(ROOM, gs_false)

        # allow_all=True
        gs_true = GroupSet([ROOM], allow_all=True)
        self.assertIn(ROOM, gs_true)
        self.assertIn("dynamic_room@chatroom", gs_true)
        self.assertIn("any_valid_group123@chatroom", gs_true)

        # 非法群 ID 或私聊非群
        self.assertNotIn("wxid_user", gs_true)
        self.assertNotIn("filehelper", gs_true)
        self.assertNotIn(None, gs_true)
        self.assertNotIn(12345, gs_true)
        self.assertNotIn("invalid name with space@chatroom", gs_true)
        self.assertNotIn("x" * 120 + "@chatroom", gs_true)
        self.assertNotIn("", gs_true)
        self.assertNotIn("@chatroom", gs_true)

        # 集合通用方法
        self.assertEqual(len(gs_true), 1)
        self.assertEqual(list(gs_true), [ROOM])
        gs_copy = gs_true.copy()
        self.assertTrue(gs_copy.allow_all)
        self.assertIn("other@chatroom", gs_copy)

    def test_dynamic_group_lifecycle_record_snapshot_personality_clear_rollover(self):
        fixed_now = [datetime(2026, 10, 24, 12, 0, 0, tzinfo=timezone.utc)]
        options = {
            "database_file": "data.sqlite3",
            "history_enabled": True,
            "daily_clear": True,
            "allow_all_groups": True,
            "groups": {ROOM: []},
        }
        store = self.create_store(options, now=lambda: fixed_now[0])
        store.initialize()

        dynamic_room = "new_dynamic_group@chatroom"
        user_wxid = "wxid_dynamic_user"

        # 1. 未配置的群不在静态 options["groups"]，但在 store.groups 中
        self.assertNotIn(dynamic_room, options["groups"])
        self.assertIn(dynamic_room, store.groups)

        # 2. record 正常写入，自动懒初始化 chat_buffers
        msg1 = Message("第一句消息", dynamic_room, sender_id=user_wxid, is_group=True,
                       sequence=1, session_id="sess_1")
        self.assertTrue(store.record(msg1))

        # 3. conversation_snapshot 正常获取且 state[0] 不为 None
        conv_snap = store.conversation_snapshot(msg1)
        self.assertEqual(conv_snap.room, dynamic_room)
        self.assertEqual(conv_snap.generation, 0)
        self.assertEqual(conv_snap.current, "第一句消息")

        # 4. today_group_snapshot 正常读取本群今日消息
        msg2 = Message("第二句消息", dynamic_room, sender_id=user_wxid, is_group=True,
                       sequence=2, session_id="sess_1")
        store.record(msg2)
        group_snap = store.today_group_snapshot(dynamic_room)
        self.assertEqual(len(group_snap.messages), 2)
        self.assertEqual(group_snap.generation, 0)
        self.assertTrue(store.snapshot_is_current(group_snap))

        # 5. 设置群人格
        store.set_personality(dynamic_room, "嘴臭")
        p_state = store.personality(dynamic_room)
        self.assertEqual(p_state.name, "嘴臭")
        self.assertEqual(p_state.generation, 1)

        # 6. clear 清空消息并递增 generation
        cleared = store.clear(dynamic_room)
        self.assertEqual(cleared, 2)
        group_snap_after_clear = store.today_group_snapshot(dynamic_room)
        self.assertEqual(len(group_snap_after_clear.messages), 0)
        self.assertEqual(group_snap_after_clear.generation, 1)
        self.assertFalse(store.snapshot_is_current(group_snap))

        # 7. rollover 跨天清理覆盖动态群
        msg3 = Message("清空后的新消息", dynamic_room, sender_id=user_wxid, is_group=True,
                       sequence=3, session_id="sess_2")
        store.record(msg3)
        self.assertEqual(len(store.today_group_snapshot(dynamic_room).messages), 1)

        # 模拟跨天：前进一天
        fixed_now[0] += timedelta(days=1)
        self.assertTrue(store.rollover())
        # daily_clear=True 会清理前一天的消息，且 generation 递增
        group_snap_next_day = store.today_group_snapshot(dynamic_room)
        self.assertEqual(len(group_snap_next_day.messages), 0)
        self.assertEqual(group_snap_next_day.generation, 2)

    def test_mention_chat_and_reminder_in_dynamic_group_and_vip_isolation(self):
        from plugins.mention_chat import Plugin as MentionChatPlugin
        from plugins.reminder import Plugin as ReminderPlugin
        from wxbot_framework.reminders import ReminderStore, prepare_reminder
        from wxbot_framework.llm import ChatResult

        vip_user = "wxid_vip_boss"
        normal_user = "wxid_normal_user"
        dynamic_room = "dynamic_branch_test@chatroom"

        options = {
            "database_file": "data.sqlite3",
            "history_enabled": True,
            "daily_clear": True,
            "allow_all_groups": True,
            "agent_vip_wxids": [vip_user],
            "groups": {ROOM: []},
        }
        store = self.create_store(options)
        store.initialize()

        # VIP 校验
        self.assertTrue(store.is_vip(vip_user))
        self.assertFalse(store.is_vip(normal_user))

        # 设置 ServiceFacade 和 MockBackend
        sent_messages = []
        backend = Mock()
        backend.session_id = "test_session"

        def mock_send_text(to_wxid, content, **kw):
            sent_messages.append(("text", to_wxid, content))
            return True

        def mock_send_at(room, content, at_list, **kw):
            sent_messages.append(("at", room, content, at_list))
            return True

        backend.send_text = Mock(side_effect=mock_send_text)
        backend.send_at_text = Mock(side_effect=mock_send_at)

        service = ServiceFacade(text_delay={"enabled": False})
        service.bind(backend)

        # 模拟 LLM
        fake_llm = Mock()
        fake_llm.complete_chat.return_value = ChatResult("你好！我是机器人助手。", 10, 10, None)
        fake_llm.enable_tools = False

        # 初始化 mention_chat
        chat_plugin = MentionChatPlugin()
        chat_ctx = PluginContext("mention_chat", {"agent": {}}, service, [], member_store=store)
        chat_ctx._llm = fake_llm
        chat_plugin.on_load(chat_ctx)
        self.addCleanup(lambda: chat_plugin.on_unload(chat_ctx))

        # (a) 普通非 VIP 群成员在动态新群发送 @机器人
        group_msg = Message(
            "@机器人\u2005你好", dynamic_room, sender_id=normal_user, is_group=True,
            sequence=1, session_id="test_session",
            data={"is_at_self": True, "at_members": (ROBOT,), "self_wxid": ROBOT}
        )
        store.record(group_msg)
        chat_plugin.on_message(group_msg, chat_ctx)
        self.assertEqual(len(sent_messages), 1)
        self.assertEqual(sent_messages[0][0], "at")
        self.assertEqual(sent_messages[0][1], dynamic_room)
        self.assertIn("你好！我是机器人助手。", sent_messages[0][2])
        self.assertEqual(sent_messages[0][3], (normal_user,))

        # (b) 普通非 VIP 私聊发送消息 -> 静默忽略，不响应
        sent_messages.clear()
        private_msg_normal = Message(
            "私聊你好", normal_user, sender_id=normal_user, is_group=False,
            sequence=1, session_id="test_session",
            data={"self_wxid": ROBOT}
        )
        chat_plugin.on_message(private_msg_normal, chat_ctx)
        self.assertEqual(len(sent_messages), 0)

        # (c) VIP 用户私聊发送消息 -> 正常响应
        private_msg_vip = Message(
            "VIP私聊你好", vip_user, sender_id=vip_user, is_group=False,
            sequence=1, session_id="test_session",
            data={"self_wxid": ROBOT}
        )
        chat_plugin.on_message(private_msg_vip, chat_ctx)
        self.assertEqual(len(sent_messages), 1)
        self.assertEqual(sent_messages[0][0], "text")
        self.assertEqual(sent_messages[0][1], vip_user)
        self.assertIn("你好！我是机器人助手。", sent_messages[0][2])

        # (d) 动态群中普通非 VIP 发送提醒指令
        sent_messages.clear()
        reminder_store = ReminderStore(self.root / ".runtime/reminders.sqlite3", ROBOT, self.root)
        self.addCleanup(reminder_store.close)

        reminder_ctx = PluginContext("reminder", {}, service, [], member_store=store, reminder_store=reminder_store)
        base_timestamp = datetime(2026, 10, 24, 15, 0, 0, tzinfo=timezone.utc).timestamp()
        reminder_plugin = ReminderPlugin(clock=lambda: base_timestamp)
        reminder_plugin.on_load(reminder_ctx)

        reminder_raw_msg = Message(
            "@机器人\u200510分钟后提醒我喝水", dynamic_room, sender_id=normal_user, is_group=True,
            sequence=2, session_id="test_session",
            data={"is_at_self": True, "at_members": (ROBOT,), "self_wxid": ROBOT}
        )
        prepared_msg = prepare_reminder(reminder_raw_msg, store, agent_fallback=True)
        self.assertTrue(prepared_msg.data.get("reminder_intent"))

        reminder_plugin.on_message(prepared_msg, reminder_ctx)
        self.assertEqual(len(sent_messages), 1)
        self.assertEqual(sent_messages[0][0], "text")
        self.assertEqual(sent_messages[0][1], dynamic_room)
        self.assertIn("已设置提醒 [#1]", sent_messages[0][2])
        self.assertIn("喝水", sent_messages[0][2])

    def test_allow_all_groups_empty_groups_config_lifecycle(self):
        # 覆盖最典型的默认配置：groups 为空字典 {}，搭配 allow_all_groups: true
        fixed_now = [datetime(2026, 10, 24, 10, 0, 0, tzinfo=timezone.utc)]
        options = {
            "database_file": "data.sqlite3",
            "history_enabled": True,
            "daily_clear": True,
            "allow_all_groups": True,
            "groups": {},
        }
        store = self.create_store(options, now=lambda: fixed_now[0])
        store.initialize()

        self.assertEqual(len(store.groups), 0)
        dynamic_room = "empty_cfg_room@chatroom"
        self.assertIn(dynamic_room, store.groups)

        msg = Message("空配置测试消息", dynamic_room, sender_id="wxid_user_1", is_group=True, sequence=1, session_id="s1")
        self.assertTrue(store.record(msg))
        snap = store.today_group_snapshot(dynamic_room)
        self.assertEqual(len(snap.messages), 1)
        self.assertEqual(snap.generation, 0)

        # 跨天 rollover 正常清理
        fixed_now[0] += timedelta(days=1)
        self.assertTrue(store.rollover())
        snap_next = store.today_group_snapshot(dynamic_room)
        self.assertEqual(len(snap_next.messages), 0)
        self.assertEqual(snap_next.generation, 1)

    def test_allow_all_groups_false_default_behavior(self):
        options = {
            "database_file": "data.sqlite3",
            "history_enabled": True,
            "daily_clear": True,
            "allow_all_groups": False,
            "groups": {ROOM: []},
        }
        store = self.create_store(options)
        store.initialize()

        dynamic_room = "unknown_group@chatroom"
        self.assertNotIn(dynamic_room, store.groups)

        msg = Message("新群消息", dynamic_room, sender_id=SMALL, is_group=True, sequence=1, session_id="s1")
        # 未配置的群不入库
        self.assertFalse(store.record(msg))

        # 未配置的群快照抛异常
        with self.assertRaisesRegex(MemberError, "这个群尚未启用聊天记录"):
            store.today_group_snapshot(dynamic_room)

        with self.assertRaisesRegex(MemberError, "这个群尚未启用聊天记录"):
            store.clear(dynamic_room)

        with self.assertRaisesRegex(MemberError, "在已启用的群中切换"):
            store.set_personality(dynamic_room, "嘴臭")
