"""长期记忆存储与记忆命令测试。

- 数据库路径强校验位于 .runtime/ 下。
- 长期记忆按 (account, room) 隔离。
- 增、替换、删、列出、清空与快照文本。
- 容量限制：50 条 / 200 字 / 6000 字，内容去重，满额返回提示不淘汰。
- 幂等性：重复添加、重复删除安全。
- 命令鉴权：/记忆、/删除记忆、/清空记忆 仅 VIP 或主号生效，其他人静默忽略；/重置对话 对所有人开放。
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from wxbot_framework.agent import ConversationStore, SessionKey
from wxbot_framework.memory import (
    GroupMemoryStore,
    MemoryConfig,
    execute_memory_command,
    validate_runtime_db_path,
)

ACCOUNT = "wxid_bot_test"
OTHER_ACCOUNT = "wxid_bot_other"
ROOM = "10001@chatroom"
OTHER_ROOM = "10002@chatroom"
VIP_SENDER = "wxid_vip_user"
NORMAL_SENDER = "wxid_normal_user"


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.runtime_dir = Path(self.temp_dir.name) / ".runtime"
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.runtime_dir / "member_data.sqlite3"
        self.store = GroupMemoryStore(self.db_path, ACCOUNT)

    def tearDown(self):
        self.store.close()
        self.temp_dir.cleanup()

    def test_database_path_must_be_under_runtime_directory(self):
        # 位于 .runtime 下的路径合法
        self.assertEqual(validate_runtime_db_path(self.db_path), self.db_path.resolve())

        # 不在 .runtime 下的路径被强校验拒绝
        bad_path = Path(self.temp_dir.name) / "outside.sqlite3"
        with self.assertRaises(ValueError) as ctx:
            validate_runtime_db_path(bad_path)
        self.assertIn(".runtime", str(ctx.exception))

        # 文件名为 .runtime 但没有 .runtime 父目录的路径也被拒绝
        with self.assertRaises(ValueError):
            validate_runtime_db_path(Path(self.temp_dir.name) / ".runtime")

        # 尝试使用不合法路径初始化 Store
        with self.assertRaises(ValueError):
            GroupMemoryStore(bad_path, ACCOUNT)

    def test_memory_config_validation(self):
        cfg = MemoryConfig(max_entries=30, max_entry_chars=150, max_total_chars=4000)
        self.assertEqual(cfg.max_entries, 30)
        self.assertEqual(cfg.max_entry_chars, 150)
        self.assertEqual(cfg.max_total_chars, 4000)

        with self.assertRaises(ValueError):
            MemoryConfig(max_entries=0)
        with self.assertRaises(ValueError):
            MemoryConfig(max_entry_chars=-1)
        with self.assertRaises(ValueError):
            MemoryConfig(max_total_chars="6000")  # type: ignore

    def test_add_list_and_snapshot(self):
        res1 = self.store.add(ACCOUNT, ROOM, "每周五晚 20:00 语音开黑", created_by=VIP_SENDER)
        self.assertTrue(res1["ok"])
        self.assertEqual(res1["entry_id"], 1)
        self.assertEqual(res1["count"], 1)
        self.assertFalse(res1["already_existed"])

        res2 = self.store.add(ACCOUNT, ROOM, "小明海鲜过敏，聚餐避开海鲜", created_by=VIP_SENDER)
        self.assertTrue(res2["ok"])
        self.assertEqual(res2["entry_id"], 2)
        self.assertEqual(res2["count"], 2)

        memories = self.store.list_memories(ACCOUNT, ROOM)
        self.assertEqual(len(memories), 2)
        self.assertEqual(memories[0].entry_id, 1)
        self.assertEqual(memories[0].content, "每周五晚 20:00 语音开黑")
        self.assertEqual(memories[1].entry_id, 2)
        self.assertEqual(memories[1].content, "小明海鲜过敏，聚餐避开海鲜")

        snapshot = self.store.snapshot_text(ACCOUNT, ROOM)
        expected = "#1 每周五晚 20:00 语音开黑\n#2 小明海鲜过敏，聚餐避开海鲜"
        self.assertEqual(snapshot, expected)

    def test_content_deduplication_is_idempotent(self):
        res1 = self.store.add(ACCOUNT, ROOM, "群规第一条：禁止发广告")
        self.assertTrue(res1["ok"])
        self.assertEqual(res1["entry_id"], 1)
        self.assertFalse(res1["already_existed"])

        # 再次添加相同内容，幂等返回已有条目，条数不增加
        res2 = self.store.add(ACCOUNT, ROOM, "  群规第一条：禁止发广告  ")
        self.assertTrue(res2["ok"])
        self.assertEqual(res2["entry_id"], 1)
        self.assertTrue(res2["already_existed"])
        self.assertEqual(res2["count"], 1)
        self.assertEqual(len(self.store.list_memories(ACCOUNT, ROOM)), 1)

    def test_content_validation(self):
        # 空内容
        res_empty = self.store.add(ACCOUNT, ROOM, "   ")
        self.assertFalse(res_empty["ok"])
        self.assertIn("不能为空", res_empty["error"])

        # 包含换行符
        res_nl = self.store.add(ACCOUNT, ROOM, "第一行\n第二行")
        self.assertFalse(res_nl["ok"])
        self.assertIn("换行", res_nl["error"])

        # 超出单条字数
        long_text = "字" * 201
        res_long = self.store.add(ACCOUNT, ROOM, long_text)
        self.assertFalse(res_long["ok"])
        self.assertIn("单条记忆不能超过", res_long["error"])

    def test_capacity_limits_and_no_auto_eviction(self):
        small_config = MemoryConfig(max_entries=3, max_entry_chars=50, max_total_chars=100)
        store = GroupMemoryStore(self.db_path, ACCOUNT, config=small_config)

        store.add(ACCOUNT, ROOM, "条目一")
        store.add(ACCOUNT, ROOM, "条目二")
        store.add(ACCOUNT, ROOM, "条目三")
        self.assertEqual(len(store.list_memories(ACCOUNT, ROOM)), 3)

        # 达到条数上限 3 条，添加第 4 条时返回错误提示与 hint，旧条目不淘汰
        res4 = store.add(ACCOUNT, ROOM, "条目四")
        self.assertFalse(res4["ok"])
        self.assertIn("已达上限（3/3）", res4["error"])
        self.assertIn("hint", res4)

        # 旧记忆完整保留
        memories = store.list_memories(ACCOUNT, ROOM)
        self.assertEqual([m.content for m in memories], ["条目一", "条目二", "条目三"])

        # 测试总字数上限
        store.clear(ACCOUNT, ROOM)
        store.add(ACCOUNT, ROOM, "A" * 45)
        store.add(ACCOUNT, ROOM, "B" * 45)
        # 当前 90 字，再加 20 字超过 100 字上限
        res_chars = store.add(ACCOUNT, ROOM, "C" * 20)
        self.assertFalse(res_chars["ok"])
        self.assertIn("总字数已达上限", res_chars["error"])

    def test_replace(self):
        self.store.add(ACCOUNT, ROOM, "每周五跑团")
        self.store.add(ACCOUNT, ROOM, "聚餐去火锅店")

        # 替换不存在的条目
        res_notfound = self.store.replace(ACCOUNT, ROOM, 999, "新内容")
        self.assertFalse(res_notfound["ok"])
        self.assertIn("未找到记忆 #999", res_notfound["error"])

        # 正常替换
        res = self.store.replace(ACCOUNT, ROOM, 2, "聚餐改为烤肉店")
        self.assertTrue(res["ok"])
        self.assertEqual(res["entry_id"], 2)
        self.assertEqual(res["status"], "replaced")

        memories = self.store.list_memories(ACCOUNT, ROOM)
        self.assertEqual(memories[1].content, "聚餐改为烤肉店")

        # 幂等替换：替换为同样的内容
        res_same = self.store.replace(ACCOUNT, ROOM, 2, "聚餐改为烤肉店")
        self.assertTrue(res_same["ok"])
        self.assertEqual(res_same["status"], "unchanged")

        # 替换内容与另一条目冲突
        res_dup = self.store.replace(ACCOUNT, ROOM, 2, "每周五跑团")
        self.assertFalse(res_dup["ok"])
        self.assertIn("重复", res_dup["error"])

    def test_delete_and_idempotency(self):
        self.store.add(ACCOUNT, ROOM, "条目1")
        self.store.add(ACCOUNT, ROOM, "条目2")

        # 正常删除
        res1 = self.store.delete(ACCOUNT, ROOM, 1)
        self.assertTrue(res1["ok"])
        self.assertEqual(res1["entry_id"], 1)
        self.assertTrue(res1["deleted"])
        self.assertEqual(res1["count"], 1)

        # 重复删除相同条目（幂等）
        res2 = self.store.delete(ACCOUNT, ROOM, 1)
        self.assertTrue(res2["ok"])
        self.assertEqual(res2["entry_id"], 1)
        self.assertFalse(res2["deleted"])
        self.assertEqual(res2["count"], 1)

        # 槽位重用：新条目复用已释放的最小未使用正整数编号 1
        res3 = self.store.add(ACCOUNT, ROOM, "条目3")
        self.assertTrue(res3["ok"])
        self.assertEqual(res3["entry_id"], 1)

    def test_clear_and_isolation(self):
        self.store.add(ACCOUNT, ROOM, "群1记忆")
        self.store.add(ACCOUNT, OTHER_ROOM, "群2记忆")

        other_store = GroupMemoryStore(self.db_path, OTHER_ACCOUNT)
        other_store.add(OTHER_ACCOUNT, ROOM, "账号2记忆")

        # 隔离校验
        self.assertEqual([m.content for m in self.store.list_memories(ACCOUNT, ROOM)], ["群1记忆"])
        self.assertEqual([m.content for m in self.store.list_memories(ACCOUNT, OTHER_ROOM)], ["群2记忆"])
        self.assertEqual([m.content for m in other_store.list_memories(OTHER_ACCOUNT, ROOM)], ["账号2记忆"])

        # 清空群1
        count = self.store.clear(ACCOUNT, ROOM)
        self.assertEqual(count, 1)
        self.assertEqual(self.store.list_memories(ACCOUNT, ROOM), ())
        # 其他群和账号不受影响
        self.assertEqual(len(self.store.list_memories(ACCOUNT, OTHER_ROOM)), 1)
        self.assertEqual(len(other_store.list_memories(OTHER_ACCOUNT, ROOM)), 1)
        other_store.close()

    def test_execute_memory_command_permissions(self):
        conv_store = ConversationStore()
        # 预设一条对话上下文
        handle = conv_store.load(SessionKey.group(ACCOUNT, ROOM, NORMAL_SENDER), 1)
        conv_store.commit(handle, "你好", "我是助手")
        self.assertEqual(len(conv_store.history(SessionKey.group(ACCOUNT, ROOM, NORMAL_SENDER), 1)), 1)

        # 1. 非 VIP、非主号执行记忆类命令：静默忽略（返回 None）
        self.assertIsNone(execute_memory_command("/记忆", ACCOUNT, ROOM, NORMAL_SENDER,
                                                 is_vip=False, is_self=False, memory_store=self.store))
        self.assertIsNone(execute_memory_command("/删除记忆 1", ACCOUNT, ROOM, NORMAL_SENDER,
                                                 is_vip=False, is_self=False, memory_store=self.store))
        self.assertIsNone(execute_memory_command("/清空记忆", ACCOUNT, ROOM, NORMAL_SENDER,
                                                 is_vip=False, is_self=False, memory_store=self.store))

        # 2. VIP 执行 /记忆：空记忆提示
        res = execute_memory_command("/记忆", ACCOUNT, ROOM, VIP_SENDER,
                                     is_vip=True, is_self=False, memory_store=self.store)
        self.assertEqual(res, "本群暂无长期记忆。")

        # 添加两条记忆
        self.store.add(ACCOUNT, ROOM, "周五桌游")
        self.store.add(ACCOUNT, ROOM, "周日聚餐")

        # VIP 查看记忆
        res = execute_memory_command("/记忆", ACCOUNT, ROOM, VIP_SENDER,
                                     is_vip=True, is_self=False, memory_store=self.store)
        self.assertIn("本群长期记忆（2/50）：", res)
        self.assertIn("#1 周五桌游", res)
        self.assertIn("#2 周日聚餐", res)

        # 删除记忆参数错误
        res_err = execute_memory_command("/删除记忆", ACCOUNT, ROOM, VIP_SENDER,
                                         is_vip=True, is_self=False, memory_store=self.store)
        self.assertEqual(res_err, "用法：/删除记忆 编号")
        res_err2 = execute_memory_command("/删除记忆 abc", ACCOUNT, ROOM, VIP_SENDER,
                                          is_vip=True, is_self=False, memory_store=self.store)
        self.assertEqual(res_err2, "用法：/删除记忆 编号")

        # 删除不存在的编号
        res_nf = execute_memory_command("/删除记忆 99", ACCOUNT, ROOM, VIP_SENDER,
                                        is_vip=True, is_self=False, memory_store=self.store)
        self.assertEqual(res_nf, "未找到记忆 #99，请发送 /记忆 查看。")

        # 主号本人删除记忆 #1
        res_del = execute_memory_command("/删除记忆 1", ACCOUNT, ROOM, ACCOUNT,
                                         is_vip=False, is_self=True, memory_store=self.store)
        self.assertEqual(res_del, "已删除记忆 #1。")
        self.assertEqual(len(self.store.list_memories(ACCOUNT, ROOM)), 1)

        # 清空记忆
        res_clear = execute_memory_command("/清空记忆", ACCOUNT, ROOM, VIP_SENDER,
                                           is_vip=True, is_self=False, memory_store=self.store)
        self.assertEqual(res_clear, "已清空本群长期记忆（共 1 条）。")
        self.assertEqual(len(self.store.list_memories(ACCOUNT, ROOM)), 0)

        # 3. /重置对话：任何合法发送者（包括非 VIP）均可触发
        res_reset = execute_memory_command("/重置对话", ACCOUNT, ROOM, NORMAL_SENDER,
                                           is_vip=False, is_self=False,
                                           memory_store=self.store, conversation_store=conv_store)
        self.assertEqual(res_reset, "已重置你与我的对话上下文。")
        self.assertEqual(len(conv_store.history(SessionKey.group(ACCOUNT, ROOM, NORMAL_SENDER), 1)), 0)


if __name__ == "__main__":
    unittest.main()
