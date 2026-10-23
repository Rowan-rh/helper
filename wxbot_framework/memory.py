"""定长分群长期记忆：存储在 .runtime/member_data.sqlite3 的 group_memories 表中。

- 按 (account, room) 隔离；私聊与文件传输助手按各自 chat_id 作为独立 room。
- 容量默认 50 条 / 单条 200 字 / 总计 6000 字（可配置），内容去重。
- 满额时返回错误提示与 hint，不自动淘汰旧记忆（借鉴 hermes-agent 受控策展）。
- 写操作（save/delete）设计为幂等，并在返回结果中带编号与条数。
- 记忆类命令（/记忆、/删除记忆、/清空记忆）只对 VIP 与主号生效，其他人发送时静默忽略。
- /重置对话 对任何合法触发者开放。
"""
from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .account import validate_account


def validate_runtime_db_path(path: str | Path) -> Path:
    """强校验数据库路径必须位于 .runtime/ 目录下。"""
    if not path or not isinstance(path, (str, Path)) or "\0" in str(path):
        raise ValueError("数据库路径必须为有效字符串或 Path")
    resolved = Path(path).resolve()
    # 路径的非末尾父级目录中必须包含 '.runtime'
    if ".runtime" not in resolved.parts[:-1]:
        raise ValueError(f"数据库路径必须位于 .runtime/ 目录下: {resolved}")
    return resolved


@dataclass(frozen=True)
class MemoryConfig:
    max_entries: int = 50
    max_entry_chars: int = 200
    max_total_chars: int = 6000

    def __post_init__(self):
        if type(self.max_entries) is not int or self.max_entries < 1:
            raise ValueError("max_entries 应为正整数")
        if type(self.max_entry_chars) is not int or self.max_entry_chars < 1:
            raise ValueError("max_entry_chars 应为正整数")
        if type(self.max_total_chars) is not int or self.max_total_chars < 1:
            raise ValueError("max_total_chars 应为正整数")


@dataclass(frozen=True)
class MemoryEntry:
    account: str
    room: str
    entry_id: int
    content: str
    created_by: str = ""
    updated_at: str = ""


class GroupMemoryStore:
    """管理 group_memories 表的长期记忆存储。"""

    def __init__(self, db_path: str | Path, account: str, config: MemoryConfig | None = None):
        self.account = validate_account(account)
        self.path = validate_runtime_db_path(db_path)
        self.config = config if config is not None else MemoryConfig()
        self._lock = threading.RLock()
        self._db = None

    def _connection(self):
        if self._db is not None:
            return self._db
        self.path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.path, timeout=5, check_same_thread=False)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA foreign_keys=ON")
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS group_memories (
                    account TEXT NOT NULL,
                    room TEXT NOT NULL,
                    entry_id INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    created_by TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(account, room, entry_id)
                );
                CREATE INDEX IF NOT EXISTS idx_group_memories_lookup
                    ON group_memories(account, room, entry_id);
            """)
            self._db = db
            return db
        except BaseException:
            db.close()
            raise

    def list_memories(self, account: str, room: str) -> tuple[MemoryEntry, ...]:
        """按编号升序列出指定会话的全部长期记忆。"""
        with self._lock:
            db = self._connection()
            rows = db.execute(
                "SELECT account, room, entry_id, content, created_by, updated_at "
                "FROM group_memories WHERE account=? AND room=? ORDER BY entry_id ASC",
                (account, room)
            ).fetchall()
            return tuple(
                MemoryEntry(
                    account=r["account"],
                    room=r["room"],
                    entry_id=r["entry_id"],
                    content=r["content"],
                    created_by=r["created_by"],
                    updated_at=r["updated_at"],
                )
                for r in rows
            )

    def add(self, account: str, room: str, content: str, created_by: str = "") -> dict:
        """新增记忆，内容去重，满额不淘汰旧条目，幂等返回。"""
        text = content.strip() if isinstance(content, str) else ""
        if not text:
            return {"ok": False, "error": "记忆内容不能为空"}
        if len(text) > self.config.max_entry_chars:
            return {"ok": False, "error": f"单条记忆不能超过 {self.config.max_entry_chars} 个字符"}
        if any(c in text for c in "\r\n\0"):
            return {"ok": False, "error": "记忆内容不能包含换行或控制字符"}

        with self._lock:
            entries = self.list_memories(account, room)
            # 内容去重：如果内容完全相同，幂等返回已有条目
            for e in entries:
                if e.content == text:
                    return {
                        "ok": True,
                        "action": "add",
                        "entry_id": e.entry_id,
                        "content": text,
                        "count": len(entries),
                        "limit": self.config.max_entries,
                        "already_existed": True,
                    }

            if len(entries) >= self.config.max_entries:
                return {
                    "ok": False,
                    "error": f"本群长期记忆已达上限（{len(entries)}/{self.config.max_entries}）",
                    "hint": "请使用 memory_save(action='replace', entry_id=...) 替换过时条目，或先调用 memory_delete",
                    "count": len(entries),
                    "limit": self.config.max_entries,
                }

            current_chars = sum(len(e.content) for e in entries)
            if current_chars + len(text) > self.config.max_total_chars:
                return {
                    "ok": False,
                    "error": f"本群长期记忆总字数已达上限（{current_chars}/{self.config.max_total_chars}）",
                    "hint": "请使用 memory_save(action='replace', entry_id=...) 替换或缩短过时条目，或先调用 memory_delete",
                    "count": len(entries),
                    "limit_chars": self.config.max_total_chars,
                }

            existing_ids = {e.entry_id for e in entries}
            available_slots = set(range(1, self.config.max_entries + 1)) - existing_ids
            entry_id = min(available_slots)

            now_utc = datetime.now(timezone.utc).isoformat()
            db = self._connection()
            with db:
                db.execute(
                    "INSERT INTO group_memories(account, room, entry_id, content, created_by, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (account, room, entry_id, text, created_by, now_utc)
                )

            return {
                "ok": True,
                "action": "add",
                "entry_id": entry_id,
                "content": text,
                "count": len(entries) + 1,
                "limit": self.config.max_entries,
                "already_existed": False,
            }

    def replace(self, account: str, room: str, entry_id: int, content: str, created_by: str = "") -> dict:
        """替换指定编号的记忆，内容去重，总字数受控，幂等返回。"""
        if type(entry_id) is not int or entry_id < 1:
            return {"ok": False, "error": "entry_id 应为正整数"}
        text = content.strip() if isinstance(content, str) else ""
        if not text:
            return {"ok": False, "error": "记忆内容不能为空"}
        if len(text) > self.config.max_entry_chars:
            return {"ok": False, "error": f"单条记忆不能超过 {self.config.max_entry_chars} 个字符"}
        if any(c in text for c in "\r\n\0"):
            return {"ok": False, "error": "记忆内容不能包含换行或控制字符"}

        with self._lock:
            entries = self.list_memories(account, room)
            target = next((e for e in entries if e.entry_id == entry_id), None)
            if target is None:
                return {
                    "ok": False,
                    "error": f"未找到记忆 #{entry_id}，无法替换",
                    "count": len(entries),
                    "limit": self.config.max_entries,
                }

            # 幂等：若目标条目已是该内容
            if target.content == text:
                return {
                    "ok": True,
                    "action": "replace",
                    "entry_id": entry_id,
                    "content": text,
                    "count": len(entries),
                    "limit": self.config.max_entries,
                    "status": "unchanged",
                }

            # 不能与其他已有条目重复
            duplicate = next((e for e in entries if e.entry_id != entry_id and e.content == text), None)
            if duplicate is not None:
                return {
                    "ok": False,
                    "error": f"记忆内容与已有的 #{duplicate.entry_id} 重复",
                    "existing_id": duplicate.entry_id,
                    "count": len(entries),
                }

            new_chars = sum(len(e.content) for e in entries) - len(target.content) + len(text)
            if new_chars > self.config.max_total_chars:
                return {
                    "ok": False,
                    "error": f"替换后总字数将超限（{new_chars}/{self.config.max_total_chars}）",
                    "count": len(entries),
                }

            now_utc = datetime.now(timezone.utc).isoformat()
            db = self._connection()
            with db:
                db.execute(
                    "UPDATE group_memories SET content=?, created_by=?, updated_at=? "
                    "WHERE account=? AND room=? AND entry_id=?",
                    (text, created_by, now_utc, account, room, entry_id)
                )

            return {
                "ok": True,
                "action": "replace",
                "entry_id": entry_id,
                "content": text,
                "count": len(entries),
                "limit": self.config.max_entries,
                "status": "replaced",
            }

    def delete(self, account: str, room: str, entry_id: int) -> dict:
        """删除指定编号的记忆，幂等操作。"""
        if type(entry_id) is not int or entry_id < 1:
            return {"ok": False, "error": "entry_id 应为正整数"}

        with self._lock:
            db = self._connection()
            with db:
                cursor = db.execute(
                    "DELETE FROM group_memories WHERE account=? AND room=? AND entry_id=?",
                    (account, room, entry_id)
                )
                existed = cursor.rowcount > 0
            remaining = self.list_memories(account, room)
            return {
                "ok": True,
                "entry_id": entry_id,
                "count": len(remaining),
                "limit": self.config.max_entries,
                "deleted": existed,
            }

    def clear(self, account: str, room: str) -> int:
        """清空指定会话的全部记忆，返回清除条数。"""
        with self._lock:
            db = self._connection()
            with db:
                cursor = db.execute(
                    "DELETE FROM group_memories WHERE account=? AND room=?",
                    (account, room)
                )
                return cursor.rowcount

    def snapshot_text(self, account: str, room: str) -> str:
        """只读快照文本，格式如：\n#1 每周五晚 20:00 读书分享\n#2 小明海鲜过敏，聚餐避开海鲜"""
        entries = self.list_memories(account, room)
        if not entries:
            return ""
        return "\n".join(f"#{e.entry_id} {e.content}" for e in entries)

    def close(self):
        with self._lock:
            if self._db is not None:
                self._db.close()
                self._db = None


def execute_memory_command(
    text: str,
    account: str,
    room: str,
    sender_id: str,
    *,
    is_vip: bool = False,
    is_self: bool = False,
    memory_store: GroupMemoryStore | None = None,
    conversation_store: Any = None,
) -> str | None:
    """处理记忆与重置相关斜杠命令。

    - 记忆类命令（/记忆、/删除记忆、/清空记忆）只对 VIP 或主号本人开放，其他人发送时静默忽略（返回 None）。
    - /重置对话 对任何合法触发者开放。
    - 非相关命令返回 None。
    """
    if not isinstance(text, str):
        return None
    raw = text.strip()
    if not raw:
        return None

    # 1. /重置对话（对所有人开放）
    if raw == "/重置对话":
        if conversation_store is not None:
            conversation_store.reset(account, room, sender_id)
        return "已重置你与我的对话上下文。"

    # 2. 记忆类命令鉴权：仅 VIP 或主号本人
    privileged = bool(is_vip or is_self)

    # 2.1 /记忆
    if raw == "/记忆":
        if not privileged:
            return None
        if memory_store is None:
            return "尚未配置长期记忆存储。"
        entries = memory_store.list_memories(account, room)
        if not entries:
            return "本群暂无长期记忆。"
        lines = [f"本群长期记忆（{len(entries)}/{memory_store.config.max_entries}）："]
        for e in entries:
            lines.append(f"#{e.entry_id} {e.content}")
        return "\n".join(lines)

    # 2.2 /清空记忆
    if raw == "/清空记忆":
        if not privileged:
            return None
        if memory_store is None:
            return "尚未配置长期记忆存储。"
        count = memory_store.clear(account, room)
        return f"已清空本群长期记忆（共 {count} 条）。"

    # 2.3 /删除记忆 <编号>
    if raw == "/删除记忆" or raw.startswith("/删除记忆 "):
        if not privileged:
            return None
        parts = raw.split(maxsplit=1)
        if len(parts) != 2:
            return "用法：/删除记忆 编号"
        try:
            entry_id = int(parts[1])
            if entry_id <= 0:
                raise ValueError
        except ValueError:
            return "用法：/删除记忆 编号"

        if memory_store is None:
            return "尚未配置长期记忆存储。"
        entries = memory_store.list_memories(account, room)
        if not any(e.entry_id == entry_id for e in entries):
            return f"未找到记忆 #{entry_id}，请发送 /记忆 查看。"
        memory_store.delete(account, room, entry_id)
        return f"已删除记忆 #{entry_id}。"

    return None
