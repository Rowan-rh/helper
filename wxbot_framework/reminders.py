"""群聊单次提醒：独立存储、框架内调度与意图标记。

- 数据存放在 .runtime/reminders.sqlite3（WAL），按 (account, room, creator) 隔离。
- 每人每群最多 5 条、每群最多 30 条待触发提醒；编号取本人本群最小空闲号 1–5。
- 提醒发送成功、取消或作废后立即删除，磁盘不保留历史正文。
- 日志只记录编号与原定时间，不记录提醒正文、群 ID 或成员 ID。
"""
from __future__ import annotations

import math
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .account import validate_account
from .memory import validate_runtime_db_path
from .text import validate_mentions

PER_MEMBER_LIMIT = 5
PER_ROOM_LIMIT = 30
MIN_DELAY_SECONDS = 60
MAX_DELAY_SECONDS = 30 * 24 * 3600
MAX_CONTENT_CHARS = 100
GRACE_SECONDS = 600


class ReminderError(ValueError):
    """提醒参数或状态不合法；插件据此给出固定提示。"""


class ReminderLimit(ReminderError):
    def __init__(self, kind, limit):
        # kind 为 "member"（每人每群）或 "room"（每群）。
        self.kind, self.limit = kind, limit
        super().__init__(f"待触发提醒已达上限：{kind} {limit}")


class ReminderTimeError(ReminderError):
    def __init__(self, kind):
        # kind 为 "past"、"too_soon" 或 "too_far"。
        self.kind = kind
        super().__init__(f"提醒时间不合法：{kind}")


@dataclass(frozen=True)
class Reminder:
    id: int
    account: str
    room: str
    creator: str
    number: int
    due_at: float
    content: str
    created_at: float
    state: str


def validate_content(content):
    if not isinstance(content, str):
        raise ReminderError("提醒事项必须是字符串")
    text = content.strip()
    if not 1 <= len(text) <= MAX_CONTENT_CHARS or any(ord(ch) < 32 or ord(ch) == 127 for ch in text):
        raise ReminderError(f"提醒事项应为 1–{MAX_CONTENT_CHARS} 个字符的单行文本")
    return text


def _timestamp(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ReminderError(f"{name} 应为有限的 UTC 时间戳")
    return float(value)


class ReminderStore:
    """提醒数据库；单连接加锁，供插件线程与调度线程共用。"""

    def __init__(self, db_path, account, runtime_root):
        self.account = validate_account(account)
        self.runtime_root = Path(runtime_root).resolve()
        self.path = validate_runtime_db_path(db_path, self.runtime_root)
        self._lock = threading.RLock()
        self._db = None
        self._closed = False

    @classmethod
    def default(cls, account, runtime_root):
        return cls(Path(runtime_root) / ".runtime" / "reminders.sqlite3", account, runtime_root)

    @property
    def exists(self):
        return self._db is not None or self.path.exists()

    def _connection(self):
        if self._closed:
            raise RuntimeError("提醒存储已关闭")
        if self._db is not None:
            return self._db
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # 自动提交模式，事务由 _transaction 显式控制。
        db = sqlite3.connect(self.path, timeout=5, check_same_thread=False, isolation_level=None)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS reminders (
                    id INTEGER PRIMARY KEY,
                    account TEXT NOT NULL, room TEXT NOT NULL, creator TEXT NOT NULL,
                    number INTEGER NOT NULL, due_at REAL NOT NULL, content TEXT NOT NULL,
                    created_at REAL NOT NULL, state TEXT NOT NULL DEFAULT 'pending',
                    UNIQUE(account, room, creator, number));
                CREATE INDEX IF NOT EXISTS reminders_due ON reminders(account, state, due_at);
            """)
            self._db = db
            return db
        except BaseException:
            db.close()
            raise

    def _transaction(self, callback):
        with self._lock:
            db = self._connection()
            db.execute("BEGIN IMMEDIATE")
            try:
                result = callback(db)
            except BaseException:
                db.execute("ROLLBACK")
                raise
            db.execute("COMMIT")
            return result

    def _row(self, row):
        return Reminder(row["id"], row["account"], row["room"], row["creator"], row["number"],
                        row["due_at"], row["content"], row["created_at"], row["state"])

    def create(self, room, creator, due_at, content, *, now=None):
        validate_mentions(room, [creator])
        due_at = _timestamp(due_at, "due_at")
        now = _timestamp(time.time() if now is None else now, "now")
        text = validate_content(content)
        if due_at <= now:
            raise ReminderTimeError("past")
        if due_at - now < MIN_DELAY_SECONDS:
            raise ReminderTimeError("too_soon")
        if due_at - now > MAX_DELAY_SECONDS:
            raise ReminderTimeError("too_far")

        def insert(db):
            used = {row[0] for row in db.execute(
                "SELECT number FROM reminders WHERE account=? AND room=? AND creator=?",
                (self.account, room, creator))}
            if len(used) >= PER_MEMBER_LIMIT:
                raise ReminderLimit("member", PER_MEMBER_LIMIT)
            total = db.execute("SELECT COUNT(*) FROM reminders WHERE account=? AND room=?",
                               (self.account, room)).fetchone()[0]
            if total >= PER_ROOM_LIMIT:
                raise ReminderLimit("room", PER_ROOM_LIMIT)
            number = min(set(range(1, PER_MEMBER_LIMIT + 1)) - used)
            cursor = db.execute("""INSERT INTO reminders(account,room,creator,number,due_at,content,created_at,state)
                VALUES(?,?,?,?,?,?,?,'pending')""", (self.account, room, creator, number, due_at, text, now))
            return Reminder(cursor.lastrowid, self.account, room, creator, number, due_at, text, now, "pending")

        return self._transaction(insert)

    def list_pending(self, room, creator):
        """本人本群尚未作废的提醒（含发送中），按时间升序。"""
        with self._lock:
            rows = self._connection().execute(
                "SELECT * FROM reminders WHERE account=? AND room=? AND creator=? ORDER BY due_at, number",
                (self.account, room, creator)).fetchall()
            return [self._row(row) for row in rows]

    def cancel(self, room, creator, number):
        """只取消本人本群待触发的提醒；发送中的不能撤回，返回 None。"""
        if type(number) is not int:
            return None

        def remove(db):
            row = db.execute("""SELECT * FROM reminders WHERE account=? AND room=? AND creator=? AND number=?
                AND state='pending'""", (self.account, room, creator, number)).fetchone()
            if row is None:
                return None
            db.execute("DELETE FROM reminders WHERE id=?", (row["id"],))
            return self._row(row)

        return self._transaction(remove)

    def cancel_all(self, room, creator):
        with self._lock:
            return self._connection().execute(
                "DELETE FROM reminders WHERE account=? AND room=? AND creator=? AND state='pending'",
                (self.account, room, creator)).rowcount

    def due(self, now, *, limit=50):
        now = _timestamp(now, "now")
        with self._lock:
            rows = self._connection().execute(
                """SELECT * FROM reminders WHERE account=? AND state='pending' AND due_at<=?
                ORDER BY due_at, id LIMIT ?""", (self.account, now, limit)).fetchall()
            return [self._row(row) for row in rows]

    def claim(self, reminder_id):
        """原子认领 pending→sending；多个调度器同时认领时只有一个成功。"""
        with self._lock:
            return self._connection().execute(
                "UPDATE reminders SET state='sending' WHERE id=? AND account=? AND state='pending'",
                (reminder_id, self.account)).rowcount == 1

    def release(self, reminder_id):
        """尚未调用原生发送就失败时，退回 pending 等待下一轮。"""
        with self._lock:
            return self._connection().execute(
                "UPDATE reminders SET state='pending' WHERE id=? AND account=? AND state='sending'",
                (reminder_id, self.account)).rowcount == 1

    def finish(self, reminder_id):
        """终态（已发送、作废）立即删除，不保留正文。"""
        with self._lock:
            return self._connection().execute(
                "DELETE FROM reminders WHERE id=? AND account=?", (reminder_id, self.account)).rowcount == 1

    def discard(self, reminder_id):
        """只作废仍在 pending 的提醒，不影响其他调度器已认领的记录。"""
        with self._lock:
            return self._connection().execute(
                "DELETE FROM reminders WHERE id=? AND account=? AND state='pending'",
                (reminder_id, self.account)).rowcount == 1

    def expire_stale(self, now, grace=GRACE_SECONDS):
        """删除已超过宽限期仍未发送的 pending 提醒，返回被作废的记录。"""
        cutoff = _timestamp(now, "now") - grace

        def expire(db):
            rows = db.execute("SELECT * FROM reminders WHERE account=? AND state='pending' AND due_at<?",
                              (self.account, cutoff)).fetchall()
            db.execute("DELETE FROM reminders WHERE account=? AND state='pending' AND due_at<?",
                       (self.account, cutoff))
            return [self._row(row) for row in rows]

        return self._transaction(expire)

    def abandon_sending(self):
        """启动时残留的 sending 可能已发出，一律作废，宁可少发也不重复 @。"""
        def abandon(db):
            rows = db.execute("SELECT * FROM reminders WHERE account=? AND state='sending'",
                              (self.account,)).fetchall()
            db.execute("DELETE FROM reminders WHERE account=? AND state='sending'", (self.account,))
            return [self._row(row) for row in rows]

        return self._transaction(abandon)

    def close(self):
        with self._lock:
            self._closed = True
            if self._db is not None:
                self._db.close()
                self._db = None
