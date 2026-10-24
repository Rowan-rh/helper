"""群聊单次提醒：独立存储、框架内调度与意图标记。

- 数据存放在 .runtime/reminders.sqlite3（WAL），按 (account, room, creator) 隔离。
- 每人每群最多 5 条、每群最多 30 条待触发提醒；编号取本人本群最小空闲号 1–5。
- 提醒发送成功、取消或作废后立即删除，磁盘不保留历史正文。
- 日志只记录编号与原定时间，不记录提醒正文、群 ID 或成员 ID。
"""
from __future__ import annotations

import logging
import math
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .account import validate_account
from .members import BEIJING
from .memory import validate_runtime_db_path
from .sdk import SendNotAttempted, mention_display_name
from .text import validate_mentions

logger = logging.getLogger("Reminders")

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


ON_TIME_SECONDS = 60
TICK_SECONDS = 1.0
FALLBACK_NAME = "群友"


def _local(timestamp):
    return datetime.fromtimestamp(timestamp, BEIJING)


def _log_time(timestamp):
    return _local(timestamp).strftime("%Y-%m-%d %H:%M")


def reminder_time_text(due_at, now):
    """当天只显示 HH:MM；非当天带 M月D日。"""
    due = _local(due_at)
    if due.date() == _local(now).date():
        return due.strftime("%H:%M")
    return f"{due.month}月{due.day}日 {due:%H:%M}"


def reminder_text(reminder, now):
    """到点固定模板，不经大模型或群人格改写。"""
    when = reminder_time_text(reminder.due_at, now)
    if now - reminder.due_at > ON_TIME_SECONDS:
        return f"⏰ [延迟提醒] {reminder.content}（原定 {when}）"
    return f"⏰ 提醒时间到：{reminder.content}（设定时间 {when}）"


def _safe_name(name):
    if not isinstance(name, str):
        return FALLBACK_NAME
    name = name.strip()
    if not name or len(name) > 64 or any(ch in name for ch in ("\n", "\r", "\0")):
        return FALLBACK_NAME
    return name


class ReminderScheduler:
    """框架内唯一的提醒调度器；插件热重载不会产生第二个发送线程。

    send(room, content, at_list, display_names) 默认由框架绑定到
    ServiceFacade.send_at_text(..., expected_session=None)。
    """

    def __init__(self, store, members, send, *, is_active=lambda: True, ready=lambda: True,
                 allowlist=(), clock=time.time, grace=GRACE_SECONDS, interval=TICK_SECONDS):
        if store.account != members.account:
            raise ValueError("提醒存储与成员目录必须属于同一个机器人账号")
        self.store, self.members, self.send = store, members, send
        self.is_active, self.ready = is_active, ready
        self.allowlist = frozenset(allowlist)
        self.clock, self.grace, self.interval = clock, grace, interval
        self._tick_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._thread = None
        self._stop = threading.Event()
        self._started = False
        self._recovered = False

    def recover(self):
        """启动时作废残留的 sending；数据库尚不存在时不创建文件。"""
        if not self.store.exists:
            return []
        rows = self.store.abandon_sending()
        for row in rows:
            logger.warning("提醒 #%d（原定 %s）上次停止时正在发送，结果未知，作废不补发",
                           row.number, _log_time(row.due_at))
        return rows

    def _room_allowed(self, room):
        return room in self.members.groups and (not self.allowlist or room in self.allowlist)

    def tick(self, now=None):
        """处理一轮到期提醒，返回本轮成功发送的条数。"""
        with self._tick_lock:
            # 插件停用时提醒原样保留，重新启用后再按宽限规则处理。
            if not self.is_active():
                return 0
            now = self.clock() if now is None else now
            for row in self.store.expire_stale(now, self.grace):
                logger.warning("提醒 #%d（原定 %s）已超过 %d 秒宽限期，作废不补发",
                               row.number, _log_time(row.due_at), self.grace)
            if not self.ready():
                return 0
            sent = 0
            for row in self.store.due(now):
                if not self._room_allowed(row.room):
                    if self.store.discard(row.id):
                        logger.warning("提醒 #%d（原定 %s）所在群已不在启用范围，作废不发送",
                                       row.number, _log_time(row.due_at))
                    continue
                # 认领失败说明已被其他调度器处理或刚被取消。
                if self.store.claim(row.id):
                    sent += self._deliver(row, now)
            return sent

    def _deliver(self, row, now):
        name = _safe_name(mention_display_name(self.members, row.creator, row.room))
        try:
            self.send(row.room, reminder_text(row, now), [row.creator], [name])
        except SendNotAttempted:
            if self.clock() - row.due_at <= self.grace and self.store.release(row.id):
                logger.info("提醒 #%d（原定 %s）暂未发出，等待连接恢复后重试", row.number, _log_time(row.due_at))
            else:
                self.store.finish(row.id)
                logger.warning("提醒 #%d（原定 %s）在宽限期内未能发出，作废", row.number, _log_time(row.due_at))
            return 0
        except Exception as error:
            # 原生发送可能已部分完成，宁可少发一次也不重复 @。
            self.store.finish(row.id)
            logger.warning("提醒 #%d（原定 %s）发送失败，作废不重试：%s",
                           row.number, _log_time(row.due_at), type(error).__name__)
            return 0
        self.store.finish(row.id)
        logger.info("提醒 #%d（原定 %s）已发送", row.number, _log_time(row.due_at))
        return 1

    def _spawn(self):
        self._thread = ReminderTicker(self, self._stop)
        self._thread.start()

    def start(self):
        with self._state_lock:
            if self._started and self._thread is not None and self._thread.is_alive():
                return
            if not self._recovered:
                self.recover()
                self._recovered = True
            self._stop = threading.Event()
            self._started = True
            self._spawn()

    def restart_if_dead(self):
        with self._state_lock:
            if not self._started or self._stop.is_set() or (self._thread is not None and self._thread.is_alive()):
                return False
            logger.warning("提醒调度线程已退出，正在重启")
            self._spawn()
            return True

    @property
    def alive(self):
        return self._thread is not None and self._thread.is_alive()

    def stop(self, timeout=3):
        with self._state_lock:
            self._started = False
            self._stop.set()
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)
            if thread.is_alive():
                logger.warning("提醒调度线程仍在等待发送结束；主程序继续退出")


class ReminderTicker(threading.Thread):
    def __init__(self, scheduler, stop_event):
        super().__init__(name="ReminderTicker", daemon=True)
        self.scheduler, self.stop_event = scheduler, stop_event

    def run(self):
        while not self.stop_event.wait(self.scheduler.interval):
            try:
                self.scheduler.tick()
            except Exception:
                logger.exception("提醒调度失败；下一轮继续")
