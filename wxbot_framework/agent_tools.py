"""Agent 群内安全工具集与外部 MCP / WebSearch 工具桥接。

- 4 个群类只读工具：get_group_context、get_member_recent_messages、list_group_members、get_group_status
  - 在非群会话（私聊、文件传输助手）中调用时，返回 {"ok": false, "error": "非群会话"}。
  - 读取历史记录时过滤 bot_generated、斜杠命令与语气词。
  - 返回结果脱敏：禁止出现 wxid_* 或 *@chatroom，统一使用称呼或稳定匿名编号 P*。
- 2 个受控记忆写工具：memory_save、memory_delete
  - write=True, privileged_only=True，由 ToolExecutor 统一进行 VIP/主号权限与调用次数限制。
  - 操作保证幂等，返回当前记忆状态（编号与条数）。
- 3 个可选群聊提醒工具（include_reminders=True 时启用）：reminder_create、reminder_list、reminder_cancel
  - 来源会话 (account, room, sender_id) 强制由 ToolContext 绑定，不开放给模型参数。
  - reminder_create / reminder_cancel 为 write=True, privileged_only=False（允许普通群友操作本人数据，仍受 max_write_calls 限制）。
- 可选外部工具动态桥接（传入 mcp_manager 时按需懒加载）：内置只读 web_search 与隔离后的 mcp__* 工具。
"""
from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timezone
from typing import Any

from .agent import ToolContext, ToolError, ToolRegistry, ToolSpec, current_tool_timeout, is_filler
from .members import BEIJING
from .memory import GroupMemoryStore
from .reminder_parser import (
    INVALID_CONTENT_TIP,
    NO_REPEAT_TIP,
    ONLY_SELF_TIP,
    RE_PERIODIC,
    SUPPLEMENT_TIME_TIP,
    USAGE_TIP,
    ReminderParseError,
    format_error,
)
from .reminders import (
    PER_MEMBER_LIMIT,
    ReminderError,
    ReminderLimit,
    ReminderStore,
    ReminderTimeError,
    validate_content,
)
from .summary import CONTROL

logger = logging.getLogger(__name__)

WEEKDAY_NAMES = ("星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日")
REMINDER_GROUP_ONLY_ERROR = "仅支持在群聊中管理提醒"

_WXID_PATTERN = re.compile(r"wxid_[a-zA-Z0-9_-]+", re.IGNORECASE)
_CHATROOM_PATTERN = re.compile(r"[a-zA-Z0-9_-]+@chatroom", re.IGNORECASE)


def sanitize_text(text: str) -> str:
    """脱敏：替换正文中可能出现的 wxid_* 或 *@chatroom。"""
    if not isinstance(text, str):
        return ""
    text = _WXID_PATTERN.sub("[微信ID]", text)
    text = _CHATROOM_PATTERN.sub("[群聊]", text)
    return text


def _is_command_or_filler(text: str, filler_words=None) -> bool:
    """判断是否为斜杠命令或语气词。"""
    val = text.strip()
    if not val:
        return True
    first_word = val.split(maxsplit=1)[0]
    if first_word in CONTROL or val.startswith("/"):
        return True
    if is_filler(val, filler_words):
        return True
    return False


# ---------------------------------------------------------------- 1. get_group_context

GET_GROUP_CONTEXT_SCHEMA = {
    "type": "object",
    "properties": {
        "limit": {
            "type": "integer",
            "description": "获取群聊最近发言条数，默认 20，范围 1–50",
            "minimum": 1,
            "maximum": 50,
        }
    },
    "additionalProperties": False,
}


def handle_get_group_context(context: ToolContext, arguments: dict) -> dict:
    if not context.is_group:
        return {"ok": False, "error": "非群会话"}
    limit = arguments.get("limit", 20)
    if type(limit) is not int or not 1 <= limit <= 50:
        raise ToolError("limit 应为 1–50 的整数")
    if context.store is None:
        return {"ok": False, "error": "群成员数据不可用"}

    snapshot = context.store.today_group_snapshot(context.room)
    filler_words = context.extras.get("filler_words")

    messages = []
    for msg in snapshot.messages:
        if msg.get("bot_generated"):
            continue
        text = msg.get("text", "").strip()
        if _is_command_or_filler(text, filler_words):
            continue
        messages.append({
            "speaker": msg.get("speaker", "P?"),
            "speaker_name": sanitize_text(msg.get("speaker_name", "")),
            "text": sanitize_text(text),
            "time": msg.get("received_at", ""),
        })

    result_msgs = messages[-limit:]
    return {
        "ok": True,
        "data": result_msgs,
        "count": len(result_msgs),
    }


# ---------------------------------------------------------------- 2. get_member_recent_messages

GET_MEMBER_RECENT_MESSAGES_SCHEMA = {
    "type": "object",
    "properties": {
        "member_name": {
            "type": "string",
            "description": "群成员称呼（登记称呼、微信昵称或 P 编号）",
        },
        "limit": {
            "type": "integer",
            "description": "获取该成员最近发言条数，默认 15，范围 1–50",
            "minimum": 1,
            "maximum": 50,
        },
    },
    "required": ["member_name"],
    "additionalProperties": False,
}


def handle_get_member_recent_messages(context: ToolContext, arguments: dict) -> dict:
    if not context.is_group:
        return {"ok": False, "error": "非群会话"}
    name = arguments.get("member_name")
    if not isinstance(name, str) or not name.strip():
        raise ToolError("缺少参数 member_name 或称呼为空")
    member_name = name.strip()
    limit = arguments.get("limit", 15)
    if type(limit) is not int or not 1 <= limit <= 50:
        raise ToolError("limit 应为 1–50 的整数")
    if context.store is None:
        return {"ok": False, "error": "群成员数据不可用"}

    snapshot = context.store.today_group_snapshot(context.room)
    filler_words = context.extras.get("filler_words")

    # 尝试按绑定称呼匹配目标 wxid
    target_wxid = None
    try:
        bound = context.store.resolve(member_name, context.room)
        if bound and bound.wxid:
            target_wxid = bound.wxid
    except Exception:
        pass

    found_member = target_wxid is not None
    matched = []
    for msg in snapshot.messages:
        is_match = False
        if target_wxid and msg.get("sender_id") == target_wxid:
            is_match = True
        elif msg.get("speaker_name", "").casefold() == member_name.casefold():
            is_match = True
        elif msg.get("speaker", "").casefold() == member_name.casefold():
            is_match = True

        if is_match:
            found_member = True
            if msg.get("bot_generated"):
                continue
            text = msg.get("text", "").strip()
            if _is_command_or_filler(text, filler_words):
                continue
            matched.append({
                "speaker": msg.get("speaker", "P?"),
                "speaker_name": sanitize_text(msg.get("speaker_name", "")),
                "text": sanitize_text(text),
                "time": msg.get("received_at", ""),
            })

    if not found_member:
        return {"ok": False, "error": f"未找到成员“{sanitize_text(member_name)}”"}

    result_msgs = matched[-limit:]
    return {
        "ok": True,
        "member_name": sanitize_text(member_name),
        "data": result_msgs,
        "count": len(result_msgs),
    }


# ---------------------------------------------------------------- 3. list_group_members

LIST_GROUP_MEMBERS_SCHEMA = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}


def handle_list_group_members(context: ToolContext, arguments: dict) -> dict:
    if not context.is_group:
        return {"ok": False, "error": "非群会话"}
    if context.store is None:
        return {"ok": False, "error": "群成员数据不可用"}

    bound_members = []
    for m in context.store.list_members(context.room):
        bound_members.append({
            "name": sanitize_text(m.name),
            "display_name": sanitize_text(m.display_name),
            "bound": bool(m.wxid),
        })

    snapshot = context.store.today_group_snapshot(context.room)
    active_speakers = set()
    total_messages = 0
    for msg in snapshot.messages:
        if not msg.get("bot_generated"):
            active_speakers.add(msg.get("speaker", ""))
            total_messages += 1

    return {
        "ok": True,
        "bound_members": bound_members,
        "active_members_count": len(active_speakers),
        "today_messages_count": total_messages,
    }


# ---------------------------------------------------------------- 4. get_group_status

GET_GROUP_STATUS_SCHEMA = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}


def handle_get_group_status(context: ToolContext, arguments: dict) -> dict:
    if not context.is_group:
        return {"ok": False, "error": "非群会话"}

    now_bj = datetime.now(timezone.utc).astimezone(BEIJING).strftime("%Y-%m-%d %H:%M:%S")
    persona_name = "默认"
    if context.store:
        try:
            persona_name = context.store.personality(context.room).name
        except Exception:
            pass

    idiom_info = context.extras.get("idiom_chain", {})
    memory_store: GroupMemoryStore | None = context.extras.get("memory_store")
    mem_status = {}
    if memory_store:
        memories = memory_store.list_memories(context.account, context.room)
        mem_status = {
            "used": len(memories),
            "max": memory_store.config.max_entries,
            "slots": f"{len(memories)}/{memory_store.config.max_entries}",
        }

    return {
        "ok": True,
        "beijing_time": now_bj,
        "personality": persona_name,
        "idiom_chain": {
            "active": bool(idiom_info.get("active", False)),
            "current_word": sanitize_text(idiom_info.get("current_word", "")),
        },
        "memory": mem_status,
    }


# ---------------------------------------------------------------- 5. memory_save

MEMORY_SAVE_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["add", "replace"],
            "description": "操作类型：add 为新增，replace 为替换指定编号的已有记忆",
        },
        "content": {
            "type": "string",
            "description": "记忆内容，1–200 字单行纯文本",
        },
        "entry_id": {
            "type": "integer",
            "description": "替换更新时的记忆编号，action 为 replace 时必填",
            "minimum": 1,
        },
    },
    "required": ["action", "content"],
    "additionalProperties": False,
}


def handle_memory_save(context: ToolContext, arguments: dict) -> dict:
    action = arguments.get("action")
    content = arguments.get("content")
    entry_id = arguments.get("entry_id")
    memory_store: GroupMemoryStore | None = context.extras.get("memory_store")
    if memory_store is None:
        return {"ok": False, "error": "长期记忆存储未配置"}

    if action == "add":
        return memory_store.add(context.account, context.room, content, created_by=context.sender_id)
    elif action == "replace":
        if entry_id is None:
            return {"ok": False, "error": "replace 操作必须提供 entry_id"}
        return memory_store.replace(context.account, context.room, entry_id, content, created_by=context.sender_id)
    else:
        return {"ok": False, "error": f"不支持的操作 {action}"}


# ---------------------------------------------------------------- 6. memory_delete

MEMORY_DELETE_SCHEMA = {
    "type": "object",
    "properties": {
        "entry_id": {
            "type": "integer",
            "description": "要删除的记忆编号（正整数）",
            "minimum": 1,
        },
    },
    "required": ["entry_id"],
    "additionalProperties": False,
}


def handle_memory_delete(context: ToolContext, arguments: dict) -> dict:
    entry_id = arguments.get("entry_id")
    memory_store: GroupMemoryStore | None = context.extras.get("memory_store")
    if memory_store is None:
        return {"ok": False, "error": "长期记忆存储未配置"}

    return memory_store.delete(context.account, context.room, entry_id)


# ---------------------------------------------------------------- 7–9. 群聊定时提醒工具

def format_beijing_now(now_value: Any = None) -> str:
    """格式化当前北京时间，供群聊提醒工具注入系统提示词。"""
    if now_value is None:
        dt = datetime.now(timezone.utc).astimezone(BEIJING)
    elif isinstance(now_value, (int, float)):
        dt = datetime.fromtimestamp(now_value, BEIJING)
    elif isinstance(now_value, datetime):
        dt = now_value.astimezone(BEIJING) if now_value.tzinfo is not None else now_value.replace(tzinfo=BEIJING)
    else:
        raise TypeError("now_value 必须是 datetime 或数值时间戳")
    return f"当前北京时间：{dt.strftime('%Y-%m-%d %H:%M')}（{WEEKDAY_NAMES[dt.weekday()]}）"


def _current_timestamp(context: ToolContext) -> float:
    now_fn = context.extras.get("now_fn")
    raw = now_fn() if callable(now_fn) else time.time()
    if isinstance(raw, datetime):
        dt = raw.astimezone(BEIJING) if raw.tzinfo is not None else raw.replace(tzinfo=BEIJING)
        return dt.timestamp()
    return float(raw)


_DATE_ONLY_PATTERN = re.compile(
    r"^(?:\d{4}[-/.年]\d{1,2}[-/.月]\d{1,2}[日号]?|\d{1,2}[-/.月]\d{1,2}[日号]?|今天|今日|明天|明日|后天|大后天|"
    r"(?:本|这|下)?(?:周|星期|礼拜)[一二三四五六七日天])$"
)


def _parse_due_at_arg(due_at_str: str) -> float:
    s = due_at_str.strip()
    if not s:
        raise ReminderParseError(USAGE_TIP)
    if _DATE_ONLY_PATTERN.fullmatch(s):
        raise ReminderParseError(SUPPLEMENT_TIME_TIP)
    if ":" not in s:
        raise ReminderParseError(SUPPLEMENT_TIME_TIP)

    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=BEIJING).timestamp()
        except ValueError:
            pass

    iso_s = s[:-1] + "+00:00" if s.endswith(("Z", "z")) else s
    if not re.match(r"^\d{4}-\d{2}-\d{2}[T ]", iso_s):
        raise ReminderParseError(
            "due_at 格式不正确，请提供北京时间 'YYYY-MM-DD HH:MM' 或带时区的完整 ISO 时间字符串。"
        )
    try:
        dt = datetime.fromisoformat(iso_s)
    except ValueError as err:
        raise ReminderParseError(
            "due_at 格式不正确，请提供北京时间 'YYYY-MM-DD HH:MM' 或带时区的完整 ISO 时间字符串。"
        ) from err

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=BEIJING)
    return dt.timestamp()


REMINDER_CREATE_SCHEMA = {
    "type": "object",
    "properties": {
        "content": {
            "type": "string",
            "description": "提醒事项内容，1–100 字单行纯文本；仅支持为提问者本人设置单次提醒，不支持每天/每周等周期提醒",
        },
        "due_at": {
            "type": "string",
            "description": "目标触发时刻（北京时间 'YYYY-MM-DD HH:MM' 或带时区的 ISO 时间字符串，必须含具体时分；与 delay_minutes 二选一）",
        },
        "delay_minutes": {
            "type": "integer",
            "description": "相对当前时间的延迟分钟数（正整数，如一刻钟为 15、半小时为 30；与 due_at 二选一）",
            "minimum": 1,
        },
    },
    "required": ["content"],
    "additionalProperties": False,
}


def handle_reminder_create(context: ToolContext, arguments: dict) -> dict:
    if not context.is_group:
        return {"ok": False, "error": REMINDER_GROUP_ONLY_ERROR}

    reminder_active = context.extras.get("reminder_active")
    if reminder_active is not None:
        active = reminder_active() if callable(reminder_active) else bool(reminder_active)
        if not active:
            return {"ok": False, "error": "提醒插件已停用，暂时无法设置提醒。"}

    reminder_store: ReminderStore | None = context.extras.get("reminder_store")
    if reminder_store is None:
        return {"ok": False, "error": "提醒存储未配置"}

    if context.extras.get("has_other_mentions"):
        return {"ok": False, "error": ONLY_SELF_TIP}

    raw_content = arguments.get("content")
    try:
        content_text = validate_content(raw_content)
    except ReminderError:
        return {"ok": False, "error": INVALID_CONTENT_TIP}

    if RE_PERIODIC.search(content_text):
        return {"ok": False, "error": NO_REPEAT_TIP}
    if re.match(r"^提醒(?:他人|大家|全体|大伙|别人|他|她|他们|她们)", content_text):
        return {"ok": False, "error": ONLY_SELF_TIP}

    due_at_arg = arguments.get("due_at")
    delay_arg = arguments.get("delay_minutes")
    has_due = due_at_arg is not None and (not isinstance(due_at_arg, str) or bool(due_at_arg.strip()))
    has_delay = delay_arg is not None

    if due_at_arg is not None and isinstance(due_at_arg, str) and not due_at_arg.strip() and not has_delay:
        return {"ok": False, "error": SUPPLEMENT_TIME_TIP}
    if has_due and has_delay:
        return {"ok": False, "error": "due_at 与 delay_minutes 只能二选一，请勿同时指定。"}
    if not has_due and not has_delay:
        return {"ok": False, "error": "请指定 due_at（具体提醒时刻）或 delay_minutes（延迟分钟数）其中之一。"}

    now_ts = _current_timestamp(context)
    if has_delay:
        if type(delay_arg) is not int or delay_arg < 1:
            return {"ok": False, "error": format_error(ReminderTimeError("too_soon"))}
        due_ts = now_ts + delay_arg * 60.0
    else:
        if not isinstance(due_at_arg, str):
            return {"ok": False, "error": USAGE_TIP}
        try:
            due_ts = _parse_due_at_arg(due_at_arg)
        except (ReminderParseError, ReminderError) as err:
            return {"ok": False, "error": sanitize_text(format_error(err))}

    try:
        reminder = reminder_store.create(
            context.room,
            context.sender_id,
            due_ts,
            content_text,
            now=now_ts,
        )
    except (ReminderLimit, ReminderTimeError, ReminderError) as err:
        return {"ok": False, "error": sanitize_text(format_error(err, due_at=due_ts))}

    due_at_beijing = datetime.fromtimestamp(reminder.due_at, BEIJING).strftime("%Y-%m-%d %H:%M")
    safe_content = sanitize_text(reminder.content)
    confirmation = f"已设置提醒 [#{reminder.number}]：{due_at_beijing} 提醒你「{safe_content}」"
    return {
        "ok": True,
        "number": reminder.number,
        "due_at_beijing": due_at_beijing,
        "content": safe_content,
        "message": confirmation,
    }


REMINDER_LIST_SCHEMA = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}


def handle_reminder_list(context: ToolContext, arguments: dict) -> dict:
    if not context.is_group:
        return {"ok": False, "error": REMINDER_GROUP_ONLY_ERROR}

    reminder_store: ReminderStore | None = context.extras.get("reminder_store")
    if reminder_store is None:
        return {"ok": False, "error": "提醒存储未配置"}

    rows = reminder_store.list_pending(context.room, context.sender_id)
    items = [
        {
            "number": row.number,
            "due_at_beijing": datetime.fromtimestamp(row.due_at, BEIJING).strftime("%Y-%m-%d %H:%M"),
            "content": sanitize_text(row.content),
        }
        for row in rows
    ]
    return {
        "ok": True,
        "reminders": items,
        "count": len(items),
        "limit": PER_MEMBER_LIMIT,
    }


REMINDER_CANCEL_SCHEMA = {
    "type": "object",
    "properties": {
        "number": {
            "type": "integer",
            "description": "要取消的待触发提醒编号（1–5），与 cancel_all 二选一",
            "minimum": 1,
        },
        "cancel_all": {
            "type": "boolean",
            "description": "设为 true 时取消提问者本人在本群的全部待触发提醒，与 number 二选一",
        },
    },
    "additionalProperties": False,
}


def handle_reminder_cancel(context: ToolContext, arguments: dict) -> dict:
    if not context.is_group:
        return {"ok": False, "error": REMINDER_GROUP_ONLY_ERROR}

    reminder_store: ReminderStore | None = context.extras.get("reminder_store")
    if reminder_store is None:
        return {"ok": False, "error": "提醒存储未配置"}

    number = arguments.get("number")
    cancel_all = arguments.get("cancel_all", False)
    has_number = number is not None

    if has_number and cancel_all is True:
        return {"ok": False, "error": "number 与 cancel_all 只能二选一，请勿同时指定。"}
    if not has_number and cancel_all is not True:
        return {"ok": False, "error": "请指定要取消的提醒编号 number，或设置 cancel_all=true 取消全部提醒。"}

    if cancel_all is True:
        count = reminder_store.cancel_all(context.room, context.sender_id)
        if count > 0:
            msg = f"已取消你在本群的全部待触发提醒（共 {count} 条）。"
        else:
            msg = "你在本群暂无待触发的提醒。"
        return {
            "ok": True,
            "cancelled_count": count,
            "message": msg,
        }

    if type(number) is not int or number < 1:
        return {"ok": False, "error": "请指定有效的提醒编号，例如：1"}
    row = reminder_store.cancel(context.room, context.sender_id, number)
    if row is None:
        return {
            "ok": False,
            "error": f"未找到你的待触发提醒 #{number}，请发送 /我的提醒 查看。",
        }
    dt_str = datetime.fromtimestamp(row.due_at, BEIJING).strftime("%Y-%m-%d %H:%M")
    safe_content = sanitize_text(row.content)
    return {
        "ok": True,
        "number": row.number,
        "due_at_beijing": dt_str,
        "content": safe_content,
        "message": f"已取消提醒 [#{row.number}]（{dt_str} {safe_content}）。",
    }


# ---------------------------------------------------------------- 工具工厂与注册

def create_reminder_tool_specs() -> tuple[ToolSpec, ...]:
    """创建群聊提醒 3 工具规格（来源会话与创建人强制由 ToolContext 绑定）。"""
    return (
        ToolSpec(
            name="reminder_create",
            description=(
                "为当前提问者本人在本群创建单次定时提醒，到点后机器人会在群内自动 @ 本人。"
                "需要提供事项 content，并二选一提供 delay_minutes（延迟正整数分钟）或 due_at（北京时间 'YYYY-MM-DD HH:MM'）。"
                "不支持周期重复提醒，不支持替他人设置；回复时请保留返回的编号 [#N] 与提醒时间。"
            ),
            parameters_schema=REMINDER_CREATE_SCHEMA,
            handler=handle_reminder_create,
            write=True,
            privileged_only=False,
        ),
        ToolSpec(
            name="reminder_list",
            description="查看当前提问者本人在本群尚未触发的定时提醒列表（编号、北京时间、事项与配额占用）。",
            parameters_schema=REMINDER_LIST_SCHEMA,
            handler=handle_reminder_list,
            write=False,
        ),
        ToolSpec(
            name="reminder_cancel",
            description="取消当前提问者本人在本群的待触发提醒（指定 number 取消单条，或指定 cancel_all=true 取消全部）。",
            parameters_schema=REMINDER_CANCEL_SCHEMA,
            handler=handle_reminder_cancel,
            write=True,
            privileged_only=False,
        ),
    )


def create_agent_tool_specs(*, include_reminders: bool = False) -> tuple[ToolSpec, ...]:
    """创建群内工具规格；默认仅含基础 6 工具，include_reminders=True 时附加 3 个提醒工具。"""
    base = (
        ToolSpec(
            name="get_group_context",
            description="获取当前微信群今日最近发言记录（已过滤机器人自发文本、命令与语气词）。",
            parameters_schema=GET_GROUP_CONTEXT_SCHEMA,
            handler=handle_get_group_context,
            write=False,
        ),
        ToolSpec(
            name="get_member_recent_messages",
            description="按群内称呼、昵称或 P 编号获取某群成员今日发言记录。",
            parameters_schema=GET_MEMBER_RECENT_MESSAGES_SCHEMA,
            handler=handle_get_member_recent_messages,
            write=False,
        ),
        ToolSpec(
            name="list_group_members",
            description="列出当前群已登记称呼的成员名单及今日群内活跃人数概况。",
            parameters_schema=LIST_GROUP_MEMBERS_SCHEMA,
            handler=handle_list_group_members,
            write=False,
        ),
        ToolSpec(
            name="get_group_status",
            description="获取当前群运行状态：当前北京时间、当前人格、成语接龙状态与长期记忆槽位占用。",
            parameters_schema=GET_GROUP_STATUS_SCHEMA,
            handler=handle_get_group_status,
            write=False,
        ),
        ToolSpec(
            name="memory_save",
            description="新增或替换当前会话的长期记忆。仅当提问者为 VIP 或主号时可用；内容去重且满额不淘汰。",
            parameters_schema=MEMORY_SAVE_SCHEMA,
            handler=handle_memory_save,
            write=True,
        ),
        ToolSpec(
            name="memory_delete",
            description="按编号删除当前会话的一条长期记忆。仅当提问者为 VIP 或主号时可用；删除操作幂等。",
            parameters_schema=MEMORY_DELETE_SCHEMA,
            handler=handle_memory_delete,
            write=True,
        ),
    )
    if include_reminders:
        return base + create_reminder_tool_specs()
    return base


WEB_SEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "联网搜索关键词（1–200 字单行或简短查询词）",
            "minLength": 1,
            "maxLength": 200,
        },
        "limit": {
            "type": "integer",
            "description": "返回搜索结果条数上限，可选，范围 1–10",
            "minimum": 1,
            "maximum": 10,
        },
    },
    "required": ["query"],
    "additionalProperties": False,
}


def _make_web_search_handler(mcp_manager: Any, default_timeout: float):
    def _handler(context: ToolContext, arguments: dict) -> dict:
        del context
        query = arguments.get("query")
        if not isinstance(query, str) or not (1 <= len(query.strip()) <= 200):
            raise ToolError("搜索关键词长度须为 1–200 个字符")
        limit = arguments.get("limit")
        if limit is not None and (type(limit) is not int or not (1 <= limit <= 10)):
            raise ToolError("limit 应为 1–10 的整数")
        eff_timeout = current_tool_timeout(default_timeout)
        try:
            return mcp_manager.search_web(query, limit=limit, timeout=eff_timeout)
        except ValueError as err:
            raise ToolError(sanitize_text(str(err)) or "搜索参数无效") from err

    return _handler


def _make_mcp_tool_handler(mcp_manager: Any, tool_name: str, default_timeout: float):
    def _handler(context: ToolContext, arguments: dict) -> dict:
        del context
        eff_timeout = current_tool_timeout(default_timeout)
        return mcp_manager.call_tool(tool_name, arguments, timeout=eff_timeout)

    return _handler


def build_external_tool_specs(mcp_manager: Any) -> tuple[ToolSpec, ...]:
    """按需从 MCPManager 构建内置 web_search 与各懒加载 MCP Server 的 ToolSpec 元组。"""
    if mcp_manager is None:
        return ()

    specs: list[ToolSpec] = []
    if getattr(mcp_manager, "web_search_enabled", False):
        ws_cfg = mcp_manager.web_search_config
        specs.append(
            ToolSpec(
                name="web_search",
                description="通过配置的搜索引擎检索公开网页关键词摘要（只读，结果仅供参考）。",
                parameters_schema=WEB_SEARCH_SCHEMA,
                handler=_make_web_search_handler(mcp_manager, ws_cfg.timeout_seconds),
                write=False,
                privileged_only=False,
                access="public",
                allow_in_group=True,
                timeout=ws_cfg.timeout_seconds,
                result_limit=ws_cfg.max_result_chars,
                external=True,
            )
        )

    for desc in mcp_manager.list_tool_descriptors():
        try:
            specs.append(
                ToolSpec(
                    name=desc.name,
                    description=desc.description,
                    parameters_schema=desc.parameters_schema,
                    handler=_make_mcp_tool_handler(mcp_manager, desc.name, desc.timeout_seconds),
                    write=desc.write,
                    privileged_only=desc.privileged_only,
                    access=desc.access,
                    allow_in_group=desc.allow_in_group,
                    timeout=desc.timeout_seconds,
                    result_limit=desc.max_result_chars,
                    external=True,
                )
            )
        except ValueError as err:
            logger.warning(
                "跳过无效的外部 MCP 工具规格 %s：%s",
                sanitize_text(str(getattr(desc, "name", ""))),
                sanitize_text(str(err)),
            )

    return tuple(specs)


def create_agent_tool_registry(
    *,
    include_reminders: bool = False,
    mcp_manager: Any = None,
) -> ToolRegistry:
    """创建包含群内工具的 ToolRegistry；传入 mcp_manager 时挂载按需懒加载的外部工具源。"""
    provider = (lambda: build_external_tool_specs(mcp_manager)) if mcp_manager is not None else None
    return ToolRegistry(
        create_agent_tool_specs(include_reminders=include_reminders),
        dynamic_provider=provider,
    )
