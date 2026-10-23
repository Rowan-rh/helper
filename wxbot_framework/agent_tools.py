"""Agent 群内安全工具集（首批 6 个工具，纯本地零外网依赖）。

- 4 个群类只读工具：get_group_context、get_member_recent_messages、list_group_members、get_group_status
  - 在非群会话（私聊、文件传输助手）中调用时，返回 {"ok": false, "error": "非群会话"}。
  - 读取历史记录时过滤 bot_generated、斜杠命令与语气词。
  - 返回结果脱敏：禁止出现 wxid_* 或 *@chatroom，统一使用称呼或稳定匿名编号 P*。
- 2 个受控写工具：memory_save、memory_delete
  - write=True，由 ToolExecutor 统一进行 VIP/主号权限与调用次数限制。
  - 操作保证幂等，返回当前记忆状态（编号与条数）。
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from .agent import ToolContext, ToolError, ToolRegistry, ToolSpec, is_filler
from .members import BEIJING
from .memory import GroupMemoryStore
from .summary import CONTROL

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


# ---------------------------------------------------------------- 工具工厂与注册

def create_agent_tool_specs() -> tuple[ToolSpec, ...]:
    """创建标准群内 6 工具规格。"""
    return (
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


def create_agent_tool_registry() -> ToolRegistry:
    """创建包含标准 6 工具的 ToolRegistry。"""
    return ToolRegistry(create_agent_tool_specs())
