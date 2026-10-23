"""消息模型和插件上下文；插件不需要接触 Hook 或 DLL。"""
from __future__ import annotations

import copy
import logging
import threading
import time
from dataclasses import dataclass, field
from contextlib import nullcontext, contextmanager
from types import MappingProxyType
from typing import Mapping, Any

from .text import validate_target, validate_mentions
from .pacing import TextSendDelay

logger = logging.getLogger("TextSend")


class CapabilityError(NotImplementedError):
    pass


@dataclass(frozen=True)
class Message:
    text: str
    chat_id: str
    sender_id: str = ""
    receiver_id: str = ""
    protocol: int = 11046
    type: int = 1
    is_group: bool = False
    is_self: bool = False
    sequence: int = 0
    session_id: str = ""
    # 统一用 perf_counter：Windows 的 time.monotonic 只有 15.6ms 刻度，
    # 同刻度内的事件无法排序；插件比较时间时必须与这里保持同一时钟。
    received_at: float = field(default_factory=time.perf_counter)
    data: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_native(cls, protocol, data, session_id):
        snapshot = copy.deepcopy(data)
        room = snapshot.get("room_wxid") or ""
        return cls(text=snapshot.get("msg") or snapshot.get("content") or "",
                   chat_id=snapshot.get("chat_id") or room or snapshot.get("from_wxid") or "",
                   sender_id=snapshot.get("sender_id") or snapshot.get("native_sender") or snapshot.get("from_wxid") or "",
                   receiver_id=snapshot.get("native_receiver") or "", protocol=protocol,
                   is_group=bool(room), is_self=bool(snapshot.get("is_self", False)), sequence=snapshot.get("sequence", 0),
                   session_id=session_id, data=MappingProxyType(snapshot))


class ServiceFacade:
    """序列化发送与连接切换，避免卸载正在使用的原生桥接。"""

    def __init__(self, *, text_delay=None):
        self._lock = threading.RLock()
        self._send_lock = threading.Lock()
        self._backend = None
        self._available = False
        self._cancel = threading.Event()
        self.text_delay = TextSendDelay.from_config(text_delay)

    def bind(self, backend):
        with self._lock:
            if self._backend is not None:
                raise RuntimeError("已有连接，需先关闭")
            self._backend = backend
            self._available = True
            self._cancel = threading.Event()

    @property
    def connected(self):
        with self._lock:
            return self._backend is not None and self._available

    def send_text(self, to_wxid, content, *, expected_session=None, _mentions=(), delivery_guard=None):
        validate_target(to_wxid)
        if not isinstance(content, str):
            raise ValueError("文本正文必须是字符串")
        encoded = content.encode("utf-8")
        if not encoded or len(encoded) > 1023 or b"\0" in encoded:
            raise ValueError("文本应为 1–1023 个 UTF-8 字节，且不能包含 NUL")
        with self._lock:
            if self._backend is None or not self._available:
                raise RuntimeError("微信尚未连接")
            if expected_session and expected_session != self._backend.session_id:
                raise RuntimeError("消息所属的微信连接已结束，跳过旧会话回复")
            if _mentions and not callable(getattr(self._backend, "send_at_text", None)):
                raise CapabilityError("当前后端尚未开放真实群艾特")
            backend, cancel = self._backend, self._cancel
        # 等待不占用连接锁，状态查询、断线清理和退出可以及时进行。
        # 发送锁覆盖等待及原生发送，多个插件不会同时等待后集中发送。
        with self._send_lock:
            with self._lock:
                self._check_pending(backend, cancel)
            seconds = self.text_delay.seconds_for(content)
            logger.debug("文本发送等待 %.3f 秒（%d 个字符）", seconds, len(content))
            if cancel.wait(seconds):
                raise RuntimeError("微信连接已结束，取消待发送文本")
            with self._lock:
                self._check_pending(backend, cancel)
                # 在文本节奏等待之后检查记录版本，清空后不发送旧总结。
                with delivery_guard() if delivery_guard is not None else nullcontext():
                    if _mentions:
                        return backend.send_at_text(to_wxid, content, _mentions)
                    return backend.send_text(to_wxid, content)

    def send_at_text(self, room_wxid, content, at_list, *, expected_session=None, display_names=None, delivery_guard=None):
        members = validate_mentions(room_wxid, at_list)
        if not isinstance(content, str) or not content:
            raise ValueError("文本正文必须是非空字符串")
        names = members if display_names is None else display_names
        if not isinstance(names, (list, tuple)) or len(names) != len(members) or any(
                not isinstance(name, str) or not name.strip() or len(name) > 64 or
                any(ch in name for ch in ("\n", "\r", "\0")) for name in names):
            raise ValueError("艾特显示昵称必须与成员列表一一对应，且每个昵称最多 64 字符")
        # 标签用于显示；真实通知始终使用 members 中的微信 ID。
        labels = " ".join("@" + name + "\u2005" for name in names)
        return self.send_text(room_wxid, labels + "\n" + content,
                              expected_session=expected_session, _mentions=members, delivery_guard=delivery_guard)

    def _check_pending(self, backend, cancel):
        if cancel.is_set() or not self._available or self._backend is not backend or self._cancel is not cancel:
            raise RuntimeError("微信连接已结束，取消待发送文本")

    def cancel_pending(self):
        with self._lock:
            self._available = False
            self._cancel.set()

    def send_payload(self, payload, *, expected_session=None):
        if payload.get("type") != 11237:
            raise CapabilityError("当前后端只支持文本发送 11237")
        data = payload["data"]
        return self.send_text(data["to_wxid"], data["content"], expected_session=expected_session)

    def close(self):
        with self._lock:
            self.cancel_pending()
            if self._backend is None:
                return None
            # 清理失败时仍持有此实例，下次可以继续清理，不能重新注入。
            result = self._backend.stop()
            self._backend = None
            return result


class LegacyService:
    def __init__(self, service, session_id):
        self._service, self._session_id = service, session_id

    def send_text(self, to_wxid, content):
        return self._service.send_text(to_wxid, content, expected_session=self._session_id)

    def send_payload(self, payload):
        return self._service.send_payload(payload, expected_session=self._session_id)


class PluginContext:
    def __init__(self, name, options, service, capabilities, *, llm_config_path=None, member_store=None):
        self.name = name
        self.config = MappingProxyType(copy.deepcopy(options))
        self.logger = logging.getLogger(f"Plugin.{name}")
        self.capabilities = frozenset(capabilities)
        self._service = service
        self._llm_config_path = llm_config_path
        self._llm = None
        self._members = member_store

    @property
    def members(self):
        if self._members is None:
            raise CapabilityError("尚未配置框架成员目录")
        return self._members

    @property
    def llm(self):
        if self._llm is None:
            if self._llm_config_path is None:
                raise ValueError("尚未配置大模型文件")
            from .llm import LLMClient
            self._llm = LLMClient.from_file(self._llm_config_path)
        return self._llm

    def send_text(self, chat_id, text):
        return self._service.send_text(chat_id, text)

    def personality(self, message):
        from .personality import PersonalityState
        return self._members.personality(message.chat_id) if self._members is not None else PersonalityState(message.chat_id)

    def llm_for(self, message, *, personality=None):
        from .personality import PersonalityLLM
        state = personality if personality is not None else self.personality(message)
        return PersonalityLLM(self.llm, state, current=lambda: self._members is None or
                              self._members.personality_is_current(state))

    def personality_guard(self, state):
        return self._members.personality_delivery_guard(state) if self._members is not None else nullcontext()

    @contextmanager
    def response_guard(self, snapshot, personality):
        with self.members.delivery_guard(snapshot), self.personality_guard(personality):
            yield

    def reply(self, message, text, *, delivery_guard=None):
        return self._service.send_text(message.chat_id, text, expected_session=message.session_id,
                                       delivery_guard=delivery_guard)

    def reply_at(self, message, text, *, display_name=None, delivery_guard=None):
        if not message.is_group:
            raise ValueError("reply_at 仅用于群消息")
        if self._members is not None:
            member = self._members.find_by_wxid(message.sender_id, message.chat_id)
            if member is not None and (display_name is None or member.self_named):
                display_name = member.display_name
            elif display_name is None:
                display_name = self._members.wechat_name(message.sender_id)
        return self._service.send_at_text(message.chat_id, text, [message.sender_id],
                                         expected_session=message.session_id,
                                         display_names=[display_name] if display_name is not None else None,
                                         delivery_guard=delivery_guard)
