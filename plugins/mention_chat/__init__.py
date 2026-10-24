"""Agent 对话入口：群聊艾特多轮对话与工具调用、私聊 VIP 与文件传输助手 /ai。"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import time

from wxbot_framework.agent import (
    Agent,
    AgentConfig,
    ConversationStore,
    SessionKey,
    ToolContext,
    AgentRequest,
)
from wxbot_framework.agent_tools import create_agent_tool_registry, format_beijing_now
from wxbot_framework.llm import LLMError
from wxbot_framework.members import BEIJING, SnapshotExpired
from wxbot_framework.memory import GroupMemoryStore, execute_memory_command
from wxbot_framework.mentions import strip_at_labels
from wxbot_framework.personality import PROMPTS, personality_text
from wxbot_framework.sdk import CapabilityError
from wxbot_framework.summary import split_text


class Plugin:
    def __init__(self, clock=time.time):
        self.clock = clock

    def on_load(self, ctx):
        self.members = ctx.members
        if not self.members.history_enabled:
            raise ValueError("艾特上下文对话需要 history_enabled: true")
        ctx.llm  # 启动时核对配置，不请求接口。

        self.agent_config = AgentConfig.from_options(ctx.config.get("agent"))
        self.conversation_store = ConversationStore.from_config(self.agent_config)

        try:
            self.reminder_store = ctx.reminders
        except CapabilityError:
            self.reminder_store = None

        runtime_root = Path(__file__).resolve().parent.parent.parent
        try:
            # 若测试环境或自定义运行目录将数据库置于非默认路径（如 tempdir/.runtime），回退至该配置文件的父目录作为 runtime_root
            if not self.members.path.is_relative_to(runtime_root / ".runtime"):
                if (self.members.config_file.parent / ".runtime").is_dir() and self.members.path.is_relative_to(self.members.config_file.parent / ".runtime"):
                    runtime_root = self.members.config_file.parent
            self.memory_store = GroupMemoryStore(self.members.path, self.members.account,
                                                 runtime_root=runtime_root)
        except Exception as error:
            ctx.logger.warning("长期记忆存储初始化失败，记忆功能已停用：%s", type(error).__name__)
            self.memory_store = None

        mcp_manager = getattr(ctx, "mcp", None)
        self.tool_registry = create_agent_tool_registry(
            include_reminders=self.reminder_store is not None,
            mcp_manager=mcp_manager,
        )
        self._active_day = None

    def on_unload(self, ctx):
        if self.memory_store is not None:
            self.memory_store.close()
            self.memory_store = None
        self.conversation_store.clear_all()

    def _check_rollover(self):
        """跨天时清理多轮会话释放内存，避免竞态。"""
        today = datetime.now(timezone.utc).astimezone(BEIJING).date().isoformat()
        if self._active_day != today:
            if self._active_day is not None:
                self.conversation_store.clear_all()
            self._active_day = today
        return today

    def on_message(self, message, ctx):
        if message.type != 1 or not isinstance(message.text, str):
            return

        today = self._check_rollover()
        is_filehelper = (message.chat_id == "filehelper" or message.sender_id == "filehelper")

        # ------------------------------------------------ 1. 文件传输助手入口
        if is_filehelper:
            if message.data.get("bot_generated"):
                return
            account = self.members.account
            if not (getattr(message, "is_self", False) or message.sender_id == account):
                return
            raw_text = message.text.strip()
            # 记忆与重置命令
            cmd_reply = execute_memory_command(
                raw_text,
                account,
                "filehelper",
                "self",
                is_vip=True,
                is_self=True,
                memory_store=self.memory_store,
                conversation_store=self.conversation_store,
            )
            if cmd_reply is not None:
                ctx.reply(message, cmd_reply)
                return

            prefix = "/ai "
            if not message.text.startswith(prefix):
                return
            question = message.text[len(prefix):].strip()
            if not question:
                return

            cmd_reply = execute_memory_command(
                question,
                account,
                "filehelper",
                "self",
                is_vip=True,
                is_self=True,
                memory_store=self.memory_store,
                conversation_store=self.conversation_store,
            )
            if cmd_reply is not None:
                ctx.reply(message, cmd_reply)
                return

            self._run_filehelper_agent(message, ctx, question, today)
            return

        # ------------------------------------------------ 2. 私聊入口（VIP）
        if not message.is_group:
            if message.data.get("bot_generated"):
                return
            if not self.members.is_vip(message.sender_id):
                return
            account = self.members.account
            raw_text = message.text.strip()
            cmd_reply = execute_memory_command(
                raw_text,
                account,
                message.chat_id,
                message.sender_id,
                is_vip=True,
                is_self=message.sender_id == account,
                memory_store=self.memory_store,
                conversation_store=self.conversation_store,
            )
            if cmd_reply is not None:
                ctx.reply(message, cmd_reply)
                return

            self._run_private_agent(message, ctx, raw_text, today)
            return

        # ------------------------------------------------ 3. 群聊入口
        if message.is_group:
            if message.chat_id not in self.members.groups:
                return
            if message.data.get("reminder_intent"):
                return
            account = self.members.account
            raw_text = message.text.strip()

            # /清空：清理当前群多轮会话，不截断后续 chat_records 插件
            words = raw_text.split()
            if words and words[0] == "/清空":
                if len(words) == 1:
                    self.conversation_store.clear_chat(account, message.chat_id)
                return

            # 记忆与重置命令
            is_vip = self.members.is_vip(message.sender_id)
            is_self = message.sender_id == account
            cmd_reply = execute_memory_command(
                raw_text,
                account,
                message.chat_id,
                message.sender_id,
                is_vip=is_vip,
                is_self=is_self,
                memory_store=self.memory_store,
                conversation_store=self.conversation_store,
            )
            if cmd_reply is not None:
                ctx.reply(message, cmd_reply)
                return

            # 支持 @机器人 携带命令
            stripped = strip_at_labels(raw_text).strip()
            if stripped != raw_text:
                cmd_reply = execute_memory_command(
                    stripped,
                    account,
                    message.chat_id,
                    message.sender_id,
                    is_vip=is_vip,
                    is_self=is_self,
                    memory_store=self.memory_store,
                    conversation_store=self.conversation_store,
                )
                if cmd_reply is not None:
                    ctx.reply(message, cmd_reply)
                    return

            # 艾特机器人对话触发条件
            if ("WXBOT_AT_CHAT_PROBE" in message.text or
                    message.data.get("bot_generated") or not message.data.get("is_at_self") or
                    account not in message.data.get("at_members", ())):
                return

            self._run_group_agent(message, ctx)
            return

    def _run_group_agent(self, message, ctx):
        personality = ctx.personality(message)
        persona_prompt = PROMPTS.get(personality.name) if personality.name != "默认" else None
        delivery_guard = None
        try:
            snapshot = self.members.conversation_snapshot(message)
            generation = (snapshot.day, snapshot.generation)
            key = SessionKey.group(self.members.account, message.chat_id, message.sender_id)
            guard = lambda: bool(self.members.snapshot_is_current(snapshot) and self.members.personality_is_current(personality))
            delivery_guard = lambda: ctx.response_guard(snapshot, personality)

            user_text = strip_at_labels(snapshot.current).strip()
            background = tuple(strip_at_labels(row["text"]) for row in snapshot.messages)
            at_members = message.data.get("at_members", ())
            has_other_mentions = any(m != self.members.account for m in at_members)
            extras = {
                "memory_store": self.memory_store,
                "filler_words": self.agent_config.filler_words,
            }
            now_text = None
            if self.reminder_store is not None:
                extras["reminder_store"] = self.reminder_store
                extras["has_other_mentions"] = has_other_mentions
                extras["now_fn"] = self.clock
                extras["reminder_active"] = lambda: ctx.is_plugin_active("reminder")
                now_text = format_beijing_now(self.clock())

            context = ToolContext(
                account=self.members.account,
                room=message.chat_id,
                sender_id=message.sender_id,
                is_vip=self.members.is_vip(message.sender_id),
                is_self=message.sender_id == self.members.account,
                is_group=True,
                store=self.members,
                extras=extras,
            )
            request = AgentRequest(
                key=key,
                generation=generation,
                text=user_text,
                context=context,
                persona=persona_prompt,
                memory_snapshot=(lambda: self.memory_store.snapshot_text(self.members.account, message.chat_id)) if self.memory_store else None,
                background=background,
                guard=guard,
                now_text=now_text,
            )
            agent = Agent(ctx.llm, store=self.conversation_store, registry=self.tool_registry, config=self.agent_config)
            agent_reply = agent.run(request)
            if any(name.startswith("reminder_") for name in agent_reply.tool_calls):
                reply = agent_reply.text.strip()
                if not reply:
                    raise LLMError("大模型未给出有效回复")
            else:
                reply = personality_text(agent_reply.text, personality.name)

            display = self.members.display_name(message.sender_id, message.chat_id) or snapshot.speaker_name
            budget = 1023 - len(("@" + display + "\u2005\n").encode("utf-8"))
            for part in split_text(reply, byte_limit=budget):
                if not ctx.reply_at(message, part, display_name=display, delivery_guard=delivery_guard):
                    raise RuntimeError("原生艾特发送未成功")
            ctx.logger.info("艾特对话完成：room=%s persona=%s context=%d", message.chat_id,
                            personality.name, len(snapshot.messages))
        except SnapshotExpired:
            ctx.logger.info("艾特对话已取消：记录或人格状态已变更")
        except LLMError as error:
            ctx.logger.warning("艾特对话失败：%s", error)
            try:
                ctx.reply(message, "暂时无法回复，请稍后再艾特我。", delivery_guard=delivery_guard)
            except SnapshotExpired:
                pass

    def _run_private_agent(self, message, ctx, text, today):
        key = SessionKey.private(self.members.account, message.chat_id)
        generation = (today, 0)
        context = ToolContext(
            account=self.members.account,
            room=message.chat_id,
            sender_id=message.sender_id,
            is_vip=True,
            is_self=False,
            is_group=False,
            store=self.members,
            extras={
                "memory_store": self.memory_store,
                "filler_words": self.agent_config.filler_words,
            }
        )
        request = AgentRequest(
            key=key,
            generation=generation,
            text=text,
            context=context,
            persona=None,
            memory_snapshot=(lambda: self.memory_store.snapshot_text(self.members.account, message.chat_id)) if self.memory_store else None,
            background=(),
            guard=None,
        )
        try:
            agent = Agent(ctx.llm, store=self.conversation_store, registry=self.tool_registry, config=self.agent_config)
            agent_reply = agent.run(request)
            reply = agent_reply.text
            for part in split_text(reply, byte_limit=1023):
                ctx.reply(message, part)
            ctx.logger.info("私聊对话完成：rounds=%d", agent_reply.rounds)
        except SnapshotExpired:
            ctx.logger.info("私聊对话已取消")
        except LLMError as error:
            ctx.logger.warning("私聊对话失败：%s", error)
            ctx.reply(message, "暂时无法回复，请稍后再试。")

    def _run_filehelper_agent(self, message, ctx, question, today):
        account = self.members.account
        key = SessionKey.filehelper(account)
        generation = (today, 0)
        context = ToolContext(
            account=account,
            room="filehelper",
            sender_id="self",
            is_vip=True,
            is_self=True,
            is_group=False,
            store=self.members,
            extras={
                "memory_store": self.memory_store,
                "filler_words": self.agent_config.filler_words,
            }
        )
        request = AgentRequest(
            key=key,
            generation=generation,
            text=question,
            context=context,
            persona=None,
            memory_snapshot=(lambda: self.memory_store.snapshot_text(account, "filehelper")) if self.memory_store else None,
            background=(),
            guard=None,
        )
        try:
            agent = Agent(ctx.llm, store=self.conversation_store, registry=self.tool_registry, config=self.agent_config)
            agent_reply = agent.run(request)
            reply = agent_reply.text
            for part in split_text(reply, byte_limit=1023):
                ctx.reply(message, part)
            ctx.logger.info("文件传输助手对话完成：rounds=%d", agent_reply.rounds)
        except SnapshotExpired:
            ctx.logger.info("文件传输助手对话已取消")
        except LLMError as error:
            ctx.logger.warning("文件传输助手对话失败：%s", error)
            ctx.reply(message, "暂时无法回复，请稍后再试。")
