"""群聊定时提醒插件：处理群内提醒创建、查看与取消指令。

指令覆盖：
1. 创建：@机器人 <时间>提醒我<事项>、@机器人 提醒我<时间><事项>、/提醒 <时间> <事项>
2. 查看：@机器人 我的提醒、@机器人 查看提醒、@机器人 提醒列表、/我的提醒、/提醒列表
3. 取消单条：@机器人 取消提醒 <编号>、/取消提醒 <编号>
4. 取消全部：@机器人 取消全部提醒、/取消全部提醒、/取消提醒 全部
"""
from __future__ import annotations

import time
from datetime import datetime

from wxbot_framework.members import BEIJING
from wxbot_framework.mentions import strip_at_labels
from wxbot_framework.reminders import ReminderError
from .parser import ReminderParseError, format_error, parse

VIEW_COMMANDS = frozenset({"/我的提醒", "/提醒列表", "我的提醒", "查看提醒", "提醒列表"})
CANCEL_ALL_COMMANDS = frozenset({"/取消全部提醒", "取消全部提醒", "/取消提醒 全部", "取消提醒 全部"})


class Plugin:
    def __init__(self, clock=time.time):
        self.clock = clock

    def on_load(self, ctx):
        self.members = ctx.members
        self.reminders = ctx.reminders
        ctx.logger.info("群聊定时提醒插件已加载")

    def on_message(self, message, ctx):
        if message.type != 1 or not message.is_group:
            return
        if message.chat_id not in self.members.groups:
            return
        if not message.data.get("reminder_intent"):
            return
        if message.data.get("bot_generated"):
            return

        raw_text = message.text.strip()
        stripped = strip_at_labels(raw_text).strip()
        words = stripped.split()

        # 1. 查看提醒列表
        if stripped in VIEW_COMMANDS:
            self._list_reminders(message, ctx)
            return

        # 2. 取消全部提醒
        if stripped in CANCEL_ALL_COMMANDS:
            self._cancel_all_reminders(message, ctx)
            return

        # 3. 取消单条提醒
        if words and words[0] in ("/取消提醒", "取消提醒"):
            if len(words) == 1:
                ctx.reply(message, "请指定要取消的提醒编号，例如：/取消提醒 1")
                return
            arg = words[1]
            if arg == "全部":
                self._cancel_all_reminders(message, ctx)
                return
            try:
                num = int(arg)
            except ValueError:
                ctx.reply(message, "请指定有效的提醒编号，例如：/取消提醒 1")
                return
            self._cancel_single_reminder(message, ctx, num)
            return

        # 4. 创建提醒（或意图命中但无法解析时的用法提示）
        self._create_reminder(message, ctx, raw_text)

    def _list_reminders(self, message, ctx):
        reminders = self.reminders.list_pending(message.chat_id, message.sender_id)
        if not reminders:
            ctx.reply(message, "你在本群暂无待触发的提醒。")
            return
        lines = [f"你在本群的待触发提醒（{len(reminders)}/5）："]
        for r in reminders:
            dt_str = datetime.fromtimestamp(r.due_at, BEIJING).strftime("%Y-%m-%d %H:%M")
            lines.append(f"#{r.number} {dt_str} {r.content}")
        ctx.reply(message, "\n".join(lines))

    def _cancel_all_reminders(self, message, ctx):
        count = self.reminders.cancel_all(message.chat_id, message.sender_id)
        if count > 0:
            ctx.reply(message, f"已取消你在本群的全部待触发提醒（共 {count} 条）。")
        else:
            ctx.reply(message, "你在本群暂无待触发的提醒。")

    def _cancel_single_reminder(self, message, ctx, number: int):
        row = self.reminders.cancel(message.chat_id, message.sender_id, number)
        if row is not None:
            dt_str = datetime.fromtimestamp(row.due_at, BEIJING).strftime("%Y-%m-%d %H:%M")
            ctx.reply(message, f"已取消提醒 [#{row.number}]（{dt_str} {row.content}）。")
        else:
            ctx.reply(message, f"未找到你的待触发提醒 #{number}，请发送 /我的提醒 查看。")

    def _create_reminder(self, message, ctx, raw_text: str):
        at_members = message.data.get("at_members", ())
        bot_account = self.members.account
        has_other_mentions = any(m != bot_account for m in at_members)

        now_ts = self.clock()
        try:
            parsed = parse(raw_text, now_ts, has_other_mentions=has_other_mentions)
        except (ReminderError, ReminderParseError) as err:
            ctx.reply(message, format_error(err))
            return
        except Exception as err:
            ctx.reply(message, format_error(err))
            return

        try:
            reminder = self.reminders.create(
                message.chat_id,
                message.sender_id,
                parsed.due_at,
                parsed.content,
                now=now_ts,
            )
        except ReminderError as err:
            ctx.reply(message, format_error(err, due_at=parsed.due_at))
            return

        dt_str = datetime.fromtimestamp(reminder.due_at, BEIJING).strftime("%Y-%m-%d %H:%M")
        ctx.reply(message, f"已设置提醒 [#{reminder.number}]：{dt_str} 提醒你「{reminder.content}」")

    def on_unload(self, ctx):
        ctx.logger.info("群聊定时提醒插件已卸载")
