"""群聊定时提醒输入解析器（纯函数）。

覆盖：
1. 相对时间：N分钟后、半小时后、N小时后、N小时M分钟后、N天后（保留当前时分）及中文数字；
2. 当天时刻：24小时制 HH:MM、HH点MM分、HH点半、HH点；搭配时段词（早上/上午/中午/下午/傍晚/晚上/今晚）；
3. 跨天日期+时刻：明天/明日、后天、大后天 + 时刻，以及 M月D日 HH:MM、YYYY-MM-DD HH:MM；
4. 拒绝校验：周期词、仅日期无具体时刻、时间已过（不顺延）、<1分钟、>30天、事项1–100字单行、替他人/@他人拒绝。
"""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from typing import Any, NamedTuple, Optional

from wxbot_framework.members import BEIJING
from wxbot_framework.mentions import strip_at_labels
from wxbot_framework.reminders import (
    MAX_CONTENT_CHARS,
    MAX_DELAY_SECONDS,
    MIN_DELAY_SECONDS,
    ReminderError,
    ReminderLimit,
    ReminderTimeError,
)

ZH_DIGITS = {
    "零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}

USAGE_TIP = "未识别到有效的提醒时间或事项。用法示例：@机器人 10分钟后提醒我喝水，或 /提醒 明天09:30 开会"
ONLY_SELF_TIP = "仅支持为自己设置提醒，不支持替其他群成员设置。"
NO_REPEAT_TIP = "暂不支持重复提醒，请指定单次具体时间。"
SUPPLEMENT_TIME_TIP = "请补充具体提醒时刻，例如：@机器人 明天09:00提醒我交报告"
INVALID_CONTENT_TIP = f"设置失败：提醒事项应为 1–{MAX_CONTENT_CHARS} 个字符的单行文本。"


class ReminderParseError(ReminderError):
    """提醒命令解析异常，包含直接返回给用户的回复文案。"""


class ParsedReminder(NamedTuple):
    due_at: float
    content: str


def parse_zh_int(val: str) -> int:
    val = val.strip()
    if val.isdigit():
        return int(val)
    if not val:
        raise ValueError("空数字")
    if val == "零":
        return 0
    if val.startswith("零"):
        return parse_zh_int(val[1:])
    if val == "十":
        return 10
    if val.startswith("十"):
        return 10 + ZH_DIGITS.get(val[1:], 0)
    if "十" in val:
        parts = val.split("十", 1)
        tens = ZH_DIGITS.get(parts[0], 0)
        units = ZH_DIGITS.get(parts[1], 0) if parts[1] else 0
        return tens * 10 + units
    if val in ZH_DIGITS:
        return ZH_DIGITS[val]
    raise ValueError(f"无法解析的中文数字: {val}")


RE_PERIODIC = re.compile(
    r"(?:每天|每日|每周[一二三四五六七日天]?|每个?星期[一二三四五六七日天]?|每个?礼拜[一二三四五六七日天]?|每个?月|每年|每隔|工作日|节假日)"
)

RE_RELATIVE = re.compile(
    r"^(?:"
    r"(?P<days>\d+|[一二两三四五六七八九十]+)\s*天\s*后|"
    r"(?P<hours_half>\d+|[一二两三四五六七八九十]+)\s*个?半\s*小时\s*后|"
    r"(?P<hours_half2>\d+|[一二两三四五六七八九十]+)\s*小时半\s*后|"
    r"半\s*个?小时\s*后|"
    r"(?P<hours_min_h>\d+|[一二两三四五六七八九十]+)\s*个?小时\s*(?:又|零)?\s*(?P<hours_min_m>\d+|[一二两三四五六七八九十]+)\s*分(?:钟)?\s*后|"
    r"(?P<hours>\d+|[一二两三四五六七八九十]+)\s*个?小时\s*后|"
    r"(?P<mins>\d+|[一二两三四五六七八九十]+)\s*分(?:钟)?\s*后|"
    r"(?P<secs>\d+|[一二两三四五六七八九十]+)\s*秒(?:钟)?\s*后"
    r")"
)

RE_DATE = re.compile(
    r"^(?:"
    r"(?P<year>\d{4})[-/.年](?P<month>\d{1,2}|[一二两三四五六七八九十]+)[-/.月](?P<day>\d{1,2}|[一二两三四五六七八九十]+)[日号]?|"
    r"(?P<month2>\d{1,2}|[一二两三四五六七八九十]+)[-/.月](?P<day2>\d{1,2}|[一二两三四五六七八九十]+)[日号]?|"
    r"(?P<named_date>今天|今日|明天|明日|后天|大后天|今晚)"
    r")"
)

RE_PERIOD = re.compile(
    r"^(?P<period>早上|早晨|清晨|上午|中午|下午|傍晚|晚上|今晚|凌晨|半夜|夜里)"
)

RE_TIME = re.compile(
    r"^(?:"
    r"(?P<h_colon>\d{1,2}):(?P<m_colon>\d{2})|"
    r"(?P<h_min_fen>\d{1,2}|[一二两三四五六七八九十]+)\s*点\s*(?P<m_min_fen>\d{1,2}|[零一二两三四五六七八九十]+)\s*分(?:钟)?|"
    r"(?P<h_half>\d{1,2}|[一二两三四五六七八九十]+)\s*点半|"
    r"(?P<h_min_zh>\d{1,2}|[一二两三四五六七八九十]+)\s*点\s*(?P<m_min_zh>(?:[一二两三四五六七八九]?十[一二两三四五六七八九]?|零[一二两三四五六七八九]))|"
    r"(?P<h_min_num>\d{1,2}|[一二两三四五六七八九十]+)\s*点\s*(?P<m_min_num>\d{1,2})(?!\d)|"
    r"(?P<h_invalid_num>\d{1,2}|[一二两三四五六七八九十]+)\s*点\s*(?P<m_invalid_num>\d{3,})|"
    r"(?P<h_whole>\d{1,2}|[一二两三四五六七八九十]+)\s*点(?:整)?|"
    r"(?P<h_shi>\d{1,2})\s*时\s*(?P<m_shi>\d{1,2})\s*分?"
    r")"
)


def _match_relative_time(s: str, now_dt: datetime) -> Optional[tuple[datetime, int]]:
    m = RE_RELATIVE.match(s)
    if not m:
        return None
    end_idx = m.end()
    if m.group("days"):
        days = parse_zh_int(m.group("days"))
        due_dt = now_dt + timedelta(days=days)
    elif m.group("hours_half") or m.group("hours_half2"):
        h = parse_zh_int(m.group("hours_half") or m.group("hours_half2"))
        due_dt = now_dt + timedelta(hours=h, minutes=30)
    elif "半" in m.group(0) and "小时" in m.group(0) and not m.group("hours_half"):
        due_dt = now_dt + timedelta(minutes=30)
    elif m.group("hours_min_h"):
        h = parse_zh_int(m.group("hours_min_h"))
        mins = parse_zh_int(m.group("hours_min_m"))
        due_dt = now_dt + timedelta(hours=h, minutes=mins)
    elif m.group("hours"):
        h = parse_zh_int(m.group("hours"))
        due_dt = now_dt + timedelta(hours=h)
    elif m.group("mins"):
        mins = parse_zh_int(m.group("mins"))
        due_dt = now_dt + timedelta(minutes=mins)
    elif m.group("secs"):
        secs = parse_zh_int(m.group("secs"))
        due_dt = now_dt + timedelta(seconds=secs)
    else:
        return None
    return due_dt, end_idx


def _match_absolute_time(s: str, now_dt: datetime) -> tuple[Optional[datetime], int, bool]:
    """尝试从字符串头部匹配绝对日期与时间。

    返回 (due_dt, end_idx, had_date_without_time)。
    若只匹配到日期但缺少时刻，had_date_without_time 为 True。
    """
    idx = 0
    date_val = None
    period_val = None

    # 1. 尝试匹配日期
    m_date = RE_DATE.match(s[idx:])
    if m_date:
        idx += m_date.end()
        if m_date.group("year"):
            y = int(m_date.group("year"))
            m = parse_zh_int(m_date.group("month"))
            d = parse_zh_int(m_date.group("day"))
            try:
                date_val = date(y, m, d)
            except ValueError:
                return None, 0, False
        elif m_date.group("month2"):
            y = now_dt.year
            m = parse_zh_int(m_date.group("month2"))
            d = parse_zh_int(m_date.group("day2"))
            try:
                date_val = date(y, m, d)
            except ValueError:
                return None, 0, False
        elif m_date.group("named_date"):
            name = m_date.group("named_date")
            if name in ("今天", "今日"):
                date_val = now_dt.date()
            elif name in ("明天", "明日"):
                date_val = now_dt.date() + timedelta(days=1)
            elif name == "后天":
                date_val = now_dt.date() + timedelta(days=2)
            elif name == "大后天":
                date_val = now_dt.date() + timedelta(days=3)
            elif name == "今晚":
                date_val = now_dt.date()
                period_val = "晚上"

    # 跳过日期后的空白
    while idx < len(s) and s[idx].isspace():
        idx += 1

    # 2. 尝试匹配时段词（若未由今晚设定）
    m_period = RE_PERIOD.match(s[idx:])
    if m_period:
        matched_period = m_period.group("period")
        idx += m_period.end()
        if matched_period == "今晚":
            if date_val is None:
                date_val = now_dt.date()
            period_val = "晚上"
        else:
            period_val = matched_period

    # 跳过时段词后的空白
    while idx < len(s) and s[idx].isspace():
        idx += 1

    # 3. 尝试匹配时刻
    m_time = RE_TIME.match(s[idx:])
    if not m_time:
        if date_val is not None:
            return None, idx, True
        return None, 0, False

    idx += m_time.end()
    if m_time.group("h_colon"):
        hour = int(m_time.group("h_colon"))
        minute = int(m_time.group("m_colon"))
    elif m_time.group("h_min_fen"):
        hour = parse_zh_int(m_time.group("h_min_fen"))
        minute = parse_zh_int(m_time.group("m_min_fen"))
    elif m_time.group("h_half"):
        hour = parse_zh_int(m_time.group("h_half"))
        minute = 30
    elif m_time.group("h_min_zh"):
        hour = parse_zh_int(m_time.group("h_min_zh"))
        minute = parse_zh_int(m_time.group("m_min_zh"))
    elif m_time.group("h_min_num"):
        hour = parse_zh_int(m_time.group("h_min_num"))
        minute = int(m_time.group("m_min_num"))
    elif m_time.group("m_invalid_num"):
        return None, 0, False
    elif m_time.group("h_whole"):
        hour = parse_zh_int(m_time.group("h_whole"))
        minute = 0
    elif m_time.group("h_shi"):
        hour = int(m_time.group("h_shi"))
        minute = int(m_time.group("m_shi"))
    else:
        return None, 0, False

    if hour > 23 or minute > 59:
        return None, 0, False

    if date_val is None:
        date_val = now_dt.date()

    # 根据时段词微调小时与跨日（如晚上12点/半夜12点指次日00:00）
    if period_val in ("晚上", "半夜", "夜里"):
        if hour == 12:
            hour = 0
            date_val += timedelta(days=1)
        elif hour < 12:
            if period_val == "晚上":
                hour += 12
            elif period_val in ("半夜", "夜里") and hour == 11:
                hour = 23
    elif period_val in ("下午", "傍晚"):
        if hour < 12:
            hour += 12
    elif period_val == "中午":
        if hour in (1, 2):
            hour += 12
    elif period_val in ("早上", "早晨", "清晨", "上午", "凌晨"):
        if hour == 12:
            hour = 0

    try:
        due_dt = datetime(date_val.year, date_val.month, date_val.day, hour, minute, 0, tzinfo=BEIJING)
    except ValueError:
        return None, 0, False

    return due_dt, idx, False


def parse(text: str, now_beijing: Any = None, has_other_mentions: bool = False) -> ParsedReminder:
    """解析提醒命令中的时间与事项（纯函数）。

    参数：
    - text: 用户原始消息文本
    - now_beijing: 当前基准时间（datetime 或 UTC 时间戳），默认为当前北京时间
    - has_other_mentions: 是否同时艾特了除机器人以外的其他人

    返回：
    - ParsedReminder(due_at=UTC时间戳, content=事项纯文本)

    异常：
    - ReminderParseError: 各种命令格式、权限与不支持语法的用户提示
    - ReminderTimeError: 目标时间已过、太早或太晚
    - ReminderError: 事项字数或内容不合法
    """
    if now_beijing is None:
        now_dt = datetime.now(BEIJING)
    elif isinstance(now_beijing, (int, float)):
        now_dt = datetime.fromtimestamp(now_beijing, BEIJING)
    elif isinstance(now_beijing, datetime):
        now_dt = now_beijing if now_beijing.tzinfo is not None else now_beijing.replace(tzinfo=BEIJING)
        now_dt = now_dt.astimezone(BEIJING)
    else:
        raise TypeError("now_beijing 必须是 datetime 或数值时间戳")

    raw_text = text.strip() if isinstance(text, str) else ""
    if not raw_text:
        raise ReminderParseError(USAGE_TIP)

    # 1. 替他人设置拦截
    if has_other_mentions:
        raise ReminderParseError(ONLY_SELF_TIP)

    # 去掉开头的 /提醒 或 @机器人 标签
    if raw_text.startswith("/提醒"):
        cleaned = raw_text[3:].strip()
    else:
        cleaned = strip_at_labels(raw_text).strip()
        cleaned = re.sub(r"^@\S+\s*", "", cleaned).strip()

    if not cleaned:
        raise ReminderParseError(USAGE_TIP)

    # 检查替他人文本意图
    m_other = re.search(r"提醒(?:他人|大家|全体|大伙|别人|他|她|他们|她们)", cleaned)
    if m_other:
        m_me = re.search(r"提醒(?:我|本人)", cleaned)
        if not m_me or m_other.start() < m_me.start():
            raise ReminderParseError(ONLY_SELF_TIP)

    # 2. 重复/周期拦截
    if RE_PERIODIC.search(cleaned):
        raise ReminderParseError(NO_REPEAT_TIP)

    # 3. 语法结构归一化
    # 结构 A: 提醒我 <时间> <事项>
    # 结构 B: <时间> 提醒我 <事项> / <时间> <事项>
    if cleaned.startswith("提醒我") or cleaned.startswith("提醒本人"):
        s = re.sub(r"^提醒(?:我|本人)\s*", "", cleaned)
        is_prefix_remind = True
    else:
        s = cleaned
        is_prefix_remind = False

    s = s.strip()
    if not s:
        raise ReminderParseError(USAGE_TIP)

    # 4. 尝试解析时间
    try:
        rel_res = _match_relative_time(s, now_dt)
        if rel_res is not None:
            due_dt, end_idx = rel_res
        else:
            abs_res, end_idx, had_date_without_time = _match_absolute_time(s, now_dt)
            if had_date_without_time:
                raise ReminderParseError(SUPPLEMENT_TIME_TIP)
            if abs_res is None:
                raise ReminderParseError(USAGE_TIP)
            due_dt = abs_res
    except ReminderParseError:
        raise
    except (ValueError, OverflowError):
        raise ReminderParseError(USAGE_TIP)

    # 5. 提取事项并清理
    remainder = s[end_idx:].strip()
    # 移除时间后残留的 "提醒我" / "提醒" 词
    if remainder.startswith("提醒我") or remainder.startswith("提醒本人"):
        remainder = re.sub(r"^提醒(?:我|本人)\s*", "", remainder).strip()
    elif remainder.startswith("提醒"):
        remainder = re.sub(r"^提醒\s*", "", remainder).strip()

    remainder = re.sub(r"^[:：，,\s]+", "", remainder).strip()
    if not remainder:
        raise ReminderParseError(USAGE_TIP)

    if len(remainder) > MAX_CONTENT_CHARS or any(ord(ch) < 32 or ord(ch) == 127 for ch in remainder):
        raise ReminderError(INVALID_CONTENT_TIP)

    # 6. 时间有效性校验
    due_ts = due_dt.timestamp()
    now_ts = now_dt.timestamp()
    if due_ts <= now_ts:
        err = ReminderTimeError("past")
        err.due_at = due_ts
        raise err
    if due_ts - now_ts < MIN_DELAY_SECONDS:
        raise ReminderTimeError("too_soon")
    if due_ts - now_ts > MAX_DELAY_SECONDS:
        raise ReminderTimeError("too_far")

    return ParsedReminder(due_ts, remainder)


def format_error(err: Exception, due_at: Optional[float] = None) -> str:
    """将框架或解析器异常格式化为用户友好的回复文案。"""
    if isinstance(err, ReminderLimit):
        if err.kind == "member":
            return f"你在本群的待触发提醒已达上限（{err.limit}/{err.limit}），请先取消部分提醒后再试。"
        if err.kind == "room":
            return f"本群待触发提醒总数已达上限（{err.limit}/{err.limit}），请稍后再试。"
        return f"待触发提醒已达上限：{err.kind}"
    if isinstance(err, ReminderTimeError):
        if err.kind == "past":
            ts = getattr(err, "due_at", due_at)
            if ts is not None:
                dt_str = datetime.fromtimestamp(ts, BEIJING).strftime("%Y-%m-%d %H:%M")
                return f"设置失败：目标时间（{dt_str}）已过去，请重新指定未来的时间（例如：明天 15:00）。"
            return "设置失败：目标时间已过去，请重新指定未来的时间（例如：明天 15:00）。"
        if err.kind == "too_soon":
            return "设置失败：提醒时间至少在 1 分钟之后。"
        if err.kind == "too_far":
            return "设置失败：单次提醒最长支持未来 30 天内。"
    if isinstance(err, ReminderParseError):
        return str(err) if str(err) else USAGE_TIP
    if isinstance(err, ReminderError):
        msg = str(err)
        if "提醒事项" in msg:
            return INVALID_CONTENT_TIP
        return msg if msg else USAGE_TIP
    return USAGE_TIP
