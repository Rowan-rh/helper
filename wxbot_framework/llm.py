"""通过配置接入 OpenAI 兼容的大模型接口，不记录密钥或聊天正文。"""
from __future__ import annotations

import dataclasses
import json
import logging
import math
import re
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


# MiniMax 官方接口域名：国内 api.minimax.cn，海外 api.minimax.io，其余为历史域名。
MINIMAX_DOMAINS = frozenset({"minimax.cn", "minimaxi.com", "minimaxi.chat", "minimax.io", "minimax.chat"})

# 推理模型可能把思考内联为 <think>…</think>；思考内容不参与解析，也不进入回复。
_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.IGNORECASE | re.DOTALL)
_THINK_TAIL = re.compile(r"<think>.*\Z", re.IGNORECASE | re.DOTALL)


class LLMError(RuntimeError):
    pass


class ToolsUnsupported(LLMError):
    """接口不支持工具调用（携带 tools 时服务端返回 HTTP 400）；调用方应据此降级为普通多轮对话。"""


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ChatResult:
    """complete_chat 的结构化结果。

    content 是剥离思考块后的正文；tool_calls 是 [{"id", "name", "arguments"}]，
    参数不是合法 JSON 时该项带 "error" 且 arguments 为空字典，不抛异常；
    finish_reason 是模型给出的结束原因；message 是可原样回传给下一轮的 assistant message。
    """

    content: str = ""
    tool_calls: list = dataclasses.field(default_factory=list)
    finish_reason: str = ""
    message: dict = dataclasses.field(default_factory=dict)

    def __getitem__(self, key):
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key) from None

    def get(self, key, default=None):
        return getattr(self, key, default)


def strip_thinking(text):
    """去掉正文里内联的思考块；MiniMax 的思考在独立 reasoning_content 字段，本来就不会到这里。"""
    return _THINK_TAIL.sub("", _THINK_BLOCK.sub("", text))


def extract_json_object(text):
    """从自由文本中提取第一个完整 JSON 对象；没有则返回 None。

    MiniMax 兼容端点不支持 response_format json_object，模型可能用文字或代码块包住 JSON。"""
    start = text.find("{")
    while start != -1:
        depth, in_string, escaped = 0, False, False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
            elif char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:index + 1])
                    except ValueError:
                        break
        start = text.find("{", start + 1)
    return None


class LLMClient:
    def __init__(self, options):
        if not isinstance(options, dict):
            raise ValueError("大模型配置必须是 JSON 对象")
        base_url = options.get("base_url", "https://api.deepseek.com")
        if not isinstance(base_url, str):
            raise ValueError("大模型 base_url 必须是 HTTPS 地址字符串")
        self.base_url = base_url.rstrip("/")
        parsed = urlsplit(self.base_url)
        # 本机回环地址允许 http（本地路由或代理常见）；其他主机一律要求 HTTPS，避免密钥明文出网。
        loopback = parsed.hostname in ("127.0.0.1", "::1", "localhost")
        if not (parsed.scheme == "https" or (parsed.scheme == "http" and loopback)):
            raise ValueError("大模型 base_url 应为不含认证信息的 HTTPS 地址；本机回环地址可用 http")
        if not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("大模型 base_url 应为不含认证信息的 HTTPS 地址；本机回环地址可用 http")
        host = (parsed.hostname or "").lower()
        self.is_minimax = any(host == domain or host.endswith("." + domain) for domain in MINIMAX_DOMAINS)
        if self.is_minimax and "model" not in options:
            raise ValueError("使用 MiniMax 官方接口时请在配置中明确填写 model，例如 MiniMax-M3")
        self.model = options.get("model", "deepseek-flash")
        self._api_key = options.get("api_key", "")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("大模型 model 不能为空")
        if not isinstance(self._api_key, str) or not self._api_key.strip():
            raise ValueError("请在大模型配置文件中填写 api_key")
        self.timeout = options.get("timeout_seconds", 20)
        self.max_tokens = options.get("max_tokens", 512)
        if type(self.timeout) not in (int, float) or not math.isfinite(self.timeout) or not 1 <= self.timeout <= 60:
            raise ValueError("timeout_seconds 应为 1–60 秒")
        if type(self.max_tokens) is not int or not 128 <= self.max_tokens <= 4096:
            raise ValueError("max_tokens 应为 128–4096 的整数")
        self.disable_thinking = options.get("disable_thinking", True)
        if type(self.disable_thinking) is not bool:
            raise ValueError("disable_thinking 应为布尔值")
        self.reasoning_effort = options.get("reasoning_effort")
        if self.reasoning_effort is not None and self.reasoning_effort not in ("low", "medium", "high", "xhigh", "max"):
            raise ValueError("reasoning_effort 只能是 low/medium/high/xhigh/max")
        self.enable_tools = options.get("enable_tools", True)
        if type(self.enable_tools) is not bool:
            raise ValueError("enable_tools 应为布尔值")

    @classmethod
    def from_file(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8-sig")))

    def complete_json(self, system, user):
        payload = {"model": self.model, "messages": [{"role": "system", "content": system},
                                                       {"role": "user", "content": user}],
                   "max_tokens": self.max_tokens, "stream": False}
        if self.disable_thinking:
            # MiniMax-M3.1-Flash 等自适应思考模型不允许 disabled（HTTP 400），配置 disable_thinking: false 跳过。
            payload["thinking"] = {"type": "disabled"}
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
        if not self.is_minimax:
            # MiniMax 各模型不支持 json_object（会被忽略），改为提示词约束并宽松提取。
            payload["response_format"] = {"type": "json_object"}
        try:
            result = self._post(payload)
            choice = result["choices"][0]
            if choice.get("finish_reason") == "length":
                raise LLMError("大模型响应被截断，请增加 max_tokens")
            content = choice["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise LLMError("大模型返回了空结果")
            content = strip_thinking(content)
            if not content.strip():
                raise LLMError("大模型正文只包含思考内容")
            try:
                data = json.loads(content)
            except ValueError:
                data = extract_json_object(content)
            if not isinstance(data, dict):
                raise LLMError("大模型结果应为 JSON 对象")
            return data
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            raise LLMError("大模型响应格式不正确，请重试") from None

    def complete_chat(self, messages, *, tools=None, tool_choice="auto"):
        """多轮对话，可选工具调用。

        messages 是 OpenAI 兼容的消息列表；tools 是函数定义列表，携带 tools 时不发送 response_format。
        返回 ChatResult；携带 tools 且服务端返回 HTTP 400 时抛出 ToolsUnsupported，调用方据此降级为普通多轮。
        """
        payload = {"model": self.model, "messages": self._check_messages(messages),
                   "max_tokens": self.max_tokens, "stream": False}
        if self.disable_thinking:
            payload["thinking"] = {"type": "disabled"}
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
        sent_tools = bool(tools) and self.enable_tools
        if tools is not None:
            checked_tools = self._check_tools(tools)
            if sent_tools:
                payload["tools"] = checked_tools
                payload["tool_choice"] = self._check_tool_choice(tool_choice)
            else:
                logger.debug("配置已停用工具调用，本次请求按普通多轮对话处理")
        logger.debug("请求大模型：model=%s messages=%d tools=%d", self.model,
                     len(payload["messages"]), len(payload.get("tools", [])))
        try:
            result = self._post(payload, tools_sent=sent_tools)
            choice = result["choices"][0]
            finish_reason = choice.get("finish_reason") or ""
            if finish_reason == "length":
                raise LLMError("大模型响应被截断，请增加 max_tokens")
            message = choice.get("message")
            if not isinstance(message, dict):
                raise LLMError("大模型响应格式不正确，请重试")
            content = message.get("content")
            if content is None:
                content = ""
            elif not isinstance(content, str):
                raise LLMError("大模型响应格式不正确，请重试")
            content = strip_thinking(content)
            tool_calls = self._parse_tool_calls(message.get("tool_calls"))
            if not tool_calls and not content.strip():
                raise LLMError("大模型返回了空结果")
            raw_message = dict(message)
            raw_message.setdefault("role", "assistant")
            logger.debug("大模型返回：finish_reason=%s tool_calls=%d", finish_reason, len(tool_calls))
            return ChatResult(content=content, tool_calls=tool_calls, finish_reason=finish_reason,
                              message=raw_message)
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            raise LLMError("大模型响应格式不正确，请重试") from None

    def _post(self, payload, tools_sent=False):
        """发送一次请求并返回解析后的 JSON；错误信息不含密钥或请求头。"""
        request = Request(self.base_url + "/chat/completions",
                          data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                          headers={"Authorization": "Bearer " + self._api_key,
                                   "Content-Type": "application/json"}, method="POST")
        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw = response.read(1024 * 1024 + 1)
        except HTTPError as error:
            if tools_sent and error.code == 400:
                raise ToolsUnsupported("大模型接口不支持工具调用（HTTP 400）；请降级为普通多轮对话") from None
            # 不输出响应正文或请求头，避免服务器错误回显密钥。
            raise LLMError(f"大模型接口 HTTP {error.code}；请核对密钥、余额和模型配置") from None
        except (URLError, TimeoutError, OSError):
            raise LLMError("大模型连接失败或超时，请稍后再试") from None
        if len(raw) > 1024 * 1024:
            raise LLMError("大模型响应过大")
        try:
            return json.loads(raw)
        except ValueError:
            raise LLMError("大模型响应格式不正确，请重试") from None

    @staticmethod
    def _check_messages(messages):
        if not isinstance(messages, (list, tuple)) or not messages:
            raise ValueError("对话消息必须是非空列表")
        checked = []
        for message in messages:
            if not isinstance(message, dict):
                raise ValueError("对话消息必须是 JSON 对象")
            if message.get("role") not in ("system", "user", "assistant", "tool"):
                raise ValueError("对话消息 role 只能是 system/user/assistant/tool")
            checked.append(dict(message))
        return checked

    @staticmethod
    def _check_tools(tools):
        if not isinstance(tools, (list, tuple)) or not tools:
            raise ValueError("tools 必须是非空列表")
        checked = []
        for tool in tools:
            if not isinstance(tool, dict) or tool.get("type") != "function":
                raise ValueError("tools 只支持 type 为 function 的函数定义")
            function = tool.get("function")
            if not isinstance(function, dict):
                raise ValueError("tools 的 function 必须是 JSON 对象")
            name = function.get("name")
            if not isinstance(name, str) or not name.strip():
                raise ValueError("tools 的函数必须有 name")
            checked.append(dict(tool))
        return checked

    @staticmethod
    def _check_tool_choice(tool_choice):
        if isinstance(tool_choice, str):
            if tool_choice not in ("auto", "none", "required"):
                raise ValueError("tool_choice 只能是 auto/none/required 或函数定义")
            return tool_choice
        if isinstance(tool_choice, dict):
            return tool_choice
        raise ValueError("tool_choice 只能是 auto/none/required 或函数定义")

    @staticmethod
    def _parse_tool_calls(raw_calls):
        """解析工具调用列表；参数不是合法 JSON 时在该项标记 error，不抛异常。"""
        parsed = []
        if not isinstance(raw_calls, list):
            return parsed
        for raw in raw_calls:
            if not isinstance(raw, dict):
                continue
            function = raw.get("function")
            if not isinstance(function, dict):
                function = {}
            name = function.get("name")
            item = {"id": raw.get("id") if isinstance(raw.get("id"), str) else "",
                    "name": name if isinstance(name, str) else "", "arguments": {}}
            arguments = function.get("arguments")
            if isinstance(arguments, dict):
                item["arguments"] = arguments
            elif isinstance(arguments, str):
                try:
                    data = json.loads(arguments) if arguments.strip() else {}
                except ValueError:
                    data = None
                if isinstance(data, dict):
                    item["arguments"] = data
                else:
                    item["error"] = "工具参数不是合法的 JSON 对象"
            else:
                item["error"] = "工具参数缺失"
            parsed.append(item)
        return parsed
