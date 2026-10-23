"""通过配置接入 OpenAI 兼容的大模型接口，不记录密钥或聊天正文。"""
from __future__ import annotations

import json
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
        request = Request(self.base_url + "/chat/completions",
                          data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                          headers={"Authorization": "Bearer " + self._api_key,
                                   "Content-Type": "application/json"}, method="POST")
        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw = response.read(1024 * 1024 + 1)
            if len(raw) > 1024 * 1024:
                raise LLMError("大模型响应过大")
            result = json.loads(raw)
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
        except HTTPError as error:
            # 不输出响应正文或请求头，避免服务器错误回显密钥。
            raise LLMError(f"大模型接口 HTTP {error.code}；请核对密钥、余额和模型配置") from None
        except (URLError, TimeoutError, OSError):
            raise LLMError("大模型连接失败或超时，请稍后再试") from None
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            raise LLMError("大模型响应格式不正确，请重试") from None
