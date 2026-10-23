"""大模型协议和错误处理验证；使用假响应，不消耗实际 API。"""
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from wxbot_framework.llm import LLMClient, LLMError, ToolsUnsupported
from plugins.idiom_chain.judge import IdiomJudge


class JSONClient:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def complete_json(self, system, user):
        self.calls.append((system, user))
        return self.result


class LLMTests(unittest.TestCase):
    def setUp(self):
        self.options = {"api_key": "dummy-test-key", "model": "deepseek-flash"}
        self.client = LLMClient(self.options)

    def response(self, content, **extra):
        return io.BytesIO(json.dumps({"choices": [{"message": {"content": content}, **extra}]}).encode())

    def test_official_json_request_and_utf8(self):
        with patch("wxbot_framework.llm.urlopen", return_value=self.response('{"idiom":"一心一意"}')) as call:
            self.assertEqual(self.client.complete_json("只返回 JSON", "中文"), {"idiom": "一心一意"})
        request = call.call_args.args[0]
        self.assertEqual(request.full_url, "https://api.deepseek.com/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer dummy-test-key")
        payload = json.loads(request.data)
        self.assertEqual(payload["model"], "deepseek-flash")
        self.assertEqual(payload["messages"][1]["content"], "中文")
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertEqual(call.call_args.kwargs["timeout"], 20)

    def test_minimax_official_endpoint(self):
        client = LLMClient({**self.options, "base_url": "https://api.minimax.cn/v1", "model": "MiniMax-M3"})
        with patch("wxbot_framework.llm.urlopen", return_value=self.response('{"reply":"好的"}')) as call:
            self.assertEqual(client.complete_json("只返回 JSON", "测试"), {"reply": "好的"})
        request = call.call_args.args[0]
        self.assertEqual(request.full_url, "https://api.minimax.cn/v1/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer dummy-test-key")
        payload = json.loads(request.data)
        self.assertEqual(payload["model"], "MiniMax-M3")
        self.assertNotIn("response_format", payload)
        self.assertEqual(payload["thinking"], {"type": "disabled"})

    def test_minimax_requires_explicit_model(self):
        with self.assertRaises(ValueError):
            LLMClient({"api_key": "dummy-test-key", "base_url": "https://api.minimax.cn/v1"})

    def test_json_extracted_from_wrapped_model_text(self):
        for content in ('好的，结果如下：{"reply":"在的"} 请查收',
                        '```json\n{"reply":"在的"}\n```',
                        '前文 {“坏”:1} 干扰 {"reply":"在的"}',
                        '他说"{你好}"然后 {"reply":"在的"}'):
            with self.subTest(content=content), patch("wxbot_framework.llm.urlopen",
                                                      return_value=self.response(content)):
                self.assertEqual(self.client.complete_json("JSON", "测试"), {"reply": "在的"})
        with patch("wxbot_framework.llm.urlopen", return_value=self.response("缺 { 未闭合的正文")):
            with self.assertRaises(LLMError):
                self.client.complete_json("JSON", "测试")

    def test_thinking_and_reasoning_effort_options(self):
        options = {**self.options, "base_url": "https://api.minimax.cn/v1", "model": "MiniMax-M3.1-Flash",
                   "disable_thinking": False, "reasoning_effort": "low"}
        with patch("wxbot_framework.llm.urlopen", return_value=self.response('{"ok":1}')) as call:
            self.assertEqual(LLMClient(options).complete_json("JSON", "测试"), {"ok": 1})
        payload = json.loads(call.call_args.args[0].data)
        self.assertNotIn("thinking", payload)
        self.assertEqual(payload["reasoning_effort"], "low")
        for change in ({"disable_thinking": "no"}, {"reasoning_effort": "none"}, {"reasoning_effort": 3}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                LLMClient({**options, **change})

    def test_inline_thinking_stripped_before_parse(self):
        cases = {'<think>先想想 {"干扰":1}</think>\n{"reply":"在的"}': {"reply": "在的"},
                 '<THINK>考虑中…</THINK>{"reply":"在的"}': {"reply": "在的"},
                 '回答前 <think>没写完': None,
                 '<think>只有思考</think>': None}
        for content, expected in cases.items():
            with self.subTest(content=content), patch("wxbot_framework.llm.urlopen",
                                                      return_value=self.response(content)):
                if expected is None:
                    with self.assertRaises(LLMError):
                        self.client.complete_json("JSON", "测试")
                else:
                    self.assertEqual(self.client.complete_json("JSON", "测试"), expected)

    def test_config_from_utf8_bom_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "llm.json"
            path.write_text(json.dumps(self.options), encoding="utf-8-sig")
            self.assertEqual(LLMClient.from_file(path).model, "deepseek-flash")

    def test_invalid_configuration(self):
        for change in ({"api_key": ""}, {"model": 1}, {"base_url": None},
                       {"base_url": "http://example.com"}, {"base_url": "https://user:pass@example.com"},
                       {"timeout_seconds": True}, {"timeout_seconds": float("nan")},
                       {"max_tokens": 1}, {"max_tokens": True}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                LLMClient({**self.options, **change})

    def test_server_and_connection_errors_hide_sensitive_details(self):
        for error in (HTTPError("https://example.com", 401, "dummy-test-key", {}, None),
                      URLError("dummy-test-key"), TimeoutError("dummy-test-key")):
            with self.subTest(error=type(error).__name__), patch("wxbot_framework.llm.urlopen", side_effect=error):
                with self.assertRaises(LLMError) as caught:
                    self.client.complete_json("JSON", "测试")
                self.assertNotIn("dummy-test-key", str(caught.exception))

    def test_malformed_or_truncated_model_results(self):
        for content in ("", "not json", "[]", None):
            with self.subTest(content=content), patch("wxbot_framework.llm.urlopen", return_value=self.response(content)):
                with self.assertRaises(LLMError):
                    self.client.complete_json("JSON", "测试")
        with patch("wxbot_framework.llm.urlopen", return_value=self.response("{}", finish_reason="length")):
            with self.assertRaisesRegex(LLMError, "截断"):
                self.client.complete_json("JSON", "测试")
        with patch("wxbot_framework.llm.urlopen", return_value=io.BytesIO(b'{"choices":[null]}')):
            with self.assertRaises(LLMError):
                self.client.complete_json("JSON", "测试")


class ChatTests(unittest.TestCase):
    """complete_chat 的多消息对话、工具调用与降级验证。"""

    def setUp(self):
        self.options = {"api_key": "dummy-test-key", "model": "deepseek-flash"}
        self.client = LLMClient(self.options)
        self.tools = [{"type": "function",
                       "function": {"name": "get_group_context", "description": "读取群聊上下文",
                                    "parameters": {"type": "object", "properties": {"limit": {"type": "integer"}}}}},
                      {"type": "function",
                       "function": {"name": "list_group_members",
                                    "parameters": {"type": "object", "properties": {}}}}]

    def response(self, content, **extra):
        return io.BytesIO(json.dumps({"choices": [{"message": {"content": content}, **extra}]}).encode())

    def chat_response(self, content=None, tool_calls=None, finish_reason="stop"):
        message = {"role": "assistant", "content": content}
        if tool_calls is not None:
            message["tool_calls"] = tool_calls
        body = {"choices": [{"message": message, "finish_reason": finish_reason}]}
        return io.BytesIO(json.dumps(body).encode())

    def tool_call(self, call_id, name, arguments):
        return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}

    def test_plain_text_chat_request_and_result(self):
        messages = [{"role": "system", "content": "你是助手"}, {"role": "user", "content": "你好"}]
        with patch("wxbot_framework.llm.urlopen", return_value=self.chat_response("在的")) as call:
            result = self.client.complete_chat(messages)
        self.assertEqual(result.content, "在的")
        self.assertEqual(result.tool_calls, [])
        self.assertEqual(result.finish_reason, "stop")
        self.assertEqual(result.message["role"], "assistant")
        self.assertEqual(result["content"], "在的")
        self.assertEqual(result.get("finish_reason"), "stop")
        request = call.call_args.args[0]
        self.assertEqual(request.full_url, "https://api.deepseek.com/chat/completions")
        payload = json.loads(request.data)
        self.assertEqual(payload["messages"], messages)
        self.assertNotIn("tools", payload)
        self.assertNotIn("response_format", payload)
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertEqual(call.call_args.kwargs["timeout"], 20)

    def test_single_tool_call_is_parsed(self):
        call_item = self.tool_call("call_1", "get_group_context", '{"limit": 5}')
        with patch("wxbot_framework.llm.urlopen", return_value=self.chat_response(None, [call_item])) as call:
            result = self.client.complete_chat([{"role": "user", "content": "群里在聊什么"}], tools=self.tools)
        self.assertEqual(result.content, "")
        self.assertEqual(result.tool_calls,
                         [{"id": "call_1", "name": "get_group_context", "arguments": {"limit": 5}}])
        payload = json.loads(call.call_args.args[0].data)
        self.assertEqual(payload["tools"], self.tools)
        self.assertEqual(payload["tool_choice"], "auto")
        self.assertNotIn("response_format", payload)

    def test_multiple_tool_calls_keep_order(self):
        calls = [self.tool_call("call_1", "get_group_context", '{"limit": 5}'),
                 self.tool_call("call_2", "list_group_members", "{}")]
        with patch("wxbot_framework.llm.urlopen", return_value=self.chat_response(None, calls)):
            result = self.client.complete_chat([{"role": "user", "content": "群里在聊什么"}], tools=self.tools)
        self.assertEqual([call["id"] for call in result.tool_calls], ["call_1", "call_2"])
        self.assertEqual([call["name"] for call in result.tool_calls],
                         ["get_group_context", "list_group_members"])
        self.assertEqual(result.tool_calls[1]["arguments"], {})

    def test_invalid_tool_arguments_are_marked_not_raised(self):
        broken = self.tool_call("call_1", "get_group_context", "{不是 JSON")
        not_object = self.tool_call("call_2", "list_group_members", "[1, 2]")
        missing = {"id": "call_3", "type": "function", "function": {"name": "list_group_members"}}
        with patch("wxbot_framework.llm.urlopen", return_value=self.chat_response(None, [broken, not_object, missing])):
            result = self.client.complete_chat([{"role": "user", "content": "hi"}], tools=self.tools)
        self.assertEqual(len(result.tool_calls), 3)
        for item in result.tool_calls[:2]:
            self.assertIn("error", item)
            self.assertEqual(item["arguments"], {})
        self.assertIn("error", result.tool_calls[2])
        self.assertEqual(result.tool_calls[2]["arguments"], {})

    def test_empty_tool_arguments_mean_no_parameters(self):
        with patch("wxbot_framework.llm.urlopen",
                   return_value=self.chat_response(None, [self.tool_call("call_1", "list_group_members", "")])):
            result = self.client.complete_chat([{"role": "user", "content": "hi"}], tools=self.tools)
        self.assertNotIn("error", result.tool_calls[0])
        self.assertEqual(result.tool_calls[0]["arguments"], {})

    def test_http_400_with_tools_raises_tools_unsupported(self):
        error = HTTPError("https://api.deepseek.com/chat/completions", 400, "Bad Request", {}, None)
        with patch("wxbot_framework.llm.urlopen", side_effect=error):
            with self.assertRaises(ToolsUnsupported) as caught:
                self.client.complete_chat([{"role": "user", "content": "hi"}], tools=self.tools)
        self.assertIsInstance(caught.exception, LLMError)
        self.assertNotIn("dummy-test-key", str(caught.exception))

    def test_http_400_without_tools_stays_plain_error(self):
        error = HTTPError("https://api.deepseek.com/chat/completions", 400, "Bad Request", {}, None)
        with patch("wxbot_framework.llm.urlopen", side_effect=error):
            with self.assertRaises(LLMError) as caught:
                self.client.complete_chat([{"role": "user", "content": "hi"}])
        self.assertNotIsInstance(caught.exception, ToolsUnsupported)
        self.assertNotIn("dummy-test-key", str(caught.exception))

    def test_other_http_errors_with_tools_stay_plain_error(self):
        error = HTTPError("https://api.deepseek.com/chat/completions", 500, "Server Error", {}, None)
        with patch("wxbot_framework.llm.urlopen", side_effect=error):
            with self.assertRaises(LLMError) as caught:
                self.client.complete_chat([{"role": "user", "content": "hi"}], tools=self.tools)
        self.assertNotIsInstance(caught.exception, ToolsUnsupported)

    def test_thinking_block_stripped_from_chat_content(self):
        with patch("wxbot_framework.llm.urlopen", return_value=self.chat_response("<think>先想想</think>在的")):
            result = self.client.complete_chat([{"role": "user", "content": "hi"}])
        self.assertEqual(result.content, "在的")
        self.assertEqual(result.message["content"], "<think>先想想</think>在的")

    def test_enable_tools_configuration(self):
        self.assertTrue(LLMClient(self.options).enable_tools)
        client = LLMClient({**self.options, "enable_tools": False})
        with patch("wxbot_framework.llm.urlopen", return_value=self.chat_response("在的")) as call:
            result = client.complete_chat([{"role": "user", "content": "hi"}], tools=self.tools)
        self.assertNotIn("tools", json.loads(call.call_args.args[0].data))
        self.assertEqual(result.tool_calls, [])
        for change in ({"enable_tools": "yes"}, {"enable_tools": 1}, {"enable_tools": None}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                LLMClient({**self.options, **change})

    def test_invalid_chat_inputs_are_rejected(self):
        for messages in ([], "文本", [{"content": "缺少 role"}], [{"role": "unknown", "content": "x"}], ["文本"]):
            with self.subTest(messages=messages), self.assertRaises(ValueError):
                self.client.complete_chat(messages)
        for tools in ([], "工具", [{"type": "function"}], [{"type": "tool", "function": {"name": "x"}}],
                      [{"type": "function", "function": {}}]):
            with self.subTest(tools=tools), self.assertRaises(ValueError):
                self.client.complete_chat([{"role": "user", "content": "hi"}], tools=tools)
        for choice in ("sometimes", 1, None):
            with self.subTest(choice=choice), self.assertRaises(ValueError):
                self.client.complete_chat([{"role": "user", "content": "hi"}], tools=self.tools, tool_choice=choice)

    def test_chat_response_limits_and_empty_results(self):
        with patch("wxbot_framework.llm.urlopen", return_value=io.BytesIO(b"x" * (1024 * 1024 + 1))):
            with self.assertRaisesRegex(LLMError, "过大"):
                self.client.complete_chat([{"role": "user", "content": "hi"}])
        with patch("wxbot_framework.llm.urlopen",
                   return_value=self.chat_response("在的", finish_reason="length")):
            with self.assertRaisesRegex(LLMError, "截断"):
                self.client.complete_chat([{"role": "user", "content": "hi"}])
        for content in ("", None, "<think>只有思考</think>"):
            with self.subTest(content=content), patch("wxbot_framework.llm.urlopen",
                                                      return_value=self.chat_response(content)):
                with self.assertRaises(LLMError):
                    self.client.complete_chat([{"role": "user", "content": "hi"}])

    def test_complete_json_regression(self):
        with patch("wxbot_framework.llm.urlopen", return_value=self.response('{"reply":"在的"}')) as call:
            self.assertEqual(self.client.complete_json("只返回 JSON", "测试"), {"reply": "在的"})
        payload = json.loads(call.call_args.args[0].data)
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertEqual(payload["messages"][0]["role"], "system")
        client = LLMClient({**self.options, "base_url": "https://api.minimax.cn/v1", "model": "MiniMax-M3"})
        with patch("wxbot_framework.llm.urlopen", return_value=self.response('{"reply":"在的"}')) as call:
            self.assertEqual(client.complete_json("只返回 JSON", "测试"), {"reply": "在的"})
        self.assertNotIn("response_format", json.loads(call.call_args.args[0].data))


class JudgeTests(unittest.TestCase):
    def setUp(self):
        self.client = JSONClient({"is_idiom": True, "reason": "常见成语", "next_idiom": "发扬光大"})
        self.judge = IdiomJudge(self.client)

    def test_local_rejections_do_not_call_model(self):
        for previous, word, used in (("一心一意", "不是四字成语", ()),
                                     ("一心一意", "异想天开", ()),
                                     ("一心一意", "意气风发", ("意气风发",))):
            self.assertFalse(self.judge.evaluate(previous, word, used).accepted)
        self.assertEqual(self.client.calls, [])

    def test_success_sends_only_game_words(self):
        result = self.judge.evaluate("一心一意", "意气风发", ("一心一意",))
        self.assertTrue(result.accepted)
        self.assertEqual(result.next_idiom, "发扬光大")
        self.assertEqual(json.loads(self.client.calls[0][1]),
                         {"previous": "一心一意", "candidate": "意气风发", "used": ["一心一意", "意气风发"]})

    def test_model_can_reject_fake_four_character_word(self):
        self.client.result = {"is_idiom": False, "reason": "不是成语", "next_idiom": ""}
        result = self.judge.evaluate("一心一意", "意意意意", ("一心一意",))
        self.assertFalse(result.accepted)

    def test_invalid_model_next_word_never_advances(self):
        for word in ("假的", "一心一意", "发扬光大", "意气风发"):
            self.client.result["next_idiom"] = word
            with self.subTest(word=word), self.assertRaises(LLMError):
                self.judge.evaluate("一心一意", "意气风发", ("一心一意", "发扬光大"))

    def test_model_boolean_is_strict(self):
        self.client.result["is_idiom"] = "true"
        with self.assertRaises(LLMError):
            self.judge.evaluate("一心一意", "意气风发", ())

    def test_opening_shape(self):
        self.client.result = {"idiom": "一心一意"}
        self.assertEqual(self.judge.opening(), "一心一意")
        self.client.result = {"idiom": "假词"}
        with self.assertRaises(LLMError):
            self.judge.opening()
