"""Agent 循环、多轮会话存储与工具执行验证；全部使用假大模型，不访问网络。

示例数据均为虚构：账号 wxid_example_robot、群 10001@chatroom、成员 wxid_alice / wxid_bob。
"""
import json
import threading
import time
import unittest

from wxbot_framework.agent import (Agent, AgentConfig, AgentError, AgentRequest, AgentTimeout, ConversationStore,
                                   SessionKey, ToolContext, ToolError, ToolRegistry, ToolSpec, build_system_prompt,
                                   is_filler, truncate_result)
from wxbot_framework.llm import ChatResult, LLMError, ToolsUnsupported
from wxbot_framework.members import SnapshotExpired

ACCOUNT = "wxid_example_robot"
ROOM = "10001@chatroom"
ROOM_B = "10002@chatroom"
ALICE = "wxid_alice"
BOB = "wxid_bob"


def call(name, arguments=None, call_id="", **extra):
    item = {"id": call_id, "name": name, "arguments": arguments or {}}
    item.update(extra)
    return item


def tool_reply(*calls, content=""):
    message = {"role": "assistant", "content": content,
               "tool_calls": [{"id": c["id"], "type": "function",
                               "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
                              for c in calls]}
    return ChatResult(content=content, tool_calls=list(calls), finish_reason="tool_calls", message=message)


def text_reply(text):
    return ChatResult(content=text, finish_reason="stop", message={"role": "assistant", "content": text})


class FakeLLM:
    """按顺序返回预设结果；记录每次请求的消息和工具（深拷贝）。"""

    def __init__(self, *results, enable_tools=True):
        self.results = list(results)
        self.requests = []
        self.enable_tools = enable_tools

    def complete_chat(self, messages, *, tools=None, tool_choice="auto"):
        self.requests.append({"messages": json.loads(json.dumps(messages, ensure_ascii=False)),
                              "tools": json.loads(json.dumps(tools)) if tools is not None else None})
        result = self.results.pop(0)
        if callable(result):
            result = result(messages, tools)
        if isinstance(result, BaseException):
            raise result
        return result


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def context(sender=ALICE, room=ROOM, **extra):
    return ToolContext(account=ACCOUNT, room=room, sender_id=sender, **extra)


def request(text, sender=ALICE, room=ROOM, generation=("2026-10-23", 0), **extra):
    ctx = extra.pop("context", None) or context(sender, room)
    return AgentRequest(key=SessionKey.group(ACCOUNT, room, sender), generation=generation, text=text,
                        context=ctx, **extra)


def object_schema(properties=None, required=()):
    return {"type": "object", "properties": properties or {}, "required": list(required)}


def roles(messages):
    return [message["role"] for message in messages]


class FillerTests(unittest.TestCase):
    def test_filler_detection(self):
        for text in ("啊", "哦～", "嗯嗯", "哈哈哈哈！", "好的。", "OK", " ok ", "收到!!", "", "   ", "👍", "okok"):
            self.assertTrue(is_filler(text), text)
        for text in ("好的，那明天八点见", "哈哈你说得对，但是周五不行", "ok 那我去订餐厅", "多少钱", "嗯？为什么"):
            self.assertFalse(is_filler(text), text)

    def test_custom_word_list(self):
        self.assertFalse(is_filler("妙啊", ["啊"]))
        self.assertTrue(is_filler("妙啊", ["妙啊"]))
        self.assertTrue(is_filler("妙啊妙啊!", ["妙啊"]))
        self.assertFalse(is_filler("哈哈", ["好的"]))
        config = AgentConfig.from_options({"filler_words": ["妙啊"], "unknown": 1})
        self.assertTrue(is_filler("妙啊", config.filler_words))
        with self.assertRaises(ValueError):
            AgentConfig.from_options({"filler_words": "啊"})


class ConversationStoreTests(unittest.TestCase):
    def setUp(self):
        self.store = ConversationStore(max_turns=3, max_chars=100)
        self.key = SessionKey.group(ACCOUNT, ROOM, ALICE)

    def add(self, user, assistant, key=None, generation=0):
        handle = self.store.load(key or self.key, generation)
        return self.store.commit(handle, user, assistant)

    def test_trims_by_whole_turns_count_and_chars(self):
        for index in range(5):
            self.assertTrue(self.add(f"问题{index}", f"回答{index}"))
        turns = self.store.history(self.key, 0)
        self.assertEqual([turn.user for turn in turns], ["问题2", "问题3", "问题4"])
        self.assertTrue(self.add("长" * 60, "答" * 30))
        turns = self.store.history(self.key, 0)
        # 90 字的新轮次从最旧的整轮开始挤出，总字数不超过 100
        self.assertEqual([turn.user for turn in turns], ["问题4", "长" * 60])
        self.assertLessEqual(sum(turn.size for turn in turns), 100)
        self.assertTrue(all(turn.assistant for turn in turns))
        self.assertFalse(self.add("超" * 80, "长" * 30))  # 单轮就超限：整轮丢弃
        self.assertEqual(self.store.history(self.key, 0), ())

    def test_filler_turns_do_not_occupy_window(self):
        self.add("周五聚餐去哪", "去老地方")
        for text in ("哈哈", "好的", "嗯嗯", "ok"):
            self.assertFalse(self.add(text, "好嘞"))
        self.assertEqual([turn.user for turn in self.store.history(self.key, 0)], ["周五聚餐去哪"])

    def test_session_isolation_reset_and_generation(self):
        bob = SessionKey.group(ACCOUNT, ROOM, BOB)
        other_room = SessionKey.group(ACCOUNT, ROOM_B, ALICE)
        other_account = SessionKey.group("wxid_bot_b", ROOM, ALICE)
        private = SessionKey.private(ACCOUNT, ALICE)
        filehelper = SessionKey.filehelper(ACCOUNT)
        self.assertEqual(private, SessionKey(ACCOUNT, ALICE, ALICE))
        self.assertEqual(filehelper, SessionKey(ACCOUNT, "filehelper", "self"))
        for key in (self.key, bob, other_room, other_account, private, filehelper):
            self.add(f"{key.chat_id}/{key.sender_id} 的问题", "回答", key=key)
        for key in (self.key, bob, other_room, other_account, private, filehelper):
            self.assertEqual([turn.user for turn in self.store.history(key, 0)], [f"{key.chat_id}/{key.sender_id} 的问题"])
        self.assertTrue(self.store.reset(ACCOUNT, ROOM, ALICE))
        self.assertEqual(self.store.history(self.key, 0), ())
        self.assertEqual(len(self.store.history(bob, 0)), 1)
        self.assertEqual(self.store.clear_chat(ACCOUNT, ROOM), 1)  # 只剩 bob
        self.assertEqual(self.store.history(bob, 0), ())
        self.assertEqual(len(self.store.history(other_room, 0)), 1)
        self.assertEqual(len(self.store.history(other_account, 0)), 1)
        # 跨天/清空后 generation 变化：旧上下文作废
        self.assertEqual(self.store.history(other_room, 1), ())
        self.assertEqual(self.store.history(other_room, 0), ())

    def test_stale_handle_cannot_write_after_reset_or_new_generation(self):
        handle = self.store.load(self.key, 0)
        self.store.reset(ACCOUNT, ROOM, ALICE)
        self.assertFalse(self.store.commit(handle, "在途问题", "在途回答"))
        handle = self.store.load(self.key, 0)
        self.store.clear_chat(ACCOUNT, ROOM)
        self.assertFalse(self.store.commit(handle, "在途问题", "在途回答"))
        old = self.store.load(self.key, 0)
        new = self.store.load(self.key, 1)
        self.assertTrue(self.store.commit(new, "新一天", "回答"))
        self.assertFalse(self.store.commit(old, "旧一天", "回答"))
        self.assertEqual([turn.user for turn in self.store.history(self.key, 1)], ["新一天"])


class ToolSpecTests(unittest.TestCase):
    def test_rejects_context_parameters_and_bad_specs(self):
        handler = lambda ctx, args: {"ok": True}
        for name in ("room", "account", "chat_id", "sender_id"):
            with self.assertRaises(ValueError):
                ToolSpec("t", "d", object_schema({name: {"type": "string"}}), handler)
        with self.assertRaises(ValueError):
            ToolSpec("bad name", "d", object_schema(), handler)
        with self.assertRaises(ValueError):
            ToolSpec("t", "d", {"type": "array"}, handler)
        with self.assertRaises(ValueError):
            ToolSpec("t", "d", object_schema({}, ["missing"]), handler)
        registry = ToolRegistry([ToolSpec("t", "d", object_schema(), handler)])
        with self.assertRaises(ValueError):
            registry.register(ToolSpec("t", "d", object_schema(), handler))
        self.assertEqual(registry.definitions()[0]["function"]["name"], "t")

    def test_truncate_keeps_valid_json(self):
        text = truncate_result({"ok": True, "data": "字\"" * 3000}, 2000)
        self.assertLessEqual(len(text), 2000)
        data = json.loads(text)
        self.assertTrue(data["truncated"])
        self.assertTrue(data["ok"])


class AgentLoopTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.seen = []
        self.writes = []

        def context_tool(ctx, args):
            self.seen.append((ctx.account, ctx.room, ctx.sender_id, dict(args)))
            return {"ok": True, "data": [{"speaker": "P1", "text": "今天聊聚餐"}]}

        def status_tool(ctx, args):
            return {"ok": True, "data": {"personality": "默认"}}

        def member_tool(ctx, args):
            raise ToolError(f"未找到成员 {args['member_name']}")

        def broken_tool(ctx, args):
            raise RuntimeError("C:\\secret\\path wxid_secret")

        def save_tool(ctx, args):
            self.writes.append(args["content"])
            return {"ok": True, "entry_id": len(self.writes)}

        self.registry = ToolRegistry([
            ToolSpec("get_group_context", "读取群聊", object_schema({"limit": {"type": "integer"}}), context_tool),
            ToolSpec("get_group_status", "群状态", object_schema(), status_tool),
            ToolSpec("get_member_recent_messages", "成员发言",
                     object_schema({"member_name": {"type": "string"}}, ["member_name"]), member_tool),
            ToolSpec("broken", "总是出错", object_schema(), broken_tool),
            ToolSpec("memory_save", "保存记忆", object_schema({"content": {"type": "string"}}, ["content"]),
                     save_tool, write=True),
        ])

    def agent(self, llm, **config):
        return Agent(llm, registry=self.registry, config=AgentConfig(**config), clock=self.clock)

    def test_ac1_multi_turn_follow_up_without_orphan_tool_messages(self):
        llm = FakeLLM(tool_reply(call("get_group_context", {"limit": 5}, "c1")),
                      text_reply("总共 480 元。"),
                      text_reply("8 个人共 640 元。"))
        agent = self.agent(llm)
        first = agent.run(request("帮我算下周五聚餐如果去 6 个人人均 80 总共多少"))
        self.assertEqual(first.text, "总共 480 元。")
        self.assertTrue(first.stored)
        second = agent.run(request("那如果临时多来 2 个人呢？"))
        self.assertEqual(second.text, "8 个人共 640 元。")
        messages = llm.requests[2]["messages"]
        self.assertEqual(roles(messages), ["system", "user", "assistant", "user"])
        self.assertEqual(messages[1]["content"], "帮我算下周五聚餐如果去 6 个人人均 80 总共多少")
        self.assertEqual(messages[2], {"role": "assistant", "content": "总共 480 元。"})
        self.assertFalse(any("tool_calls" in m or m["role"] == "tool" for m in messages))

    def test_ac3_normal_multi_step_tool_calls(self):
        llm = FakeLLM(tool_reply(call("get_group_context", {"limit": 20}, "c1"), call("get_group_status", {}, "c2")),
                      text_reply("大家在聊聚餐，现在是默认人格。"))
        reply = self.agent(llm).run(request("大家今天都在聊什么？顺便看看现在是什么人格"))
        self.assertEqual(reply.text, "大家在聊聚餐，现在是默认人格。")
        self.assertEqual(reply.rounds, 2)
        self.assertEqual(reply.tool_calls, ("get_group_context", "get_group_status"))
        self.assertEqual(self.seen, [(ACCOUNT, ROOM, ALICE, {"limit": 20})])
        second = llm.requests[1]
        self.assertIsNotNone(second["tools"])
        self.assertEqual(roles(second["messages"]), ["system", "user", "assistant", "tool", "tool"])
        self.assertEqual([m["tool_call_id"] for m in second["messages"][3:]], ["c1", "c2"])
        self.assertEqual(json.loads(second["messages"][4]["content"]), {"ok": True, "data": {"personality": "默认"}})
        for definition in llm.requests[0]["tools"]:
            properties = definition["function"]["parameters"]["properties"]
            self.assertFalse({"room", "account", "sender_id"} & set(properties))

    def test_ac3_fourth_request_has_no_tools_and_converges(self):
        llm = FakeLLM(*(tool_reply(call("get_group_status", {}, f"r{i}")) for i in range(3)),
                      text_reply("根据已有信息：现在是默认人格。"))
        reply = self.agent(llm).run(request("现在什么人格"))
        self.assertEqual(reply.rounds, 4)
        self.assertEqual([r["tools"] is not None for r in llm.requests], [True, True, True, False])
        last = llm.requests[3]["messages"]
        self.assertIn("不要再调用工具", last[0]["content"])
        self.assertEqual(roles(last).count("tool"), 3)
        self.assertNotIn("不要再调用工具", llm.requests[0]["messages"][0]["content"])

    def test_forced_round_ignoring_tools_without_text_fails_without_storing(self):
        llm = FakeLLM(*(tool_reply(call("get_group_status", {}, f"r{i}")) for i in range(4)))
        agent = self.agent(llm)
        with self.assertRaises(AgentError):
            agent.run(request("现在什么人格"))
        self.assertEqual(len(llm.requests), 4)
        self.assertEqual(agent.store.history(SessionKey.group(ACCOUNT, ROOM, ALICE), ("2026-10-23", 0)), ())

    def test_per_round_call_limit_answers_every_call_id(self):
        calls = [call("get_group_status", {}, f"c{i}") for i in range(5)]
        llm = FakeLLM(tool_reply(*calls), text_reply("好了"))
        reply = self.agent(llm).run(request("查一下状态"))
        self.assertEqual(len(reply.tool_calls), 3)
        tool_messages = [m for m in llm.requests[1]["messages"] if m["role"] == "tool"]
        self.assertEqual([m["tool_call_id"] for m in tool_messages], [f"c{i}" for i in range(5)])
        self.assertEqual(json.loads(tool_messages[3]["content"]), {"ok": False, "error": "单轮工具调用数量超限"})

    def test_ac4_scenario_a_tools_unsupported_degrades_to_plain_chat(self):
        llm = FakeLLM(ToolsUnsupported("HTTP 400"), text_reply("普通多轮回复"))
        with self.assertLogs("wxbot_framework.agent", "WARNING"):
            reply = self.agent(llm).run(request("你好呀，今天怎么样"))
        self.assertEqual(reply.text, "普通多轮回复")
        self.assertTrue(reply.degraded)
        self.assertTrue(reply.stored)
        self.assertIsNotNone(llm.requests[0]["tools"])
        self.assertIsNone(llm.requests[1]["tools"])
        self.assertEqual(roles(llm.requests[1]["messages"]), ["system", "user"])

    def test_ac4_scenario_a_enable_tools_false(self):
        for agent in (self.agent(FakeLLM(text_reply("不带工具")), enable_tools=False),
                      self.agent(FakeLLM(text_reply("不带工具"), enable_tools=False))):
            reply = agent.run(request("你好呀，今天怎么样"))
            self.assertEqual(reply.text, "不带工具")
            self.assertTrue(reply.degraded)
            self.assertIsNone(agent.client.requests[0]["tools"])

    def test_ac4_scenario_b_tool_soft_failures(self):
        llm = FakeLLM(tool_reply(call("get_member_recent_messages", {"member_name": "不存在的人"}, "c1"),
                                 call("broken", {}, "c2"),
                                 call("no_such_tool", {}, "c3"),
                                 call("get_member_recent_messages", {}, "c4"),
                                 ),
                      tool_reply(call("get_group_context", {}, "c5", error="工具参数不是合法的 JSON 对象"),
                                 call("get_group_context", {"limit": "many"}, "c6")),
                      text_reply("没找到这个人。"))
        with self.assertLogs("wxbot_framework.agent", "WARNING") as logs:
            reply = self.agent(llm, max_calls_per_round=4).run(request("不存在的人今天说了啥"))
        self.assertEqual(reply.text, "没找到这个人。")
        results = [json.loads(m["content"]) for m in llm.requests[2]["messages"] if m["role"] == "tool"]
        self.assertEqual(results[0], {"ok": False, "error": "未找到成员 不存在的人"})
        self.assertEqual(results[1], {"ok": False, "error": "工具执行出错"})
        self.assertEqual(results[2], {"ok": False, "error": "未知工具 no_such_tool"})
        self.assertEqual(results[3], {"ok": False, "error": "缺少参数 member_name"})
        self.assertEqual(results[4], {"ok": False, "error": "工具参数不是合法的 JSON 对象"})
        self.assertEqual(results[5], {"ok": False, "error": "参数 limit 类型不正确"})
        joined = "\n".join(logs.output) + json.dumps(llm.requests, ensure_ascii=False)
        self.assertNotIn("wxid_secret", joined)
        self.assertNotIn("secret\\path", joined)

    def test_tool_result_truncated_to_limit(self):
        self.registry.register(ToolSpec("huge", "大结果", object_schema(), lambda ctx, args: "长" * 5000))
        llm = FakeLLM(tool_reply(call("huge", {}, "c1")), text_reply("好"))
        self.agent(llm).run(request("给我大结果"))
        content = llm.requests[1]["messages"][-1]["content"]
        self.assertLessEqual(len(content), 2000)
        self.assertTrue(json.loads(content)["truncated"])

    def test_tool_timeout(self):
        release = threading.Event()
        self.registry.register(ToolSpec("slow", "慢工具", object_schema(),
                                        lambda ctx, args: release.wait(5) or {"ok": True}))
        llm = FakeLLM(tool_reply(call("slow", {}, "c1")), text_reply("超时了"))
        started = time.monotonic()
        with self.assertLogs("wxbot_framework.agent", "WARNING"):
            self.agent(llm, tool_timeout=0.2).run(request("跑慢工具"))
        release.set()
        self.assertLess(time.monotonic() - started, 2)
        self.assertEqual(json.loads(llm.requests[1]["messages"][-1]["content"]), {"ok": False, "error": "工具执行超时"})

    def test_write_tools_need_privilege_and_are_limited_per_trigger(self):
        saves = [call("memory_save", {"content": f"记忆{i}"}, f"w{i}") for i in range(3)]
        llm = FakeLLM(tool_reply(*saves[:2]), tool_reply(saves[2]), text_reply("记好了"))
        reply = self.agent(llm).run(request("记住这些", context=context(is_vip=True)))
        self.assertEqual(reply.text, "记好了")
        self.assertEqual(self.writes, ["记忆0", "记忆1"])
        last = json.loads(llm.requests[2]["messages"][-1]["content"])
        self.assertEqual(last, {"ok": False, "error": "本次对话的写入次数已达上限"})

        llm = FakeLLM(tool_reply(call("memory_save", {"content": "删库"}, "x")), text_reply("没权限"))
        self.agent(llm).run(request("记住这个", sender=BOB))
        self.assertEqual(self.writes, ["记忆0", "记忆1"])
        self.assertEqual(json.loads(llm.requests[1]["messages"][-1]["content"]), {"ok": False, "error": "无权限"})
        llm = FakeLLM(tool_reply(call("memory_save", {"content": "主号记忆"}, "y")), text_reply("好"))
        self.agent(llm).run(request("记住这个", context=context(is_self=True)))
        self.assertEqual(self.writes[-1], "主号记忆")

    def test_ac9_filler_turn_answered_but_not_stored(self):
        llm = FakeLLM(text_reply("周五去老地方。"), text_reply("哈哈是吧"), text_reply("还是老地方。"))
        agent = self.agent(llm)
        agent.run(request("周五聚餐去哪"))
        filler = agent.run(request("哈哈哈！"))
        self.assertEqual(filler.text, "哈哈是吧")
        self.assertFalse(filler.stored)
        agent.run(request("那几点？"))
        self.assertEqual(roles(llm.requests[2]["messages"]), ["system", "user", "assistant", "user"])
        self.assertEqual(llm.requests[2]["messages"][1]["content"], "周五聚餐去哪")

    def test_session_isolation_and_reset_through_agent(self):
        llm = FakeLLM(*(text_reply(f"回答{i}") for i in range(6)))
        agent = self.agent(llm)
        agent.run(request("小明的问题"))
        agent.run(request("小红的问题", sender=BOB))
        agent.run(request("小明在乙群", room=ROOM_B))
        self.assertEqual(roles(llm.requests[1]["messages"]), ["system", "user"])
        self.assertEqual(roles(llm.requests[2]["messages"]), ["system", "user"])
        agent.store.reset(ACCOUNT, ROOM, ALICE)
        agent.run(request("小明重置后"))
        self.assertEqual(roles(llm.requests[3]["messages"]), ["system", "user"])
        agent.run(request("小红继续", sender=BOB))
        self.assertEqual(llm.requests[4]["messages"][1]["content"], "小红的问题")
        agent.run(request("清空后", sender=BOB, generation=("2026-10-23", 1)))
        self.assertEqual(roles(llm.requests[5]["messages"]), ["system", "user"])

    def test_guard_expired_aborts_without_storing(self):
        state = {"current": True}

        def expire(messages, tools):
            state["current"] = False
            return tool_reply(call("get_group_status", {}, "c1"))

        llm = FakeLLM(expire, text_reply("不该发出"))
        agent = self.agent(llm)
        with self.assertRaises(SnapshotExpired):
            agent.run(request("现在什么人格", guard=lambda: state["current"]))
        self.assertEqual(len(llm.requests), 1)
        self.assertEqual(agent.store.history(SessionKey.group(ACCOUNT, ROOM, ALICE), ("2026-10-23", 0)), ())

        def raising_guard():
            raise SnapshotExpired("人格已切换")

        llm = FakeLLM(text_reply("不该发出"))
        with self.assertRaises(SnapshotExpired):
            self.agent(llm).run(request("你好", guard=raising_guard))
        self.assertEqual(llm.requests, [])

    def test_guard_expired_after_final_answer_does_not_store(self):
        state = {"current": True}

        def answer(messages, tools):
            state["current"] = False
            return text_reply("过期回答")

        agent = self.agent(FakeLLM(answer))
        with self.assertRaises(SnapshotExpired):
            agent.run(request("你好", guard=lambda: state["current"]))
        self.assertEqual(len(agent.store), 0)

    def test_total_budget_and_llm_errors_do_not_store(self):
        def slow(messages, tools):
            self.clock.now += 50
            return tool_reply(call("get_group_status", {}, "c1"))

        agent = self.agent(FakeLLM(slow, text_reply("不该到这里")))
        with self.assertRaises(AgentTimeout):
            agent.run(request("你好"))
        self.assertIsInstance(AgentTimeout("x"), LLMError)
        agent = self.agent(FakeLLM(tool_reply(call("get_group_status", {}, "c1")), LLMError("HTTP 500")))
        with self.assertRaises(LLMError):
            agent.run(request("你好"))
        self.assertEqual(len(agent.store), 0)

    def test_system_prompt_persona_memory_background_and_missing_ids(self):
        llm = FakeLLM(tool_reply(call("get_group_status", {}, "")), text_reply("好"), text_reply("再好"))
        snapshots = []

        def memory():
            snapshots.append(1)
            return "#1 每周五晚 20:00 读书分享"

        agent = self.agent(llm)
        agent.run(request("你好啊，周五干嘛", persona="说话简短", memory_snapshot=memory,
                          background=("之前说过想去爬山", "哈哈")))
        self.assertEqual(snapshots, [1])  # Frozen snapshot：一次触发只读一次
        system = llm.requests[0]["messages"][0]["content"]
        self.assertIn("不可信数据", system)
        self.assertIn("#1 每周五晚 20:00 读书分享", system)
        self.assertIn("说话简短", system)
        self.assertIn("之前说过想去爬山", system)
        self.assertNotIn("- 哈哈", system)
        self.assertLess(system.index("长期记忆"), system.index("当前人格"))
        assistant = llm.requests[1]["messages"][2]
        tool = llm.requests[1]["messages"][3]
        self.assertEqual(assistant["tool_calls"][0]["id"], tool["tool_call_id"])
        self.assertTrue(tool["tool_call_id"])
        # 已有多轮历史时不再注入冷启动背景
        agent.run(request("那周六呢", background=("之前说过想去爬山",)))
        self.assertNotIn("之前说过想去爬山", llm.requests[2]["messages"][0]["content"])

    def test_build_system_prompt_without_tools(self):
        prompt = build_system_prompt(with_tools=False)
        self.assertIn("不可信数据", prompt)
        self.assertNotIn("长期记忆快照", prompt)


if __name__ == "__main__":
    unittest.main()
