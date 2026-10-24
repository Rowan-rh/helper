# 微信文本机器人框架
项目已在linux.do公开声明https://linux.do/


面向 Windows x64、微信 4.1.15.13 的 Python 插件框架。统一入口为 `main.py`，原生桥接负责文本收发，功能以插件加载。


# 本项目仅供学习，请勿用于违法行为

## 功能

- 文件传输助手、好友和群聊文本收发，以及真正的群艾特。
- 群聊定时提醒（`@机器人` 自然语言/规则提醒与 `/提醒`、`/我的提醒`、`/取消提醒`、`/取消全部提醒`，到点自动 `@` 设置人，停机 10 分钟宽限补发，无需配置大模型即可使用）。
- 成语接龙：独立玩家、合并抢答和退出模式。
- `/绑定 昵称`：只设置当前群的称呼；不同群各自保存。未绑定时使用联系人缓存中的微信名。
- `/总结 昵称`、`/总结群聊`：通过配置的大模型总结当前群当天的文字记录。
- `/切换人格 嘴臭|卑微|夸夸|取消`：为当前群的对话、总结和接龙点评设置表达风格。
- 真正艾特机器人后对话，携带该成员在当前群当天此前最多十条有效发言，支持多轮上下文与群内安全工具调用。
- 私聊 VIP 对话与文件传输助手 `/ai` 对话入口。
- 每天北京时间 0 点清理旧日文本与多轮上下文；`/清空` 手动清理当前群文本与会话；定长长期记忆与 `/记忆` 管理。
- 按文本长度随机等待后发送。
- 插件启停、重载与开发模板。

## 运行环境

Windows 10/11 x64、64 位 Python（已验证 Python 3.13）、`requirements.txt` 中的依赖，以及 Microsoft Visual C++ x64 运行库。

## 首次配置

在项目目录运行：

```powershell
python -m pip install -r requirements.txt
python tools/init_config.py
```

1. `bot_config.json`：将 `wechat_account` 改为自己的机器人微信 ID。绑定保存的是账号 ID，重新登录后 PID 改变不影响选择；多开时会等待指定账号，不改用其他账号。
2. `members_config.json`：在 `groups` 中添加自己要启用的群 ID，成员列表可以为空。若希望对所有群聊开放，可配置 `"allow_all_groups": true`（配合 `bot_config.json` 中的 `"chat_allowlist": []` 可对所有群聊开放 `@机器人`，`agent_vip_wxids` 仍仅控制私聊与群记忆写入）：

```json
{
  "database_file": ".runtime/member_data.sqlite3",
  "history_enabled": true,
  "daily_clear": true,
  "allow_all_groups": false,
  "groups": {
    "12345678@chatroom": []
  }
}
```

3. `llm_config.json`：填写自己的 `api_key`、`base_url` 和 `model`。模板保留 DeepSeek 接口格式，密钥为空。也支持 MiniMax 官方接口（国内 `https://api.minimax.cn/v1`，海外 `https://api.minimax.io/v1`），参考 `llm_config.minimax.example.json`，注意：`model` 必须明确填写（如 `MiniMax-M3`）；MiniMax 端点不支持 JSON 输出模式，框架会用提示词约束并从回复文本中提取 JSON；M2 系列和 M3.1-Flash 无法关闭思考且思考计入 `max_tokens`，需把 `max_tokens` 调大（如 2048），M3.1-Flash 也不能与关闭思考参数同用。
4. 模板默认关闭需要模型的 `idiom_chain`、`chat_summary`、`mention_chat` 插件。填写模型配置后，把它们的 `enabled` 改为 `true`。

账号 ID 必须是实际微信 ID；群 ID 以 `@chatroom` 结尾。`test` 消息会由 `id_responder` 插件回复当前会话 ID，`chats` 控制台命令可以列出本次连接收到消息的会话。所有示例 ID 和测试用名称均为虚构数据。

## Agent 与多轮对话配置

框架支持基于 OpenAI 兼容接口 / MiniMax 工具调用的 Agent 循环，提供整轮裁剪的多轮上下文、群内安全只读/写工具以及分群定长长期记忆。

### 1. 入口说明
- **群聊入口**：群内真实艾特机器人触发 Agent 循环；保留群人格语气约束，超长文本按 1023 字节自动切分，携带说话人近期发言作为冷启动背景（自动过滤语气词）。
- **私聊入口**：仅 VIP 发送者可触发私聊多轮对话，回复不带 `@` 标签；非 VIP 发送者的私聊消息一律静默忽略。
- **文件传输助手入口**：机器人主号本人在手机端文件传输助手发送带 `/ai ` 前缀的消息（如 `/ai 帮我分析今天的任务`）时触发 Agent，前缀自动剥离后作为提问；框架自动排除 `bot_generated` 回显，防止自己触发自己形成死循环。

### 2. VIP 微信 ID 配置与核对
在 `members_config.json` 中配置 `agent_vip_wxids` 列表（默认为空 `[]`）：
```json
{
  "database_file": ".runtime/member_data.sqlite3",
  "history_enabled": true,
  "daily_clear": true,
  "allow_all_groups": false,
  "agent_vip_wxids": [
    "wxid_example_vip"
  ],
  "groups": {
    "12345678@chatroom": []
  }
}
```
> **注意**：VIP 必须填写微信底层 ID（`wxid_*`），不能填写微信号或昵称。可以在微信中让目标好友向机器人发送 `test`，由 `id_responder` 插件回复该会话的真实 wxid 进行核对。

### 3. 命令用法
- `/重置对话`：重置当前发送者在当前会话（群聊/私聊/文件助手）的多轮上下文；所有人开放，不影响同群其他人，长期记忆保留。
- `/记忆`：查看当前群聊或私聊保存的长期记忆列表（仅 VIP 与主号本人可用）。
- `/删除记忆 编号`：删除指定编号的长期记忆条目（仅 VIP 与主号本人可用）。
- `/清空记忆`：清空当前群或当前会话的全部长期记忆（仅 VIP 与主号本人可用）。
- `/ai 问题`：文件传输助手中唤醒 Agent 的专用前缀。
- `/清空`：清理当前群当天的聊天记录，同时释放当前群的多轮会话内存，长期记忆保留。
- `/help`：查看命令帮助清单。

### 4. 工具说明
- **群聊只读工具**：`get_group_context`、`get_member_recent_messages`、`list_group_members`、`get_group_status`。
- **记忆写工具**：`memory_save`、`memory_delete`（仅 VIP 与主号本人可用）。
- **群聊定时提醒工具**：`reminder_create`、`reminder_list`、`reminder_cancel`（开启 `mention_chat` 时支持更口语化的时间表达，与规则路径共用存储与安全配额）。

## 启动与验证

```powershell
python main.py --check
python main.py
```

`--check` 只检查配置、插件清单与语法，不注入微信，也不调用模型。正常运行前打开微信并登录配置中的账号。

也可以运行 `start_bot.ps1`：脚本优先使用项目的 `.venv`，其次使用 PATH 中的 Python；可通过 `-Python` 或环境变量 `WXBOT_PYTHON` 指定解释器，不依赖某台电脑的安装路径。

控制台命令：

```text
status
chats
members [群ID]
history 称呼 [条数]
plugins
enable 插件名
disable 插件名
reload [插件名]
quit
```

增加群、修改账号或成员配置后重启。停止时使用 `quit`，等待正常卸载原生桥接；微信可以保持打开。

`wechat_account` 设为 `null` 时，原生收发层可以选择唯一登录账号；启用成员目录和群记录需要明确配置机器人账号，以隔离不同账号的数据。

## 目录

```text
main.py                  统一入口
wxbot_framework/         框架、成员目录、消息缓存、大模型客户端
plugins/                 各功能插件和开发模板
native/                  原生桥接源码、两个必要 DLL、版本指纹和 MinHook
tests/                  本地测试
tools/                   本地配置初始化和发布前检查
```

