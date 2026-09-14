# L2 - The Agent Loop：决策、工具循环与终止

> 官方文档对应页
>
> - Graph API: <https://docs.langchain.com/oss/python/langgraph/graph-api>
> - Use the Graph API: <https://docs.langchain.com/oss/python/langgraph/use-graph-api>
> - Pregel: <https://docs.langchain.com/oss/python/langgraph/pregel>

## 通用学习目标与前置

L1 回答了"一个 super-step 里发生什么"；本课回答"super-step 如何串成循环"。学完本课应能独立解释并验证：

1. **原生 Python `while` 循环与图循环是同一个决策结构**：`break` 对应条件边/Command 路由到 `END`，回边 `tools -> model` 对应 `while True`。同一脚本驱动三种实现，轨迹逐一相同（`test_main.py` 前三条测试断言）。
2. **ReAct 式行动—观察**：模型发出结构化 `tool_calls`，runtime 执行工具并把 observation 追加回消息历史，模型下一轮看到它。坏参数也是 observation，不是异常。
3. **条件边与 Command 两种路由载体**：条件边把路由写成图上的函数；`Command(goto=..., update=...)` 把"更新"与"路由"合并进节点返回值本身。
4. **终止是 runtime 的决定**：成功 / 失败（连续工具错误）/ 无进展（重复调用）/ 预算（步数或 token），四种 `stop_reason` 都在状态里留下可从 trace 读出的指纹。
5. **从 trace 解释停止原因**：不看代码，只看最后的 `updates` / `values` chunk 和终态，说出这次为什么停。

前置：L1 的 channel/reducer、super-step、checkpoint `metadata.step`。业务场景沿用 PCB 工艺参数（`unit_convert`、`lookup_spec` 两个确定性工具）。

## 方案取舍

| 选择 | 本课做法 | 为什么 | 生产替代 |
| --- | --- | --- |
| 循环载体 | 同一循环写三遍：`while` / 条件边图 / Command 图 | 三者可逐行对照，"图循环"不再神秘 | 图版本（可 checkpoint、可恢复，L4 验证） |
| 工具调用格式 | OpenAI chat-completions `tools` wire format，**手写解析与分发** | 解析、执行、回传每一步可见、可测 | `langgraph.prebuilt` 的 `ToolNode` / `create_react_agent`（产品里应使用） |
| 工具出错 | 返回结构化 error envelope（`{"ok": false, "error": ..., "hint": ...}`）作为 observation 回传 | 运行存活，模型有机会自我纠正（DEMO 3 已验证恢复） | 同左；崩溃才 raise |
| 重复检测 | runtime 维护 `seen_calls`（tool + 规范化 args），重复请求拒绝执行 | 无进展必须由 runtime 判定，模型自己不会承认原地踏步 | 加滑动窗口/相似度，避免误杀合法重试 |
| 预算 | `MAX_STEPS` 与 `TOKEN_BUDGET` 两个计数器，同一个 `stop_reason="budget"` | 步数防死循环、token 防账单，缺一不可 | 按 run/会话累计（L7 系统化） |
| fake 与 live | `FakeModel.respond(messages)` 与 OpenAI 适配器同签名 | 离线机制实验与真实模型闭环共用全部图代码与终止策略 | 同左 |

**核心取舍一句话：本课手写一切可手写的环节以换取可观测性，生产替换件在表右侧，不因替换而改变终止策略。**

## 运行前预测

先写下你的预测再运行 `main.py`（每条都能在输出里直接核对）：

1. 三种实现在同一脚本下，`steps` / 工具执行序列 / `stop_reason` / token 数会完全一致吗？哪一种实现会在哪一步分岔？
2. 失败脚本里模型连续 3 次发坏参数：第 3 次错误之后，脚本里剩下的"最终回答"还会被消费吗？为什么？
3. 同一工具 + 同一参数被要求第二次：这次请求会不会被执行？它消耗了一次 model turn 吗？
4. 一个节点返回 `Command(goto="tools")`，但图上同时留着它的静态出边 `add_edge`：`compile()` 报错、`invoke()` 报错，还是两个目的地都跑？
5. 工具返回 `{"ok": false}` 与工具直接 `raise`：观察消息历史，哪一种运行能走到 `stop_reason`？

## 执行模型与最小例子

图拓扑（条件边版；Command 版节点相同，除 `START -> model` 外零条边）：

```
START -> model --(has tool_calls?)--> tools --(no stop)--> model   <- 回边成环
          |                                   |
          +--------(stop_reason set)--------->+----> END
```

状态关键点：`messages` 是 `Annotated[list, operator.add]`——model 节点与 tools 节点都往里追加；其余 channel（`steps`、`tokens_used`、`consecutive_errors`、`seen_calls`、`stop_reason`、`final_answer`）都是 LastValue，每步单写者。检查顺序是三种实现共用的契约：**model 节点先预算闸门再消费响应；tools 节点先查重复再执行；错误只在批内累计、任何成功清零。**

四种终止及其 trace 指纹（**不看代码，从最后几个 chunk 就能判定的部分**）：

| stop_reason | 谁写入 | 最后的 updates chunk | 终态特征 |
| --- | --- | --- | --- |
| `success` | model | model 写入 `stop_reason='success'` + `final_answer`，无 `tool_calls` | `final_answer` 非空，`next=()` |
| `tool_error` | tools | tools 写入第 N 个 error observation + `stop_reason` | `consecutive_errors == ERROR_LIMIT`，脚本未消费完 |
| `no_progress` | tools | tools **只写** `stop_reason`，无任何 observation | 执行次数 < 请求次数 |
| `budget` | model | model **只写** `{'stop_reason': 'budget'}`，没有追加消息 | `steps == MAX_STEPS` 或 `tokens_used >= TOKEN_BUDGET` |

流式两种模式各显示什么（DEMO 2）：

- `stream_mode="updates"`：**每个完成的 task（节点执行）一个 chunk**，内容是该节点的 partial update。注意 langgraph 1.2 的 chunk 计数是 task 数不是 super-step 数——本课每步单节点所以相等，L1 的 fan-out 下不相等；**权威计数是 checkpoint `metadata.step`**（DEMO 2 打印了 history）。
- `stream_mode="values"`：**每个 super-step 后的完整状态快照**。最后一个 values chunk 即终态，`stop_reason` 字段直接可读。

## 关键反例与修复

**反例 1：坏参数——envelope 回传 vs 直接 raise。** 模型把 `to_unit` 写成 `"milimeter"`：`dispatch_tool` 返回 `{"ok": false, "error": "unknown unit 'milimeter'", "hint": "valid units: [...]"}` 作为 tool observation 进入消息历史，下一轮模型按 hint 纠正，运行以 `success` 结束（DEMO 3、`test_bad_params_are_fed_back_and_model_recovers`）。同一个调用换成 raising dispatcher：`RuntimeError: tool 'unit_convert' crashed before producing an observation` 直接炸掉 `invoke()`，没有任何 observation 被追加——**反馈回路根本没有机会发生**。修复原则：可预期的参数错误永远走 envelope，raise 只留给真正的崩溃。

**反例 2：Command 与静态边并存。** 节点返回 `Command(goto="tools", update=...)` 的同时，builder 里还留着 `add_edge("agent", "fallback")`。langgraph 1.2.11 实测：`compile()` 完全不报错；`invoke()` 时 **goto 是叠加而不是替换**——`tools`（经 Command）和 `fallback`（经静态边）进入**同一个 super-step**。两者写同一个无 reducer channel 时报错（probed 原文）：

```
InvalidUpdateError: At key 'messages': Can receive only one value per step. Use an Annotated key to handle multiple values.
```

更隐蔽的是通道不冲突的同款错误：**没有任何异常**，两个目的地都执行（DEMO 7、`test_command_does_not_suppress_static_edge_silently`），只有直接计数能发现。修复：Command 节点不挂静态出边，路由只写在一处；迁移到 Command 时删干净旧边。这也回应 DESIGN.md 校正表：Command 的价值是**更新与路由组合在一次返回里**，不是"默认更快"或提供外部原子性。

**反例 3：终止检测器之间的相互作用。** 失败脚本必须每次用**不同**的坏参数：若三次都发同一坏参数，`no_progress` 检测器会在第 2 次就终止（`stop_reason='no_progress'` 而不是 `tool_error'`）。两个检测器都在 tools 节点、先后固定（先重复、后错误），谁的条款先满足谁触发——**终止理由取决于检测顺序，不只是一组条件**。

## 跨场景迁移题（改坏它）

换成你自己的任务（如资料检索、代码检查）重定义两个工具后，做三个破坏实验，每个先预测再运行：

1. **把 `ERROR_LIMIT` 从 3 改成 1**：预测 `script_bad_params_recovery` 的轨迹——第一次坏参数就直接终止，"自我纠正"消失。验证：`poetry run pytest lessons/l2_agent_loop -v`（对应测试会失败或轨迹改变，说明恢复依赖"允许连续出错"的设计）。
2. **把 `tools_step` 里的重复检测删掉（或把 key 改成只有工具名）**：预测 `script_no_progress` 会走到哪个终止（提示：会一直执行到 `budget`）。直接计数验证第二次调用确实被执行了。
3. **给 Command 版图加一条 `add_edge("model", "tools")`**：预测报错时机与信息，再用 `pytest.raises(InvalidUpdateError, match="Can receive only one value per step")` 对照——注意本课的 model/tools 两节点都写 `messages`（有 reducer），所以要复现冲突需用一个无 reducer 的 channel（参考 `build_command_static_conflict_graph` 的构造）。

```bash
poetry run pytest lessons/l2_agent_loop -v
```

## 运行命令与来源

```bash
poetry run python lessons/l2_agent_loop/main.py        # 离线机制实验（7 个 DEMO，无需网络）
poetry run pytest lessons/l2_agent_loop -v             # 10 项机制测试
poetry run python lessons/l2_agent_loop/live_demo.py   # 真实模型闭环（需 .env；可选 MLflow）
poetry run pytest lessons/l2_agent_loop/test_live.py -v # live 测试（未配置 .env 时 skip）
```

- 版本（本地实测）：Python 3.12.8，langgraph 1.2.11，langchain-core 1.6.2，openai 3.13.0，mlflow 3.16.0，pytest 9.1.1。
- 文档：本页顶部三个官方链接；ReAct 与工具调用背景见 curriculum §7 的 L0/L1/L2/L6 条目。
- MLflow 复用 L1 已验证事实：langchain flavor 挂 tracer、trace 异步上报（读回带 `flush=True`）、`locations=` 替代 `experiment_ids=`、span 树形状可信而兄弟时间重叠不可断言。

## 完成记录

- 2026-09-14，`poetry run python lessons/l2_agent_loop/main.py`：7 个 DEMO 全部按预期输出——三实现轨迹对照两行均 `True`；四种终止各就位（`success`/`tool_error` 3 次、`no_progress` 第 2 次拒执行、`budget` 步数 6/10 与 token 150>=120 两变体）；反例 1 恢复闭环与 raise 对照；反例 2 报出与 README 相同的 probed 错误原文，静默变体两目的地均执行。
- 2026-09-14，`poetry run pytest lessons/l2_agent_loop -v`：**10 passed, 2 skipped**（skipped 均为 live 测试，理由 live model not configured）。
- 2026-09-14，`poetry run python lessons/l2_agent_loop/live_demo.py`：无 `.env`，打印配置摘要后 exit 0——**真实模型工具闭环：未验证（未配置 .env）**。
- live 相关（`test_live.py` 两项、live_demo 的 MLflow run/trace/成本记录）：**未验证（未配置 .env）**。本地 MLflow 服务器在运行（localhost:5000，experiment `agent-loop`），但无 live 模型调用故本课未产生新 trace。
- 探针记录：Command/静态边行为在写课件前用独立小脚本在 langgraph 1.2.11 上实测（`compile()` 静默、同 super-step 双目的地、冲突通道报错原文），未凭记忆断言。
