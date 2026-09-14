# L3 - Tools and Execution：工具系统与执行边界

> 官方文档对应页
>
> - MCP 文档（协议概念与 tools/list、tools/call 语义）：<https://modelcontextprotocol.io/docs/getting-started/intro>
> - MCP 规范仓库（schema 与版本演进）：<https://github.com/modelcontextprotocol/modelcontextprotocol>
> - LangChain/LangGraph Python 工具定义与注册（含 MCP server tools）：<https://docs.langchain.com/oss/python/langchain/tools>

## 通用学习目标与前置

L1 回答"状态怎么流动"，L2 回答"循环怎么转"；本课回答一个更根本的问题：**模型说"我想调用工具 X"之后、世界真的被改变之前，runtime 应该站在中间做哪些事**。课程总纲给 L3 的关键词是：工具 schema/语义/结果契约；只读与写入；并发冲突；超时、重试、取消；模型建议与权限执行分离；沙箱隔离；MCP 的协议边界。

本课不把 LangGraph 图当主角：大部分内容是一个纯 Python 的 `ToolExecutor`（这样每一层边界都由我们亲手写、亲手破坏）；只有在"工具循环穿过 PermissionGate"这一处，图是自然载体（DEMO 4 复用 L1/L2 的 propose/observe 循环形状）。

| # | 机制（必修） | 对应 DEMO | 对应测试 |
| --- | --- | --- | --- |
| 1 | 工具契约：schema 校验 + 结果 envelope；envelope 与 raise 的分工 | DEMO 1 | `test_invalid_args_return_envelope_not_exception`, `test_raising_tool_aborts_the_run` |
| 2 | 只读并行、冲突写受控（write lock 串行化，lost update 对照） | DEMO 2 | `test_reads_run_concurrently`, `test_conflicting_writes_lost_without_lock_controlled_with_it` |
| 3 | 超时 / 重试（只重试幂等只读）/ 取消，副作用计数为证 | DEMO 3 | `test_timeout_...`, `test_retry_policy_...`, `test_cancelled_...` |
| 4 | PermissionGate：模型建议、执行层裁决，拒绝作为 observation 回流 | DEMO 4 | `test_gate_rejects_...`, `test_gated_loop_feeds_...` |
| 5 | 沙箱：subprocess + cwd 钉死 + env 白名单 + 越界路径预检 | DEMO 5 | `test_sandbox_rejects_...`, `test_sandbox_pins_...` |
| 6 | MCP 协议边界：同一工具直连 vs 本地 stdio server 对照 | DEMO 6 | `test_mcp_direct_and_server_agree_...` |
| 7 | 故障可追踪：结构化执行日志 + 从日志归因失败 | DEMO 7 | （上述全部测试都在断言日志字段） |

前置：L1 的 super-step 并发与 channel 语义（DEMO 4 的循环图直接复用）；L2 的行动—观察反馈循环概念（本课的 envelope 就是给模型看的 observation）。不需要 L2 的代码。

## 方案取舍

| 决策点 | 本课选择 | 被放弃的方案 | 理由 |
| --- | --- | --- | --- |
| 工具失败的表达 | 两通道：`ToolFailure`/参数校验/权限/超时 → **结构化 envelope**（数据）；其他异常 → **raise 并终止**（bug） | 全部转 envelope；或全部抛出 | **预期的失败是数据，bug 不是**。把 traceback 喂给模型既浪费 token 又诱导它"修"不存在的参数；把 bug 转成 envelope 会让坏掉的 harness 带病继续跑 |
| 冲突写控制 | executor 持一把 write lock，**锁覆盖整个 read-modify-write 工具调用** | 每次写单独加锁；或第二个写直接拒绝 | 锁粒度必须等于"业务事务"粒度：锁在 `store.set` 上挡不住"读到旧值→算→写回"的交错。拒绝式控制留给 L4 的审批场景 |
| 重试策略 | 只重试 `kind == "readonly"` 的工具 | 按异常类型重试；全部重试 | 重试的前提是**幂等**。`flaky_write` 证明"效果已落地、响应丢失"的写重试一次就双写 |
| 隔离手段 | subprocess + cwd 钉死 + env 白名单 + 静态路径预检 | 直接 `eval`；docker/容器 | 本课要"实际配置的隔离环境、可验证越界失败"，subprocess 方案零依赖、每层可观测；同时诚实标注静态 guard 可被混淆，真正的强隔离要 OS 层机制 |
| MCP 对照方式 | 同一个函数对象：进程内直调 vs 本地 stdio server | 只测 MCP 一侧 | 只有**同工具双路径**才能把差异全部归因给协议边界本身（spawn、JSON-RPC、握手、发现） |
| LangGraph 的位置 | 只在 DEMO 4 用图（工具循环穿 gate） | 整课都建成图 | 本课主角是执行层的**横切边界**，不是调度；图循环恰好在 gate 反馈这一处是自然表达 |

## 运行前预测

先写下你的预测再运行 `main.py`（总纲要求每课至少一个关键预测由学习者先独立完成）：

1. 两个并发 `kv_add(+1)`（内部 sleep 制造交错窗口），**不加写锁**时最终值是多少？写入计数是多少？（提示：两者答案不同，这正是问题所在）
2. 挂起 5 秒的写工具配 0.3 秒 timeout，超时后 `store.writes` 是多少？
3. `flaky_write`（先写后抛 `ToolFailure`）在"只重试 readonly"策略下会尝试几次、落地几次？如果策略改成连写也重试呢？
4. 模型提议 `delete_all`（forbidden）时，工具函数体会不会执行？`store.writes` 变不变？
5. 沙箱里执行 `open('/etc/hosts').read()`，失败发生在 python 解释器（FileNotFoundError 之类）还是更早？
6. 同一个 `unit_convert`，进程内直调与走本地 MCP server 的单次调用延迟差多少量级？（先猜倍数）
7. 工具在 server 里抛出的 `ValueError("unknown to_unit ...")`，客户端在 MCP 结果里能看到多少？

## 执行模型与最小例子

```
model (proposes)              executor (decides + executes)          world (effects)
---------------              -----------------------------         ----------------
tool_call JSON  ───────────► PermissionGate ──► validate args ──► KVStore / subprocess
     ▲  feedback loop             │ reject            │ timeout / retry / cancel
     │                             ▼                   ▼
     └──────────── structured result envelope ◄── write lock
                                      │
                                      ▼
                        execution log (tool, args, decision, outcome, latency)
```

每次 `executor.call(name, args)` 的固定检查顺序：unknown tool → **PermissionGate** → 参数契约校验 →（重试循环内：write lock → timeout 包裹的调用）。每一步都会落一条 `ExecRecord`；权限拒绝额外进入 `rejections`（tool、args、reason），且**工具函数体从未被进入**——这是"模型建议、权限执行分离"的物理含义。

结果 envelope 只有两 种形状，模型侧永远不需要 try/except：

```python
{"ok": True,  "tool": "unit_convert", "args": {...}, "value": {"value": 3000.0, "unit": "m"}}
{"ok": False, "tool": "unit_convert", "args": {...}, "error": {"code": "invalid_args", "message": "..."}}
```

七个 DEMO（`python lessons/l3_tools_and_execution/main.py` 的输出按此顺序）：

1. **契约**：越界参数 → `invalid_args` envelope 且后续调用照常；对照 `boom` 工具 raise 非预期异常 → 整个 run 终止、后续步骤不执行。
2. **并发**：两个 0.3s 只读并发，墙钟 ~0.30s（证明重叠）；两个 `kv_add` 无锁 → lost update（写入计数 2、终值 1）；有锁 → 终值 2。
3. **超时/重试/取消**：挂起写在 0.3s 超时后 envelope `timeout`、`writes=0`；`flaky_read` 尝试 3 次后成功；`flaky_write` 只尝试 1 次、恰好落地 1 次；task cancel 后 `CancelledError` 上抛、日志记 `cancelled`、副作用 0。
4. **PermissionGate**：LangGraph 循环 `propose → execute → (propose | END)`，脚本模型提议 `[kv_get, kv_set, delete_all]`，后两个被拒并以 observation 回流，store 原样。
5. **沙箱**：界内写成功且文件落在 sandbox 内；`/etc/hosts` 与 `../../` 在**预检阶段**被拒（spawns 计数不变）；子进程 cwd 等于 sandbox root、env 只有白名单（macOS 会偷偷注入 `__CF_USER_TEXT_ENCODING`）。
6. **MCP 对照**：直调与本地 stdio server 结果逐项相等；实测 spawn+initialize 约 0.33~0.39 s（一次性），warm 调用约 0.7~3.6 ms/call，直调约 3~6 µs/call；`tools/list` 能拿到 `read_only_hint=True` 与 output schema；server 端 ValueError 的具体消息不出现在客户端结果里，只留在 server stderr。
7. **可追踪**：六步计划两处故障，从执行日志表格直接读出失败步、失败码、时延，配合 `store.writes` 区分"报告失败"与"实际落地"。

## 关键反例与修复

| 反例 | 现象（实测） | 根因 | 修复 |
| --- | --- | --- | --- |
| 工具内 raise 杀死整个 run | `boom` 抛 RuntimeError 后，后面的 `kv_set` 不执行，`writes=0`，异常直接穿透 executor | executor 只把 `ToolFailure` 当数据，其他异常视为 bug 上抛 | 这不是要"修掉"的行为，而是**设计**：预期失败走 envelope（模型可自纠），bug 走 raise（fail fast）。要修的是把 bug 误标成 ToolFailure 的工具 |
| lost update | 无写锁时两个 `kv_add(+1)`：`writes=2` 但终值 1 | 锁粒度错位：两个协程都先读到 0，再各自写 1 | executor 的 write lock 覆盖**整个工具调用**（含读），终值 2。注意此例中**副作用计数完全相同**——只数副作用无法发现冲突，必须同时对终值断言 |
| 重试写工具造成双写 | 若策略允许重试 `flaky_write`（效果已落地、响应丢失），将落地 3 次 | 重试默认幂等假设被非幂等写打破 | RetryPolicy 按 `kind` 门控：readonly 才重试。对确需重试的写，先幂等化（如带 request id 去重），那是 L4 的话题 |
| 超时后副作用已发生 | 若把 `slow_write` 的写挪到 sleep 之前，超时 envelope 照样返回但 `writes` 已 +1 | timeout 只能阻止"等待"，不能撤销"已发生的副作用" | 本课的工具把副作用放在挂点之后，所以可证 `writes=0`；真实系统里超时≠未执行，需要幂等键或补偿（结合执行日志判断） |
| 静态路径 guard 可被混淆 | `chr(47)+'etc'`、字符串拼接可绕过正则预检 | guard 是启发式，不是隔离机制 | 承认它是第一层；真正兜底的是 subprocess 的 cwd/env 限制与 OS 级隔离。教学点：**能证伪的才叫边界**，本课能证明的是"配置的越界访问在预检处失败、无 spawn" |
| MCP 边界吞错误细节 | server 端 `ValueError("unknown to_unit 'furlong'...")`，客户端只看到 `Error executing tool unit_convert` | 协议把工具异常折叠成通用错误，traceback 走 server stderr | 工具应当**自己**把预期失败编码进结果（如 envelope），不要依赖异常文本跨进程存活；排障时用 `stdio_client(errlog=...)` 收集 server stderr |

## 跨场景迁移题

选两三个"改坏它"，先预测再动手，跑测试看哪个断言抓住你：

1. 把 `ToolExecutor` 默认 `serialize_writes=True` 改成 `False`：预测 `test_conflicting_writes_lost_without_lock_controlled_with_it` 里哪一半先红。
2. 把 `RetryPolicy` 的 `retry_kinds` 加上 `"write"`：预测 `test_retry_policy_retries_readonly_only` 里 `store.writes - writes_before` 会变成几。
3. 把 `call()` 里 PermissionGate 的两个 if 分支删掉（权限表留着没人查，模拟"权限只写在工具 docstring 里"）：预测哪些断言会红；再注意该文件最后一个断言（approved=True 的 `kv_set` 落地、`writes==1`）改动前后**都**成立——只测"放行的写能成功"，测不出权限缺失。

```bash
poetry run pytest lessons/l3_tools_and_execution -v
```

迁移到非教学场景时问自己：我的"KVStore 计数器"在真实系统里是什么（数据库行？外部 API 调用？）——那里还有没有能直接数的 `writes`？没有的话，你拿什么证明"拒绝零副作用"？

## 运行命令与来源

```bash
# 离线机制实验（确定性，无网络）
poetry run python lessons/l3_tools_and_execution/main.py

# 全部机制测试（12 个，含真实本地 MCP stdio 往返；约 3~4 s）
poetry run pytest lessons/l3_tools_and_execution -v

# 单独把 MCP server 作为脚本跑起来（用客户端连接它才会退出）
poetry run python lessons/l3_tools_and_execution/mcp_server.py

# live 实验（未配置 .env 时打印配置说明并以 0 退出）
poetry run python lessons/l3_tools_and_execution/live_demo.py
poetry run pytest lessons/l3_tools_and_execution/test_live.py -v   # 无 .env 时 2 skip
```

| 来源 | 用途 |
| --- | --- |
| <https://modelcontextprotocol.io/docs/getting-started/intro> | MCP 协议概念：initialize 握手、tools/list、tools/call |
| <https://github.com/modelcontextprotocol/modelcontextprotocol> | 规范仓库：schema 与版本（本机实测协商版本 2025-11-25） |
| <https://docs.langchain.com/oss/python/langchain/tools> | LangChain/LangGraph 工具定义、`@tool`、MCP server tools |
| 本机安装 `mcp` 2.2.0 的实测 API（写入 `mcp_server.py` docstring） | `mcp.server.mcpserver.MCPServer`；`stdio_client`/`ClientSession` 生命周期 |

## 完成记录

- 2026-09-14：`poetry run python lessons/l3_tools_and_execution/main.py` 全部 7 个 DEMO 跑通（输出含各断言的实测值：lost update 终值 1/加锁终值 2、timeout/cancel 后 `writes=0`、flaky_write 落地 1 次、gate 拒绝 2 条、沙箱越界 0 spawn、MCP 结果逐项相等）。
- 2026-09-14：`poetry run pytest lessons/l3_tools_and_execution -v` → **12 passed, 2 skipped**（skip 的 2 个是 live 测试）。MCP 测试为真实本地 stdio 往返，无 skip。
- MCP 实测数字（多次运行，机器相关会浮动）：spawn + initialize 约 **0.33~0.39 s**（一次性）；warm 调用 **0.7~3.6 ms/call**（不同运行差异较大，首次探测 0.76 ms，带 errlog 重定向后 2.3~3.6 ms）；直调同一函数 **3~6 µs/call**；`tools/list` 可读回 `read_only_hint=True` 与 `output_schema`。
- 工具异常跨 MCP 边界的实测：server 端 `ValueError` → 客户端 `is_error=True`、content 仅 `"Error executing tool unit_convert"`，详细消息只在 server stderr（经 `stdio_client(errlog=...)` 捕获验证）。
- **未验证（未配置 .env）**：`live_demo.py` 的真实模型提议/拒绝回流场景；`test_live.py` 2 个测试 skip。配置 `.env`（OPENAI_API_KEY 等）后重跑。
- 沙箱 env 白名单实测：子进程仅见 `LANG`、`PYTHONIOENCODING`（macOS 另注入 `__CF_USER_TEXT_ENCODING`），`HOME` 不在其中。
