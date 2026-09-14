# L6 - Planning, Subgraphs and Multi-Agent：规划、子图与多 Agent

> 官方文档对应页
>
> - Graph API: <https://docs.langchain.com/oss/python/langgraph/graph-api>
> - Use the Graph API: <https://docs.langchain.com/oss/python/langgraph/use-graph-api>
> - Subgraphs: <https://docs.langchain.com/oss/python/langgraph/use-subgraphs>

## 通用学习目标与前置

一个任务家族贯穿全课：多份工艺文档分段摘要后汇总成检查报告（全部离线、脚本化 fake model，结构与协作是唯一变量）。学完本课应能独立解释并验证：

1. **三种分解结构同一任务对照**：单 Agent 大循环（L2 形状）、编译期固定流程（fixed pipeline）、planner + executor + replanner（Plan-and-Execute）；同一任务产出完全相同的报告，但 model 调用数、super-step 数、传入上下文条目数逐一可数（DEMO 1）。
2. **协作何时无收益**：任务固定、可整段拆分、无失败时固定流程在三项计数上全胜；只有"步骤集合运行时才可知、或需要失败后改计划"时规划结构才赢——赢家与输家都能用直接计数指出（DEMO 1 变体）。
3. **Send / map-reduce**：条件函数返回 `[Send("worker", payload), ...]`，N 在运行时决定；每个任务恰好执行一次，N 个 worker 落在同一个 super-step，reducer channel 汇聚（DEMO 2、`test_send_*`）。
4. **子图状态边界**：共享 schema（父 key 可见可写、"内部"写入直达父 state）vs 独立 schema + 显式状态翻译（翻译层是代码，边界由代码控制）（DEMO 3）。
5. **Command.PARENT**：子图内部节点直接路由并更新父图，跳过子图剩余路径；与父图侧静态出边并存时 goto 是叠加不是替换（DEMO 4）。
6. **checkpoint 命名空间**：完成的子图状态不在 `get_state(subgraphs=True)` 里，而在 `worker:<uuid>` 命名空间下，有独立 values、独立 step 计数、独立 history（DEMO 5）。**这是调试可见性，不是 Store**——Store 是跨会话记忆（DESIGN.md §7 校正行），本课全程未用 Store。
7. **委派与上下文隔离**：worker 只见委派载荷（message-content 级断言），隔离省掉上下文复制但需要兄弟上下文的任务必须由 supervisor 显式携带 hint，否则必败（DEMO 6）；任务失败后的重规划只重排剩余计划，已完成任务不重跑（DEMO 7）。

前置：L1 的 channel/reducer、super-step、`metadata.step`；L2 的 model/tools 循环、Command 路由、错误 envelope、终止由 runtime 决定；L5 的上下文分工概念（隔离比较一节与其呼应，本课自包含）。

## 方案取舍

| 选择 | 本课做法 | 为什么 | 生产替代 |
| --- | --- | --- | --- |
| 结构选择 | 同一任务三结构全跑，逐项计数对照 | "多 Agent 是否值得"必须先有可数的单 Agent 基线 | 按任务稳定性选其一，用 L7 评测验收 |
| 并行载体 | Send 运行时扇出 vs 编译期静态扇出（demo 1b 的 4 个 `add_node`） | N 何时确定是二者的唯一区别，对照后不再混淆 | 混合：固定主干 + Send 处理动态批次 |
| 子图状态 | 共享 schema 与独立 schema + 翻译各写一遍 | 共享的代价（无私有空间、出口重放）只有亲手数过才记得 | 大多选独立 schema + 翻译层，边界显式 |
| 父图路由 | `Command(goto=..., update=..., graph=Command.PARENT)` | "从子图内部改父图路由"这一种情形的原生载体 | 子图正常退出 + 父图条件边（可静态分析） |
| 子图观测 | checkpoint 命名空间寻址 + `stream(subgraphs=True)` 发现 ns | 完成后的子图状态只有这一条官方读出路径（probed） | MLflow trace（L7），不替代状态检查 |
| 重规划 | 失败以 envelope 进入 results，按需触发 replanner，`MAX_REPLANS` 预算 | 重规划是协调开销，只在失败时付；预算防死循环 | 真实 LLM replanner + 评测护栏 |
| fake model | 纯函数/队列脚本，`MODEL_JOURNAL` 记录每次调用的完整 context | 计数与"worker 见过什么"都是 journal 直查 | 同签名 live 适配器（本课离线，见完成记录） |

**核心取舍一句话：结构差异全部翻译成三项可数指标（model 调用、super-step、传入上下文条目），协作的收益与开销都从计数读出，不从感觉读出。**

## 运行前预测

先写下预测再运行（每条都能在输出里直接核对）：

1. 单 Agent 依次处理 4 段：第 5 轮（汇总）时模型收到几条消息？整个运行累计传入多少条？
2. 固定流程 4 个摘要节点并行：总 super-step 数是多少？plan-execute 串行执行 4 个任务又是多少？
3. 同一任务三结构的最终报告逐字相同吗？谁的 model 调用最少？
4. 第 2 段是扫描表格图片时：固定流程的报告长什么样？plan-execute 会重跑第 1 段吗？
5. `Send` 发出 5 个任务：`stream(updates)` 里 5 个 worker chunk 会落在几个 `values` 快照之间？
6. 去掉汇聚 channel 的 `Annotated` reducer：报错发生在哪一步，报错原文与 L1 的并发写冲突一致吗？
7. 子图节点返回 `Command(goto="escalate", graph=Command.PARENT)`，同时父图还留着 `add_edge("review", "normal_next")`：escalate 和 normal_next 各跑不跑？
8. 运行结束后 `get_state(config, subgraphs=True)`：能读到子图的内部 channel 吗？去哪里读？

## 执行模型与最小例子

三种结构（同一 4 段任务的实测计数，脚本化 fake，逐项可复算）：

```
(a) 单 Agent:   START -> model -> tools -> model -> ... -> END     一个上下文累积全部
(b) 固定流程:   START -> extract -> ( summary_0 ‖ summary_1 ‖ summary_2 ‖ summary_3 ) -> merge -> END
(c) Plan-Exec:  START -> planner -> executor ⇄ replanner -> report -> END   串行游标走计划
```

| 结构 | model 调用 | super-steps | 传入上下文条目 | 报告 |
| --- | --- | --- | --- | --- |
| (a) 单 Agent | 5 | 9 | 25（逐轮重发全部历史 1+3+5+7+9） | 4/4 完整 |
| (b) 固定流程 | 5（0 次协调） | 3 | 5（每调用 1 条） | 4/4 完整 |
| (c) Plan-Execute | 6（含 1 次 planner） | 6 | 6 | 4/4 完整 |

Send / map-reduce（DEMO 2，N 由输入长度在运行时决定）：

```
planner --[Send("worker", {"task": t}) for t in tasks]--> worker × N（同一 super-step）--> join
```

**Send 的载荷就是该 worker 这次调用的完整输入 state**——worker 物理上看不到 `chunks`/`results`，需要什么必须随任务携带。汇聚 channel 必须有 reducer；join 由入边屏障保证看到全部 N 份结果（`metadata.step` 与 values 快照双重验证）。

子图与父图的三条通道规则（probed on 1.2.11）：

| 机制 | 行为 |
| --- | --- |
| graph-as-node 输入 | 父 state 按子图 schema（同名 key）过滤后作为子图输入 |
| graph-as-node 输出 | 子图**最终 state**（同名 key）作为该节点的 update，经**父图 reducer** 合并 |
| 名称不匹配的 key | 双向都不流动：父图的私有 key 子图收不到，子图的私有 key 不进父 state |

checkpoint 命名空间的读出路径（probed on 1.2.11）：

| 路径 | 能看到什么 |
| --- | --- |
| `get_state(cfg, subgraphs=True)` 完成后 | `tasks == ()`，**不**显示嵌套子图状态 |
| `get_state(cfg, subgraphs=True)` 任务 pending 时（如子图内 interrupt） | `tasks[0].state` 是嵌套 StateSnapshot（values / next / interrupts / 命名空间 config） |
| `get_state({"thread_id": t, "checkpoint_ns": "worker:<uuid>"})` | 子图终态、独立 step 计数、独立 history（ns 用 `stream(subgraphs=True)` 发现） |

## 关键反例与修复

**反例 1：Send 扇出写无 reducer channel。** 去掉 `results: Annotated[list, operator.add]` 的 `Annotated`，3 个 worker 各写一次 `results`，invoke 报出与 L1 并发写冲突相同的原文（probed）：`At key 'results': Can receive only one value per step. Use an Annotated key to handle multiple values.`——**Send 不引入新的合并规则，它只是让"同节点多任务"并发发生在同一个 super-step 里**。修复：目标 channel 必须有 reducer（DEMO 2、`test_send_without_reducer_raises_invalid_update`）。

**反例 2：共享 schema 子图的出口重放。** 子图与父图共用 `DocState`，子图内部把"私有草稿"写到共享 channel：naive 出口处父 state 的 `notes` 中 `parent intake note` 出现 **2 次**（probed）。机制：子图节点的 update = 子图**最终 state**（含它 merely 收到的值），再经父图 `operator.add` 折叠——收到的东西被原样加回去。修复：给子图配 `input_schema`/`output_schema` 裁剪两个方向（出现 1 次）。但更根本的事实没变：**共享 schema 没有私有空间，"内部"写入直达父 state**（`test_shared_subgraph_writes_leak_and_duplicate`）。独立 schema + 显式翻译则让 `scratch` 在父 state 里根本不存在（`test_translated_subgraph_keeps_keys_private`）。

**反例 3：Command.PARENT 与父图静态出边并存。** 子图节点返回 `Command(goto="escalate", graph=Command.PARENT)`，父图同时留着 `add_edge("review", "normal_next")`：**两个目的地都执行**（probed，trail 直接可见），goto 是叠加不是替换——L2 反例的子图版。同时验证：Command.PARENT 会**跳过子图剩余路径**（`inner_tail` 不执行），并且**有无 checkpointer 都能工作**。修复：Command.PARENT 节点在父图侧不挂静态出边，路由只写在一处（DEMO 4、`test_command_parent_semantics`）。

**反例 4：完成后想看子图状态，读 `subgraphs=True` 读不到。** 完成的运行 `get_state(cfg, subgraphs=True).tasks` 是**空元组**；`get_state_history()` 在 1.2.11 上根本不接受 `subgraphs=` 参数（TypeError）。可行路径：用 `stream(subgraphs=True)` 拿到 `worker:<uuid>` 命名空间，再按 `checkpoint_ns` 寻址——子图有自己的 values、step 计数（从 -1 起）和 history（DEMO 5）。**注意边界：这是调试可见性问题；DESIGN.md §7 校正行明确 Store 是跨会话记忆，不是子图可见性的修复手段**，本课全程未用 Store（`test_subgraph_state_via_checkpoint_namespace`）。

**反例 5：隔离的代价——需要兄弟上下文的任务必败。** 交叉检查任务需要另一段文档的摘要，但委派载荷里没有：worker 返回错误 envelope 而不是猜（probed 附加事实：载荷里不属于子图 schema 的 key 会被**直接丢弃**，所以"偷偷多塞"也塞不进去）。修复只有一种：supervisor 显式把 `sibling_summary` 放进载荷（hint）。对照反模式 `include_history=True`：把父会话塞进每个任务，上下文复制了、隔离收益消失，还换不来任何新能力（DEMO 6、`test_delegation_isolation_and_sibling_hint`）。

## 跨场景迁移题（改坏它）

换成你自己的任务（如资料检索、代码检查）重定义 chunk 与摘要函数后，做三个破坏实验，每个先预测再运行：

1. **删掉 `MapReduceState.results` 的 `Annotated`**：预测报错发生在第几个 super-step、报错原文；对照 `test_send_without_reducer_raises_invalid_update`。
2. **把 `build_translated_subgraph_graph` 的翻译层删掉**，包装节点直接 `WORKER_SUBGRAPH.invoke(dict(state))` 传父 state：预测子图 inner 节点拿到什么（提示：名称不匹配的 key 不会流动）、在哪一行炸；对照 `test_translated_subgraph_keeps_keys_private`。
3. **把 replanner 改成从头重排**（`plan = scripted_plan(state["chunks"])` 且 `cursor=0`）：预测 `t1`/`t3`/`t4` 会被执行几次、`test_replanning_does_not_reexecute_completed_tasks` 哪条断言先失败。

```bash
poetry run pytest lessons/l6_planning_and_subgraphs -v
```

## 运行命令与来源

```bash
poetry run python lessons/l6_planning_and_subgraphs/main.py   # 离线机制实验（7 个 DEMO，无需网络）
poetry run pytest lessons/l6_planning_and_subgraphs -v        # 12 项机制测试
```

- 版本（本地实测）：Python 3.12.8，langgraph 1.2.11，langgraph-checkpoint 4.2.0，langchain-core 1.6.2，pytest 9.1.1。
- 文档：本页顶部三个官方链接（curriculum §7 的 L1/L2/L6 行）；背景见 L1/L2 README 的 super-step 与 Command 记录。
- 本课按渐进设计不配 MLflow 文件：观测全部来自 lib helpers（`print_history`/`thread`/`build_checkpointer`）与本地 stream chunk、checkpoint `metadata.step`。

## 完成记录

- 2026-09-14，`poetry run python lessons/l6_planning_and_subgraphs/main.py`：7 个 DEMO 全部按预期输出——三结构对照 5/5/6 调用、9/3/6 super-steps、25/5/6 上下文条目、报告逐字一致；变体任务固定流程 3/4 不完整（5 调用）vs plan-execute 重规划后 4/4 完整（9 调用，replans=1）；Send N=3/N=5 均为 `[1, N, 1]` 三段 super-step、每任务恰执行 1 次、`metadata.step=3`；无 reducer 反例报出与 L1 相同原文；共享子图 `parent intake note` 出现 2 次（trim 后 1 次、泄漏仍在）；Command.PARENT 跳过 `inner_tail`、与静态边并存时两目的地都执行；命名空间 `worker:<uuid>` 读出子图终态与独立 history（step -1..2），完成后 `subgraphs=True` tasks 为空、interrupt pending 时嵌套快照与 interrupt payload 可读、resume 成功；隔离委派 worker context 无父会话内容，`include_history` 泄漏可数，`include_hint` 恢复成功；重规划 t1/t3/t4 各 1 次、t2 失败 1 次 + prep 1 次 + 重试 1 次。
- 2026-09-14，`poetry run pytest lessons/l6_planning_and_subgraphs -v`：**12 passed**。
- 探针记录（写课件前用独立脚本在 1.2.11 上实测，未凭记忆断言）：`Send` 在 `langgraph.types`，载荷即该次调用完整输入 state；无 reducer 时 `InvalidUpdateError` 原文含 troubleshooting URL 尾行；graph-as-node 出口 update = 子图最终 state 经父图 reducer；载荷附加 key 被 schema 丢弃；`Command.PARENT` 无需 checkpointer、跳过子图剩余路径、与静态出边叠加；完成后 `get_state(subgraphs=True)` tasks 为空、`get_state_history` 不接受 `subgraphs=`、ns 寻址（`worker:<uuid>`）可读子图终态与独立 step/history。同一 super-step 内并行节点的完成顺序由线程调度决定（测试偶发翻转一次后改为按段内排序断言——段划分本身稳定，这正是 L1"顺序不强求一致"的再现）。
- 未验证（本课边界，非 skip 项）：真实模型下的规划/重规划质量（本课全部脚本化 fake，机制实验不声明任务效果）；跨进程/异步多 worker 的调度时序；`get_state(subgraphs=True)` 在 pending-interrupt 以外的 mid-run 场景（如 `stream` 中途外部调用）的表现。
