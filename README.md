# LangGraph Agent Learning

基于 LangGraph 官方文档的渐进式课程。目标是理解 LangGraph 的**运行机制**，而不是只会调 API。

```bash
poetry install
poetry run python lessons/l1_state_node_edge/main.py   # 跑 L1，打印完整执行过程
poetry run pytest lessons/ -v                          # 全部测试
```

## 设计原则

1. **核心原语优先** — State / Node / Edge 是地基。循环、检查点、中断、子图都只是这三样的组合变形。
2. **渐进依赖** — 每课的机制建立在前一课之上：不懂 Edge 的调度规则就讲不通循环，不懂检查点就讲不通中断。
3. **可跑可验证** — 每课都是确定性代码 + pytest，打印原始运行时输出，不需要 API key。跑两遍输出完全一样。

## 课程地图

| 课     | 主题                | 核心概念                                                    | 状态      |
| ------ | ------------------- | ----------------------------------------------------------- | --------- |
| **L1** | State / Node / Edge | super-step 执行模型、reducer、fan-out trigger rule          | ✅ 完成   |
| **L2** | 控制流              | 条件边、循环与 `recursion_limit`、`Command`、`Send`         | ⏳ 待搭建 |
| **L3** | 持久化              | checkpointer、`thread_id`、time travel、durability 三档     | ⏳ 待搭建 |
| **L4** | 中断                | human-in-the-loop、重放语义、`Command(resume=...)`          | ⏳ 待搭建 |
| **L5** | 子图                | 共享/独立 state 翻译、`Command.PARENT`、checkpoint 命名空间 | ⏳ 待搭建 |
| **L6** | 综合                | 审批 + 崩溃恢复 + 回滚的完整 agent                          | ⏳ 待搭建 |

学完 L1 你会知道：并发写无 reducer 的 channel 为什么报 `InvalidUpdateError`；同一 super-step 的两个节点为什么互相看不到写入；为什么"一短一长两条路径"会让汇合节点执行两次；私有 schema 为什么在流式输出时仍然泄露。

每课的目标、踩坑点、练习详见 [`docs/curriculum.md`](./docs/curriculum.md)。

## 官方文档索引

每次学一课，对照官方文档读代码。以下是按课程组织的阅读顺序（完整索引见本地 `llms.txt`，176 个 section）：

**通用 / 总览**

- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview.md)
- [Thinking in LangGraph](https://docs.langchain.com/oss/python/langgraph/thinking-in-langgraph.md) — 先读这篇建立心智模型
- [LangGraph runtime (Pregel)](https://docs.langchain.com/oss/python/langgraph/pregel.md) — super-step 调度模型的理论来源
- [Choosing between Graph and Functional APIs](https://docs.langchain.com/oss/python/langgraph/choosing-apis.md)

**L1 — State / Node / Edge**

- [Graph API overview](https://docs.langchain.com/oss/python/langgraph/graph-api.md) — schema、reducer、super-step、多 schema
- [Use the Graph API](https://docs.langchain.com/oss/python/langgraph/use-graph-api.md) — 条件分支、并行、递归限制
- [INVALID_CONCURRENT_GRAPH_UPDATE](https://docs.langchain.com/oss/python/langgraph/errors/INVALID_CONCURRENT_GRAPH_UPDATE.md) — 本课踩坑 A 的官方错误页

**L2 — 控制流**

- [Use the Graph API](https://docs.langchain.com/oss/python/langgraph/use-graph-api.md#create-and-control-loops) — 循环与终止条件
- [GRAPH_RECURSION_LIMIT](https://docs.langchain.com/oss/python/langgraph/errors/GRAPH_RECURSION_LIMIT.md)
- [Workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents.md) — 路由、编排、并行化模式

**L3 — 持久化**

- [Persistence](https://docs.langchain.com/oss/python/langgraph/persistence.md) — thread、checkpoint、命名空间
- [Checkpointers](https://docs.langchain.com/oss/python/langgraph/checkpointers.md) — 各后端取舍与 durability 三档
- [Use time travel](https://docs.langchain.com/oss/python/langgraph/use-time-travel.md) — 回放与分叉重跑
- [Memory](https://docs.langchain.com/oss/python/langgraph/add-memory.md) — 短期 / 长期记忆
- [MISSING_CHECKPOINTER](https://docs.langchain.com/oss/python/langgraph/errors/MISSING_CHECKPOINTER.md)

**L4 — 中断**

- [Interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts.md) — 重放语义与"每节点一次"规则
- [Fault tolerance](https://docs.langchain.com/oss/python/langgraph/fault-tolerance.md) — 重试、超时、错误处理

**L5 — 子图**

- [Subgraphs](https://docs.langchain.com/oss/python/langgraph/use-subgraphs.md) — 共享 state 与状态翻译
- [Stores](https://docs.langchain.com/oss/python/langgraph/stores.md) — 跨图共享数据
- [MULTIPLE_SUBGRAPHS](https://docs.langchain.com/oss/python/langgraph/errors/MULTIPLE_SUBGRAPHS.md)

**L6 — 综合**

- [Application structure](https://docs.langchain.com/oss/python/langgraph/application-structure.md)
- [Streaming](https://docs.langchain.com/oss/python/langgraph/streaming.md) / [Event streaming](https://docs.langchain.com/oss/python/langgraph/event-streaming.md)
- [Test](https://docs.langchain.com/oss/python/langgraph/test.md) — 图与 agent 的测试方法

**完整案例**

- [Agentic RAG](https://docs.langchain.com/oss/python/langgraph/agentic-rag.md)
- [SQL agent](https://docs.langchain.com/oss/python/langgraph/sql-agent.md)

## 目录结构

```
docs/curriculum.md      # L1-L6 学习路线图
docs/DESIGN.md          # 设计思路与课程规划
lib/helpers.py          # 教学辅助函数
lessons/lN_topic/       # 每课：README.md + main.py + test_main.py
llms.txt                # 官方文档全量索引（本地，不入库）
```

## 加新课程

每课固定三件套：

1. **README.md** — 讲清"为什么这一课在这里"、执行模型、踩坑点、改坏它的练习、对应官方文档
2. **main.py** — 可跑的最小代码，用 `lib.trace()` 打印原始 chunk，每个 demo 独立可跑
3. **test_main.py** — 每个踩坑点一个断言，用 `assert` 或 `pytest.raises`

约定：确定性、无 API key、无网络（假模型函数代替真实 LLM）。详见 [`docs/DESIGN.md`](./docs/DESIGN.md)。

## 技术栈

Python ≥3.12 · LangGraph ≥1.2.11 · langgraph-checkpoint-sqlite（L3）· pytest

MIT
