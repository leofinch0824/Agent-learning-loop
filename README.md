# LangGraph Agent Learning

面向算法工程师的 Agent 实战课程：理解 LangGraph 运行机制，逐步构建沉淀业务知识、调用和优化 Skill、分析多模态错例并挖掘后训练数据的业务模型迭代助手。沿用本地 MLflow（实验空间 `agent-loop`）进行可观测性练习。

当前学习路线见 [curriculum.md](docs/curriculum.md)，完整目录见 [落地设计](docs/course-implementation.md)，已确认目标见 [learning-goals.md](docs/learning-goals.md)。按每周 13–15 小时设计，4 周先交付最小优化闭环，8 周补齐 Skill 迭代与数据工作流。当前仅 L1 有课件；以下命令用于现有实现，后续课程待开发。

```bash
docker compose up -d                                     # 起 MLflow（localhost:5000）
poetry install
poetry run python lessons/l1_state_node_edge/main.py         # 跑 L1，打印完整执行过程
poetry run python lessons/l1_state_node_edge/mlflow_demo.py  # L1 x MLflow：trace + run + 读回
poetry run pytest lessons/ -v                                 # 全部测试（MLflow 不可达时相关测试自动 skip）
```

## 设计原则

1. **机制服务业务** — 先掌握状态、路由与恢复，再将 Skill 和知识接入同一个项目。
2. **评测先于优化** — 先建立真实任务基线与人工核验，再生成和选择候选版本。
3. **分层验证** — 离线机制实验用确定性行为断言，真实业务用付费 API 与人工结果验证，独立迁移练习检查掌握程度。

## 课程地图

| 课 | 主题 | 状态 |
| --- | --- | --- |
| L1 | State / Node / Edge | 已有课件，部分概念和断言待校正 |
| L2 | Agent 循环、工具与真实模型 | 待开发 |
| L3 | 持久化与可恢复人工审核 | 待开发 |
| L4 | 业务知识、Skill 与基础流程组合 | 待开发 |
| L5 | 多模态执行与评测 | 待开发 |
| L6 | 错例分析与归因 | 待开发 |
| L7 | 自动提示词优化 | 待开发 |
| L8 | Skill 自迭代与版本管理 | 待开发 |
| L9 | 训练数据挖掘与人工核验 | 待开发 |
| L10 | 业务模型迭代助手综合验收 | 待开发 |

学完 L1 你会知道：并发写无 reducer 的 channel 为什么报 `InvalidUpdateError`；同一 super-step 的两个节点为什么互相看不到写入；为什么"一短一长两条路径"会让汇合节点执行两次；私有 schema 为什么在流式输出时仍然泄露。

每课的目标、踩坑点、练习详见 [`docs/curriculum.md`](./docs/curriculum.md)。

## 原 LangGraph 课程的官方文档索引（旧课号）

以下保留原 L1–L6 的主题索引，课号不对应上方新课表。新课程按课阅读入口见 [curriculum.md](docs/curriculum.md)；本地 `llms.txt` 保留原官方文档索引。

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

- [Interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts.md) — 重放语义与多个中断的调用顺序
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
docs/curriculum.md      # 当前 L1-L10 学习路线图
docs/course-implementation.md # 目标目录、业务流程、实验与迁移设计
docs/learning-goals.md  # 已确认目标、时间与交付范围
CONTEXT.md             # 业务术语
docs/DESIGN.md          # 原 LangGraph 机制课设计背景
lib/helpers.py          # 教学辅助函数
lib/mlflow_utils.py     # MLflow 连接 / trace 读取 / span 树打印
lessons/lN_topic/       # 每课：README.md + main.py + test_main.py（+ mlflow_demo.py / test_mlflow.py）
docker-compose.yml      # 本地 MLflow（localhost:5000，数据落在 mlflow-data/，不入库）
llms.txt                # 官方文档全量索引（本地，不入库）
```

## 加新课程

每课固定三件套：

1. **README.md** — 讲清"为什么这一课在这里"、执行模型、踩坑点、改坏它的练习、对应官方文档
2. **main.py** — 可跑的最小代码，用 `lib.trace()` 打印原始 chunk，每个 demo 独立可跑
3. **test_main.py** — 每个踩坑点一个断言，用 `assert` 或 `pytest.raises`

外加一对可选的 MLflow 组件（L1 起提供模板）：`mlflow_demo.py`（连接本地服务器、打 trace、记 run、读回验证）与 `test_mlflow.py`（服务器不可达时自动 skip）。

约定：离线机制实验无需 API key；真实业务实验明确选择 live 模式、记录模型与预算，未运行时标记“未验证”。课件开发沿用可跑代码、行为断言和原始运行证据；新课程范围以 [落地设计](docs/course-implementation.md) 为准，原机制课背景见 [DESIGN.md](docs/DESIGN.md)。

## 技术栈

Python ≥3.12 · LangGraph ≥1.2.11 · langgraph-checkpoint-sqlite · MLflow ≥3.16（本地 docker）· pytest。GEPA 与训练数据适配器在相应课程开发时接入，目前未安装或验证。

MIT
