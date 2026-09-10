# Agent Learning - LangGraph 课程项目

这是一个基于 LangGraph 官方文档的**渐进式学习课程**，目标是让工程师理解 LangGraph 的运行机制，而不只是会调 API。

## 项目结构

```
agent-learning/
├── docs/
│   ├── curriculum.md     # L1-L6 完整学习路线图
│   └── DESIGN.md         # 设计思路、代码组织、后续规划
├── lib/
│   └── helpers.py        # 可视化 helper（trace/print_state/print_history等）
├── lessons/
│   ├── l1_state_node_edge/   # ✅ 已完成：State/Node/Edge 三原语
│   ├── l2_control_flow/      # ⏳ 待搭建：Loop/Conditional/Command/Send
│   ├── l3_persistence/       # ⏳ 待搭建：Checkpoint/Time travel
│   ├── l4_interrupts/        # ⏳ 待搭建：Human-in-the-loop
│   ├── l5_subgraphs/         # ⏳ 待搭建：Subgraph 组合
│   └── l6_capstone/          # ⏳ 待搭建：综合实战
└── README.md            # 项目首页
```

## 快速开始

```bash
# 安装依赖
poetry install

# 运行 L1 课程
poetry run python lessons/l1_state_node_edge/main.py

# 运行测试
poetry run pytest lessons/l1_state_node_edge -v
```

## 设计原则

1. **核心原语优先**：State/Node/Edge 是地基，所有后续能力（Loop/Checkpoint/Interrupt/Subgraph）都只是这三样的组合
2. **渐进式依赖**：每课必须建立在前一课的机制上（Interrupt 依赖 Checkpoint，Checkpoint 依赖 Edge 调度）
3. **可跑可验证**：每课都是确定性代码 + pytest，打印完整运行时输出，无需 API key

## 技术文档索引

### LangGraph 核心文档（本课程依赖）

完整索引见 `llms.txt`，以下是课程直接相关的页面：

#### L1 - State/Node/Edge
- [Graph API overview](https://docs.langchain.com/oss/python/langgraph/graph-api.md) - 图 API 总览
- [Use the graph API](https://docs.langchain.com/oss/python/langgraph/use-graph-api.md) - 图 API 使用指南
- [Choosing between Graph and Functional APIs](https://docs.langchain.com/oss/python/langgraph/choosing-apis.md) - API 选择
- [INVALID_CONCURRENT_GRAPH_UPDATE](https://docs.langchain.com/oss/python/langgraph/errors/INVALID_CONCURRENT_GRAPH_UPDATE.md) - 并发写入错误

#### L2 - Control Flow
- [Use the graph API](https://docs.langchain.com/oss/python/langgraph/use-graph-api.md) - 包含 loop/conditional 示例
- [GRAPH_RECURSION_LIMIT](https://docs.langchain.com/oss/python/langgraph/errors/GRAPH_RECURSION_LIMIT.md) - 递归限制错误

#### L3 - Persistence
- [Persistence](https://docs.langchain.com/oss/python/langgraph/persistence.md) - 持久化机制
- [Checkpointers](https://docs.langchain.com/oss/python/langgraph/checkpointers.md) - Checkpointer 实现
- [Use time-travel](https://docs.langchain.com/oss/python/langgraph/use-time-travel.md) - 时间旅行 API
- [Memory](https://docs.langchain.com/oss/python/langgraph/add-memory.md) - 内存管理
- [MISSING_CHECKPOINTER](https://docs.langchain.com/oss/python/langgraph/errors/MISSING_CHECKPOINTER.md) - 缺少 checkpointer 错误

#### L4 - Interrupts
- [Interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts.md) - 中断机制
- [MISSING_CHECKPOINTER](https://docs.langchain.com/oss/python/langgraph/errors/MISSING_CHECKPOINTER.md) - Interrupt 需要 checkpointer

#### L5 - Subgraphs
- [Subgraphs](https://docs.langchain.com/oss/python/langgraph/use-subgraphs.md) - 子图使用
- [MULTIPLE_SUBGRAPHS](https://docs.langchain.com/oss/python/langgraph/errors/MULTIPLE_SUBGRAPHS.md) - 多子图错误

#### L6 - Capstone
- [Application structure](https://docs.langchain.com/oss/python/langgraph/application-structure.md) - 应用结构
- [Fault tolerance](https://docs.langchain.com/oss/python/langgraph/fault-tolerance.md) - 容错机制
- [Streaming](https://docs.langchain.com/oss/python/langgraph/streaming.md) - 流式输出
- [Event streaming](https://docs.langchain.com/oss/python/langgraph/event-streaming.md) - 事件流

### 其他相关文档

- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview.md) - 总览
- [Quickstart](https://docs.langchain.com/oss/python/langgraph/quickstart.md) - 快速开始
- [Thinking in LangGraph](https://docs.langchain.com/oss/python/langgraph/thinking-in-langgraph.md) - 思维模式
- [LangGraph runtime](https://docs.langchain.com/oss/python/langgraph/pregel.md) - 运行时机制（Pregel 模型）
- [Build a custom RAG agent](https://docs.langchain.com/oss/python/langgraph/agentic-rag.md) - RAG agent 示例
- [Build a custom SQL agent](https://docs.langchain.com/oss/python/langgraph/sql-agent.md) - SQL agent 示例

## 开发指南

### 添加新课程

每课遵循统一三件套：

1. **README.md**: 中文教学文档
   - 为什么这三个词放在这一课
   - 核心概念表格（原语/机制/一句话职责）
   - 执行模型
   - 本课代码要验证的行为
   - 自己动手（改坏它）练习
   - 官方文档对应页

2. **main.py**: 可跑的最小代码
   - 打印完整运行时输出（用 `lib.trace()`）
   - 每个 demo 独立可跑
   - 所有节点打印自己看到的 state

3. **test_main.py**: pytest 测试
   - 每个踩坑点一个测试
   - 用 `assert` 或 `pytest.raises` 断言
   - 测试名以 `test_` 开头 + docstring

详见 `docs/DESIGN.md`。

### 运行测试

```bash
# 单课测试
poetry run pytest lessons/l1_state_node_edge -v

# 全部测试
poetry run pytest lessons/ -v

# 覆盖率
poetry run pytest lessons/ --cov=lessons --cov-report=html
```

### 更新文档

- 新增课程时更新 `docs/curriculum.md` 对应章节
- 新增 helper 时更新 `lib/helpers.py` docstring
- 发现官方文档错误时在 README 里注明

## 技术栈

- Python ≥3.12
- LangGraph ≥1.2.11
- LangGraph-checkpoint-sqlite (for L3)
- pytest + pytest-asyncio

## 当前进度

- ✅ **L1 (State/Node/Edge)**: 完成（358行代码 + 166行测试 + 100行文档）
- ⏳ **L2 (Control flow)**: 待搭建
- ⏳ **L3 (Persistence)**: 待搭建
- ⏳ **L4 (Interrupt)**: 待搭建
- ⏳ **L5 (Subgraph)**: 待搭建
- ⏳ **L6 (Capstone)**: 待搭建

## 许可证

MIT
