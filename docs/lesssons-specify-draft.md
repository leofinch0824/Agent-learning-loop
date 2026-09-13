# 工业级 Agent 架构师与生产落地实战课程大纲（8周速成版）

> 调研草稿保留说明（2026-09-13）：以下为原背景研究与候选主题，引用编号和部分技术结论尚待校正。结合学习者目标形成的当前路线见 [curriculum.md](curriculum.md)，目录与具体取舍见 [course-implementation.md](course-implementation.md)；原调研正文保留供后续按需查证。

## 一、 课程设计理念与架构体系

本课程面向**实际生产环境落地**与**中高级架构面试**，从单纯的“LangGraph 状态机通识”升级为涵盖 **ETCLOVG（执行、工具、上下文、编排、可观测、验证、治理）** 的完整 Agent Harness（工程套件）架构体系[cite: 2, 5]。

### 核心架构映射（ETCLOVG 体系）

- **E (Execution 环境层)**：Docker/沙箱隔离执行与环境安全[cite: 5]。
- **T (Tooling 工具层)**：流式并发调度（Streaming Tool Executor）、Fail-closed 安全基线与 MCP (Model Context Protocol) 协议[cite: 5, 6, 8]。
- **C (Context 上下文层)**：四级水位线上下文压缩、Prompt Cache 命中率防护与 Store 跨会话长期记忆[cite: 5, 7]。
- **L (Lifecycle 编排层)**：Pregel BSP 超步调度内核、Command/Send 原子控制流、子图与 Fork 隔离范式[cite: 1, 3, 6]。
- **O (Observability 可观测层)**：LangSmith 树级调用链（RunTree/Span）、延迟火焰图与 Token 成本归因（对标 MLflow GenAI Tracing）[cite: 2, 3, 11]。
- **V (Verification 评测层)**：自动化 Trajectory 轨迹评测、变异击杀率（Mutation Testing）与回归测试集闭环[cite: 5, 7]。
- **G (Governance 治理层)**：多层权限网关、Human-in-the-loop (HITL) 动态审批与生产死循环熔断哨兵[cite: 2, 5, 6]。

---

## 二、 8 周渐进式学习路线与实战任务

### 阶段一：核心原语、运行时机制与工程基座（Weeks 1-3）

#### 🧩 Week 1: LangGraph 状态机与 Pregel 调度内核

- **核心理论与运行时机制**：
  - 基于 Google Pregel 与 Apache Beam 的 BSP（Bulk Synchronous Parallel）超步调度模型：`Plan → Parallel Execute → Barrier & Reduce → Checkpoint`[cite: 1, 3, 5]。
  - Channel 与 Reducer 冲突消解机制：无 Reducer 的覆盖写限制（Last-Write Wins）vs 挂载 Reducer（如 `operator.add`）的增量合并语义[cite: 1]。
  - Fan-out OR 规则调度本质与状态快照读取的单步物理隔离性[cite: 1]。
- **工程代码与“改坏它”清单 (`lessons/w1_state_primitives/`)**[cite: 1]：
  1. **并发写爆破**：两节点同超步并发写入无 Reducer 通道，断言捕获 `InvalidUpdateError`[cite: 1]。
  2. **快照脏读验证**：并发节点分别打印 state，断言其只能读到上一超步快照，彼此写入不可见[cite: 1]。
  3. **非对称路径双发陷阱**：构造一长一短汇聚路径，断言汇聚节点被异常触发两次，并用条件边及同步屏障状态修复[cite: 1]。
- **面试考核点**：
  - _“详细说明 Pregel 超步中并发节点的执行与 Channel 合并时机；为什么并发节点读到的不是最新写入？”_[cite: 1]

#### 🛠️ Week 2: 动态控制流、Map-Reduce 扇出与真实 Tool-Calling 闭环

- **核心理论与架构演进**：
  - 传统条件路由（`add_conditional_edges`）与原子跳转原语 `Command(goto, update)` 的执行开销与性能差异[cite: 1]。
  - 动态拓扑扇出：利用 `Send` 原语在运行时动态分发任务列表，结合 Reducer 实现 Map-Reduce 汇聚[cite: 1]。
  - 真实 ChatModel 接入与 ReAct 内部循环（Agent Node ↔ Tool Node）[cite: 2, 5]。
  - 工业级流式工具并发调度（Streaming Tool Executor）：基于 `isConcurrencySafe` 属性实现只读工具（Read/Grep）并行执行，写操作（Edit）串行排队[cite: 6, 7]。
  - 防御性 Harness 设计：结构化 Tool Call 参数解析失败回传重试、`recursion_limit` 熔断机制[cite: 1, 2]。
- **工程代码与“改坏它”清单 (`lessons/w2_control_flow_and_tools/`)**[cite: 1, 2]：
  1. **递归死循环熔断**：故意移除终止条件，验证 `recursion_limit` 生效并捕获 `GraphRecursionError`[cite: 1]。
  2. **畸形 JSON 容错**：Mock LLM 吐出错误参数，验证 Tool Node 捕获异常并包装为 `ToolMessage(status="error")` 诱导模型纠错重试[cite: 2]。
  3. **Send 动态扇出**：使用 `Send` 动态分派多个子任务，主图 Reducer 完整聚合所有产物[cite: 1]。
- **面试考核点**：
  - _“如何利用 Command 原语优化图调度？当 LLM 出现工具参数幻觉或死循环时，系统如何进行图级与业务级熔断？”_[cite: 1, 2]

#### 💾 Week 3: 生产级持久化与 LangSmith 树级可观测性

- **核心理论与架构演进**：
  - Checkpointer 选型与恢复钥匙：`InMemory` vs `Sqlite` vs `PostgresSaver`（连接池与 ACID 事务）[cite: 1, 3]。
  - Durability 写入策略：`exit` / `async` / `sync` 对崩溃恢复的语义差异[cite: 1]。
  - Time-Travel 机制：`get_state_history` 与 `update_state(as_node=...)` 历史状态分叉（Fork）原理[cite: 1, 3]。
  - 可观测性（O层）：配置 `LANGSMITH_TRACING=true`，分析 RunTree、Span 树状依赖、延迟火焰图与 Token 消耗归因（对标 MLflow OpenTelemetry Trace）[cite: 2, 3, 11]。
- **工程代码与“改坏它”清单 (`lessons/w3_persistence_and_obs/`)**[cite: 1, 2]：
  1. **暴力 Kill 进程容灾**：长耗时节点注入 `os._exit(1)`，重启后传入相同 `thread_id` 验证断点续跑，断言已完成节点绝不重复执行[cite: 1]。
  2. **Time-Travel 分叉回放**：提取历史 Checkpoint 并改写状态，生成两条分叉的对话分支[cite: 1]。
  3. **LangSmith 监控实测**：注入非致命错误并在 LangSmith 界面定位异常 Span 元数据与耗时瓶颈[cite: 2, 3]。
- **面试考核点**：
  - _“生产环境为什么首选 PostgresSaver？进程突然被 OOMKilled 时，不同 durability 策略会导致什么后果？如何排查 Token 异常激增的节点？”_[cite: 1, 2]

---

### 阶段二：上下文工程、治理与生产工具系统（Weeks 4-5）

#### 🧠 Week 4: 上下文工程（Context Engineering）与 Token 经济学

- **核心理论与架构演进**：
  - 上下文衰减（Context Rot）与四级水位线压缩（MUR AI / Claude Code 实践）：
    - Tier 0: 静默通行[cite: 9]。
    - Tier 1 (Snip): 本地截断超长工具输出[cite: 6, 9]。
    - Tier 2 (Prune): 替换历史工具结果为占位符（`[Old tool result content cleared]`）[cite: 6, 9]。
    - Tier 3 (Summarize): 增量结构化摘要兜底（保护区与用户原文绝对不动）[cite: 9]。
  - **Prompt Cache 单调推进原则**：深入理解“滑窗淘汰旧结果导致每个 Super-step 缓存前缀失效、费用暴涨 12 倍”的致命陷阱[cite: 9]。
  - 记忆体系划分：`MessagesState`（短期工作记忆/RAM）与 `Store`（跨 Session 长期记忆/磁盘）[cite: 1, 3, 9]。
- **工程代码与“改坏它”清单 (`lessons/w4_context_engineering/`)**：
  1. **滑窗击穿缓存复现**：对比滑窗修剪与单调修剪的 Prompt 字节哈希值，证明滑窗导致前缀缓存完全失效[cite: 9]。
  2. **四级水位线触发器**：用真实 Token Usage 驱动分层压缩，断言用户原始 Prompt 与近端保护区不被篡改[cite: 9]。
  3. **跨会话 Store 读写**：利用 `Store` 保存用户画像，在新 `thread_id` 会话中检索并注入上下文[cite: 1, 3]。
- **面试考核点**：
  - _“什么是 Context Rot？为什么滑窗修剪工具历史会导致 Prompt Cache 命中率跌零？Store 与 Checkpointer 在架构上有何本质区别？”_[cite: 1, 3, 9]

#### 🛡️ Week 5: 工具系统演进、MCP 协议与 Fail-closed 生产治理

- **核心理论与架构演进**：
  - 工具生命周期管理与 **Fail-closed（安全默认）** 基线：默认非并发安全（强制串行）、默认非只读（强制走权限网关）[cite: 6, 7]。
  - 多层权限模型与沙箱控制：只读放行、高危写操作阻断与人工确认[cite: 6, 7]。
  - 外部协议桥接：接入 **MCP (Model Context Protocol)** Client，动态挂载标准工具池[cite: 2, 5, 8]。
  - Human-in-the-loop (HITL)：深入 LangGraph `interrupt()` 的重放语义（节点从头重跑），实现高危命令审批与外部副作用幂等设计[cite: 1, 3]。
- **工程代码与“改坏它”清单 (`lessons/w5_tooling_and_governance/`)**：
  1. **`interrupt()` 重复副作用复现**：在 `interrupt()` 前执行写文件操作，恢复后断言写操作被重复执行，设计状态标记实现幂等[cite: 1]。
  2. **只读/写排队调度器**：构造多工具并发调用场景，验证只读工具并行运行，写文件工具安全等待[cite: 6, 7]。
  3. **MCP 节点桥接**：运行本地 MCP 服务，通过 LangGraph Node 动态调用 MCP 暴露的工具[cite: 2, 5]。
- **面试考核点**：
  - _“LangGraph 的 interrupt() 在恢复时有什么重放特性？如何避免审批后的外部副作用重复执行？Fail-closed 在工具安全中如何落地？”_[cite: 1, 6]

---

### 阶段三：复合架构、评测闭环与面试冲刺（Weeks 6-8）

#### 🏢 Week 6: 复合子图编排、Planning 范式与上下文隔离

- **核心理论与架构演进**：
  - 子图（Subgraph）组合范式：共享 Key 编译图 vs 状态翻译节点[cite: 1, 3]。
  - 跨层路由：`Command(graph=Command.PARENT)` 跨层跳转[cite: 1]。
  - Planning 机制：Plan-and-Execute（TodoWrite）模式，解耦规划器与执行器，使任务完成率成倍提升[cite: 5, 6]。
  - **Sub-agent 上下文隔离（Fork 模式）**：子代理必须分配独立 `messages[]`，防止探索性工具输出污染父图上下文（Context Pollution）[cite: 6, 7, 8]。
- **工程代码与“改坏它”清单 (`lessons/w6_subgraphs_and_planning/`)**[cite: 1]：
  1. **上下文污染对比**：对比“共享全局上下文”与“Fork 独立上下文”，观察父图 Token 膨胀与指令漂移差异[cite: 6, 7, 8]。
  2. **Supervisor 模式子图集群**：搭建 Supervisor 主控图与两个垂直领域子图（如代码审查与单元测试）[cite: 1, 2]。
  3. **子图 Checkpoint 命名空间调试**：使用 `checkpoint_ns` 隔离并回溯子图状态[cite: 1, 3]。
- **面试考核点**：
  - _“多 Agent 协作系统设计中，为什么不能把所有交互记录塞进同一个全局 Context？子图如何实现独立的 Checkpoint 隔离？”_[cite: 2, 6]

#### 🎯 Week 7: 自动化评测闭环与生产防御哨兵

- **核心理论与架构演进**：
  - 评测体系架构：题集是本体，主指标采用变异击杀率（Mutation Score）与 Fail-to-Pass 机制，避免无效的纯自然语言评分[cite: 5, 7]。
  - LangSmith 评测工程：离线抽取生产 Trace 沉淀为 Golden Dataset，编写自定义 Evaluator 运行 Trajectory 轨迹级打分[cite: 2, 7, 10]。
  - LLM-as-a-judge 校准：评分标准对齐、Few-shot 校准与一致性评估[cite: 2, 7]。
  - 生产防御哨兵（Sentinel）：连续重复工具调用拦截器、死循环空转检测熔断器[cite: 2, 7]。
- **工程代码与“改坏它”清单 (`lessons/w7_eval_and_sentinel/`)**：
  1. **Golden Dataset 评测流水线**：基于 Pytest + LangSmith 编写轨迹评测脚本，评估 Tool Call 准确率[cite: 2, 7, 10]。
  2. **生产死循环注入**：模拟 LLM 重复发出相同 Tool Call，触发图级 Sentinel 熔断器并记录告警[cite: 2, 7]。
- **面试考核点**：
  - _“Prompt 或模型小版本迭代时，如何用 LangSmith 保证原有 Bad Case 不倒退？如何度量 Agent 轨迹的执行质量？”_[cite: 2, 7, 10]

#### 🚀 Week 8: 工业级 Capstone 端到端实战与大厂面试冲刺

- **综合 Capstone 实战**：
  - 构建**工业级全生命周期研发/运维 Agent**：
    - 具备 Planning 任务拆解与看板推进[cite: 5, 6]。
    - 采用 Fork Sub-agent 实现上下文隔离[cite: 6, 7, 8]。
    - 挂载四级水位线压缩与 Prompt Cache 前缀优化[cite: 7, 9]。
    - 接入 PostgresSaver 持久化与 HITL 高危命令审批[cite: 1, 3]。
    - 全链路接入 LangSmith 链路追踪与离线评测[cite: 2, 3, 10]。
- **面试系统设计与复盘模拟**：
  - 高并发 Agent 架构设计（分布式长会话连接管理、任务异步化与 Worker 架构）[cite: 2, 5]。
  - 故障复盘答辩演练（如何向面试官复盘“上下文击穿导致的空转”或“并发调度冲突”）[cite: 1, 7, 9]。
  - 简历项目亮点话术沉淀与八股强化。

---

## 三、 工程代码仓规范与组织结构

```text
agent-harness-lab/
├── docker-compose.yml             # 本地 PostgreSQL (测试 PostgresSaver)
├── pyproject.toml                 # Poetry/uv 依赖配置
├── docs/
│   └── curriculum.md              # 课程总纲与 ETCLOVG 架构映射
├── lessons/
│   ├── w1_state_primitives/       # Week 1: 状态机原语与并发冲突
│   ├── w2_control_flow_and_tools/ # Week 2: 控制流、Command、Send 与工具流式调度
│   ├── w3_persistence_and_obs/    # Week 3: PostgresSaver、Time-Travel 与 LangSmith
│   ├── w4_context_engineering/    # Week 4: 四级水位线压缩、Prompt Cache 与 Store
│   ├── w5_tooling_and_governance/ # Week 5: MCP 客户端、Fail-closed 权限与 interrupt 幂等
│   ├── w6_subgraphs_and_planning/ # Week 6: Subgraph、Planning 与 Fork 上下文隔离
│   ├── w7_eval_and_sentinel/      # Week 7: LangSmith Golden Dataset 评测与防空转哨兵
│   └── w8_capstone/               # Week 8: 工业级端到端 Agent 完整工程
└── lib/
    ├── helpers.py                 # 状态快照 Dump、可视化拓扑输出工具
    └── db.py                      # PostgresSaver 连接池封装
```

## 参考来源

- **[source: 2] 参考文献列表（《Self-Improving Agents in the Era of Experience: A Survey of Self- to Meta-Evolution》）**

- **内容**：汇集了智能体技能进化、环境基准与自演进 Agent 领域的相关学术论文与技术报告参考条目。

- **主要链接**：
- Evoskill 论文：[https://arxiv.org/abs/2603.02766](https://arxiv.org/abs/2603.02766)

- MANTRA 论文：[https://arxiv.org/abs/2605.06334](https://arxiv.org/abs/2605.06334)

- Claude 3.7 Sonnet & Claude Code 发布公告：[https://www.anthropic.com/news/claude-3-7-sonnet](https://www.anthropic.com/news/claude-3-7-sonnet)

- LangGraph 官方博客发布通告：[https://www.langchain.com/blog/langgraph](https://www.langchain.com/blog/langgraph)

- Model Context Protocol 规范：[https://github.com/modelcontextprotocol/modelcontextprotocol](https://github.com/modelcontextprotocol/modelcontextprotocol)

- ReAct 论文：[https://arxiv.org/abs/2210.03629](https://arxiv.org/abs/2210.03629)

- **[source: 3] 官方文档《Introduction to DeepEval | DeepEval - The LLM Evaluation Framework》**
- **内容**：开源大模型评测框架说明，涵盖基于 Pytest 的断言测试、50+ 内置指标、轨迹级（Trajectory-based）与组件级评测模式。

- **主要链接**：[https://deepeval.com/docs/introduction](https://deepeval.com/docs/introduction)

- **[source: 4] Anthropic 工程博客《Demystifying evals for AI agents》**
- **内容**：Anthropic 官方对 Agent 评测体系的系统拆解，涵盖自动化评测、多轮交互评估、离线测试与行业评测工具对比。

- **主要链接**：[https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)

- **[source: 5] LangChain 官方文档《LangSmith Evaluation - Docs by LangChain》**
- **内容**：LangSmith 评测模块使用手册，包含测试集构建、离线/在线实验对比、LLM-as-a-judge 与质量回归闭环。

- **主要链接**：[https://docs.langchain.com/langsmith/evaluation](https://docs.langchain.com/langsmith/evaluation)

- **[source: 6] LangChain 官方文档《LangGraph overview - Docs by LangChain》**
- **内容**：LangGraph 核心编排运行时概览，解析其受 Pregel 和 Apache Beam 启发的调度原理、StateGraph 状态图与生态产品层级。

- **主要链接**：[https://docs.langchain.com/oss/python/langgraph/overview](https://docs.langchain.com/oss/python/langgraph/overview)

- **[source: 7] MLflow 官方平台文档《MLflow for Agents and LLMs | MLflow AI Platform》**
- **内容**：MLflow 的 GenAI 平台能力总览，介绍兼容 OpenTelemetry 的 Tracing 链路追踪、评估监控以及 Prompt 管理机制。

- **主要链接**：[https://mlflow.org/docs/latest/genai/](https://mlflow.org/docs/latest/genai/)

- **[source: 8] 学术综述论文《Agent Harness Engineering: A Survey》（Junjie Li 等）**
- **内容**：系统提出 ETCLOVG 七层系统工程框架，汇总分析 170+ 开源 Harness 项目、工程设计原则与可观测性评测脱节等问题。

- **主要链接**：论文项目主页为 Awesome-Agent-Harness，涵盖各层开源项目与工程报告文献。

- **[source: 11] 技术实录手记《上下文工程与分级压缩实践》**
- **内容**：深入分析了 60%/80%/95% 四级水位线阈值、先 Snip 截断后 Prune 擦除原则、增量摘要设计与使用真实 Token usage 的考量。

- **主要链接**：[横向拆解Claude Code、Codex等六大Agent上下文压缩策略后，我们做了第 7 个](./references/横向拆解Claude%20Code、Codex等六大Agent上下文压缩策略后，我们做了第%207%20个.md)

- **[source: 12] 架构审计手记《Agent 评测方法论审计与一致性复盘》**
- **内容**：复盘了因缺乏确定性单测导致主指标过度依赖 LLM 判断，进而出现评分一致性极低（Cohen's Kappa 仅 0.10-0.21）的教训。

- **主要链接**：[两万字长文｜手把手带你趟过 AI Coding 深水区：编码让位，人退到哪里](./references/两万字长文_手把手带你趟过_AI_Coding_深水区_编码让位_人退到哪里.md)

- **[source: 13] 架构总结手记《Claude Code 架构分析与工程设计模式总结》**
- **内容**：总结了 Claude Code 未上线功能（如 Voice Mode、WebBrowserTool）以及 10 大核心工程模式（如 AsyncGenerator Streaming、StreamingToolExecutor、Context Isolation 等）。

- **主要链接**：[逆向深扒Claude%20Code源码，我发现了什么](./references/逆向深扒Claude%20Code源码，我发现了什么.md)
