# 历史快照：LangGraph L1–L6 学习路线

保存于 2026-09-13，保留课程重设计前的工作区内容（包括当时未提交的修改）。当前路线见 [curriculum.md](../curriculum.md)。以下原文包含已知待校正表述，不作为当前课程规范。

# LangGraph 学习路线图（中文 + English）

## 设计原则 / Design principles

- **核心优先**：先讲 LangGraph 的三个原语（State / Node / Edge），因为它们是一切能力的地基
- **渐进依赖**：后面的课程必须依赖前面的机制才能讲通（Interrupt 依赖 Checkpoint，Checkpoint 依赖 Edge 调度）
- **可跑 + 可验证**：每课都是确定性的独立代码，附完整测试，不依赖 API key
- **实战剖析**：每课代码都 print 完整运行时输出，配「改坏它」练习，亲手触发那个坑
- **官方文档互证**：每课 README 里给对应 docs.langchain.com 页面链接，代码与文档对照读

---

## 学习路径 / Course map

### L1 - State, Node, Edge：LangGraph 的三个原语

**学习重点：**

- State 是带 reducer 的 channel 集合
- Node 是 `state -> partial_update` 的纯函数
- Edge 决定下一个 super-step 并行跑谁
- **Super-step** 的执行循环：1. 找到本轮节点集；2. 并行执行；3. 等全部完成后对每个 channel 调用 reducer 合并写入；4. 写 checkpoint，回 1
- **Fan-out trigger rule**：节点在某 super-step 被调度 ⟺ 它的任意一个入边的源节点已在**本步或更早**跑过（OR 规则）

**踩坑验证：**

- 并发写无 reducer 的 channel → `InvalidUpdateError`
- 并发节点读到的 state 快照是**上一步**的，互相看不到
- 不对称 fan-out（一短一长两条路径）导致目标节点被触发两次
- 私有 schema 在 `stream(mode="values")` 时不会自动隐藏，需要显式 `output_keys`

**官方文档：**

- [Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api)
- [Use the Graph API](https://docs.langchain.com/oss/python/langgraph/use-graph-api)

---

### L2 - Control Flow: Conditional / Loop / Command / Send

**学习重点：**

- `add_conditional_edges`：路由表 vs 动态返回 node 名列表（fan-out 并行）
- 循环终止：条件边返回 `END` + `recursion_limit` / `GraphRecursionError`
- `Command(goto=..., update=...)` 取代"条件边 + 状态更新"两步
- `Send` 做 map-reduce（对列表里每个元素 fan-out 一个节点实例）

**踩坑验证：**

- 无终止条件的循环触发 recursion 限制
- 把同一段逻辑写成手写 for 循环 vs 图，对比可观测性

**官方文档：**

- [Create and control loops](https://docs.langchain.com/oss/python/langgraph/use-graph-api#create-and-control-loops)
- [Conditional branching](https://docs.langchain.com/oss/python/langgraph/use-graph-api#conditional-branching)

---

### L3 - Persistence: Checkpoint / Durable Execution

**学习重点：**

- `InMemorySaver` → `SqliteSaver` → `PostgresSaver` 的取舍
- `thread_id` 是恢复的钥匙
- `get_state` / `get_state_history` → **time travel**：回放、改状态、`update_state` 后从历史点分叉重跑
- durability 三档 `exit / async / sync` 与崩溃恢复的真实语义（`exit` 不能从进程崩溃中恢复）

**踩坑验证：**

- kill 掉进程后 resume，证明它真的没重跑已完成节点
- 用 `get_state_history` 回放每一步的 state，检查 super-step 边界

**官方文档：**

- [Persistence](https://docs.langchain.com/oss/python/langgraph/persistence)
- [Checkpointers](https://docs.langchain.com/oss/python/langgraph/checkpointers)

---

### L4 - Interrupt: Human-in-the-loop

**学习重点：**

- `interrupt()` 的**重放语义**：节点 resume 时从头重跑，所以 interrupt 之前不能有副作用
- "每节点只调一次 interrupt"规则 + 用条件边做重试循环（避免指数级重放）
- `Command(resume=...)`
- `interrupt_before/after` 静态断点与动态 `interrupt()` 的区别

**踩坑验证：**

- 审批流：工具调用前暂停 → 人工改参数 → 恢复，并检查 interrupt 之前的打印只出现一次

**官方文档：**

- [Interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts)
- [Human-in-the-loop](https://docs.langchain.com/oss/python/deepagents/human-in-the-loop)

---

### L5 - Subgraph: 组合与子图

**学习重点：**

- 共享 key → 直接把编译好的图 `add_node`
- 不共享 key → 在节点函数里做状态翻译
- `Command(graph=Command.PARENT)` 跨层跳转 + 父图必须给共享 key 定义 reducer
- 子图各自 checkpoint 命名空间导致"父图看不到子图中间态"的原因与解法（`Store`）

**踩坑验证：**

- 用子图搭一个 supervisor + worker，只保留父图一个 checkpointer

**官方文档：**

- [Use subgraphs](https://docs.langchain.com/oss/python/langgraph/use-subgraphs)
- [Subgraph persistence](https://docs.langchain.com/oss/python/langgraph/use-subgraphs#subgraph-persistence)

---

### L6 - Capstone: 综合架构

**学习重点：**

- 把 L2–L5 拼成一个"带审批、可断点续跑、可回滚"的完整 agent
- 原理专题：super-step 调度、reducer 合并时机、checkpoint 写入点、`Store` 与 checkpoint 的分工
- 故障注入测试

**官方文档：**

- [Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api) 总览

---

## 仓库结构 / Repo structure

```
docs/curriculum.md            # 总纲 + 官方文档阅读映射表
lessons/l1_state_node_edge/   # 每课: README.md + main.py + test_*.py
lessons/l2_control_flow/
lessons/l3_persistence/
lessons/l4_interrupts/
lessons/l5_subgraphs/
lessons/l6_capstone/
lib/helpers.py                # print_state / dump_checkpoints / 可视化辅助
```

**运行方式：**

```bash
# 跑某课的主程序（打印完整执行过程）
poetry run python lessons/l1_state_node_edge/main.py

# 跑测试（验证每个踩坑点）
poetry run pytest lessons/l1_state_node_edge -v
```

---

## 进阶提示 / Tips

- **先跑代码，再读 README**：代码里的 print 是运行时真相，README 是设计取舍
- **改坏它练习**：每课结束有个"改坏它"清单（改 reducer / 去掉终止条件 / 拔掉 checkpointer），亲手触发那个坑比读文档记得牢
- **对照官方文档**：每课 README 列了对应页链接，代码与文档互证，两边不一样时相信代码 + 提 issue

---

**当前进度：** L1 完成，L2-L6 待搭建

**贡献指南：** 每课遵循统一三件事：1. 可跑的最小代码；2. 断言其行为的 pytest；3. README 里的"为什么这样设计"与踩坑点。全部确定性、不需要 API key。
