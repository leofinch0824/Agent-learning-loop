# 项目设计文档 / Project Design Document

## 1. 设计目标 / Design goals

这个课程的目标是让工程师**理解 LangGraph 的运行机制**，而不是只会调 API。三个设计原则贯穿始终：

### 1.1 核心原语优先 (Core primitives first)

LangGraph 的所有能力都建立在三个原语之上：
- **State**: 一组带 reducer 的 channel，定义"节点之间能传什么、并发写怎么合并"
- **Node**: `state -> partial_update` 的纯函数，读全部状态，只返回想改的 key
- **Edge**: 静态或动态的连线，决定下一个 super-step 并行跑谁

后续的所有能力（循环、检查点、中断、子图）都只是这三个原语的组合变形。所以 L1 必须把这三个概念讲透，包括：
- Super-step 的执行循环（调度 → 并行跑 → join → reducer 合并 → checkpoint）
- Trigger rule（OR 规则：节点被调度 ⟺ 任一入边的源已跑过）
- 并发写入的合并机制（有 reducer 就合并，无 reducer 就报错）

### 1.2 渐进式依赖 (Progressive dependency)

每一课的机制必须建立在前一课的地基上，形成严格的依赖链：

```
L1 (State/Node/Edge)
  ↓ 理解了 Edge 的调度规则，才能理解循环何时终止
L2 (Control flow: loop/conditional/Command/Send)
  ↓ 理解了 super-step 的执行边界，才能理解 checkpoint 写在哪
L3 (Persistence: checkpointer/durability/time travel)
  ↓ 理解了 checkpoint 的恢复机制，才能理解 interrupt 从何处 resume
L4 (Interrupt: human-in-the-loop/replay semantics)
  ↓ 理解了单图的 checkpoint 命名空间，才能理解子图的隔离
L5 (Subgraph: shared/separate state/Command.PARENT)
  ↓ 理解了所有机制，才能组合成完整 agent
L6 (Capstone: 综合实战)
```

这个依赖链的好处是：如果你理解了 L1-L5，L6 就是"把前面学的拼起来"，没有新概念。

### 1.3 可跑可验证 (Verify, don't trust)

**不要相信文档，相信代码。** 每一课都包含：
1. **可跑的最小代码** (`main.py`)，打印完整的运行时输出（不是摘要，是原始 chunk）
2. **断言行为的测试** (`test_main.py`)，用 pytest 证明"并发写无 reducer 确实会报错"
3. **README**，解释"为什么这样设计"、踩坑点、以及对应的官方文档链接

所有课程都是**确定性的**，不依赖 API key，不依赖网络，不依赖随机数。你跑两遍，输出完全一样。

---

## 2. 课程架构 / Curriculum architecture

### 2.1 依赖倒置的排序

课程顺序按"后面的概念必须用前面的机制才能讲通"排列：

| 课程 | 核心机制 | 为什么在这个位置 |
|------|---------|-----------------|
| **L1** | State / Node / Edge | 地基。所有后续能力都是这三样的组合。 |
| **L2** | Loop / Conditional | 必须先理解 Edge 的调度规则，才能理解循环何时终止、条件边如何路由。 |
| **L3** | Checkpoint | 必须先理解 super-step 的边界，才能理解 checkpoint 写在哪、什么时候恢复。 |
| **L4** | Interrupt | 必须先理解 checkpoint 的恢复机制，才能理解 interrupt 从何处 resume、为什么重放。 |
| **L5** | Subgraph | 必须先理解单图的 checkpoint 命名空间，才能理解子图的隔离与父子图通信。 |
| **L6** | Capstone | 综合前面所有机制，搭一个"带审批、可断点续跑、可回滚"的完整 agent。 |

### 2.2 每课的三件套

每一课遵循统一的三件套模式：

```
lessons/lN_topic/
├── README.md          # 中文教学文档，讲清"为什么"和"坑在哪"
├── main.py            # 可跑的最小代码，打印完整执行过程
└── test_main.py       # pytest 测试，断言所有踩坑行为
```

#### README.md 的结构
1. **为什么这三个词放在这一课**：讲清这课在整体架构中的位置
2. **核心概念表格**：原语/机制/一句话职责
3. **执行模型**：这课的机制如何工作（配图或伪代码）
4. **N 种写法对比**：如果有多种实现方式，列表格对比优劣
5. **本课代码要验证的行为**：列出所有 demo 的验证目标
6. **自己动手（改坏它）**：3-5 个"把代码改坏"的练习
7. **官方文档对应页**：带链接的清单

#### main.py 的结构
```python
"""L1 - State / Node / Edge.

Run it:
    poetry run python lessons/l1_state_node_edge/main.py
"""

# 1. 原语定义（State schema / Node 函数）
# 2. Graph builder 函数（每个 demo 一个）
# 3. Demo 函数（调 trace() 打印完整输出）
# 4. if __name__ == "__main__": 按序跑所有 demo
```

关键设计：
- **所有节点内部都 `print()` 自己看到的 state**，证明"同一 super-step 的节点看不到对方的写"
- **用 `lib.trace()` 包裹执行**，打印每个 super-step 的 `updates` 和 `values`
- **不做摘要**：输出就是原始的 `stream()` chunk，让读者看到运行时真相

#### test_main.py 的结构
```python
"""Tests for L1 - State, Node, Edge."""

def test_reducer_merges_concurrent_writes():
    """两个节点并发写 reducer channel -> 合并成功"""
    ...

def test_concurrent_writes_without_reducer():
    """两个节点并发写无 reducer channel -> InvalidUpdateError"""
    with pytest.raises(InvalidUpdateError, match="..."):
        ...
```

每个测试对应 README 里的一个踩坑点，用 `assert` 或 `pytest.raises` 断言。

---

## 3. 代码组织 / Code organization

### 3.1 目录结构

```
agent-learning/
├── docs/
│   ├── curriculum.md     # 完整学习路线图（L1-L6）
│   └── DESIGN.md         # 本文档：设计思路与规划
├── lib/
│   ├── __init__.py       # 导出所有 helper
│   └── helpers.py        # trace/print_state/print_history/build_checkpointer...
├── lessons/
│   ├── l1_state_node_edge/
│   │   ├── README.md
│   │   ├── main.py
│   │   └── test_main.py
│   ├── l2_control_flow/     # 待搭建
│   ├── l3_persistence/      # 待搭建
│   ├── l4_interrupts/       # 待搭建
│   ├── l5_subgraphs/        # 待搭建
│   └── l6_capstone/         # 待搭建
├── pyproject.toml
├── README.md            # 项目首页
└── .gitignore
```

### 3.2 lib/helpers.py 的设计

这个文件存在的唯一目的是**让运行时行为可见**。所有 helper 都遵循"打印，不隐藏"的原则：

| Helper | 职责 | 为什么需要它 |
|--------|------|------------|
| `trace(graph, input, ...)` | 跑图并打印每个 chunk | 让学员看到 `stream()` 的原始输出，而不是摘要 |
| `print_state(graph, config)` | 打印当前 checkpoint 的 state + metadata | L3 用：证明 `get_state()` 返回什么 |
| `print_history(graph, config)` | 打印 checkpoint 历史 | L3 用：证明 time travel 的每一步 |
| `build_checkpointer(kind, path)` | 构造 memory/sqlite checkpointer | L3 用：让学员切换持久化后端 |
| `thread(thread_id, **extra)` | 构造 `RunnableConfig` | L3 用：简化 `{"configurable": {"thread_id": ...}}` 的写法 |
| `show_graph(graph)` | 打印 Mermaid 源码 | 可视化图结构（贴到 mermaid.live） |

设计原则：**如果一个 helper 隐藏了什么，它就打印它**。例如 `trace()` 包裹了 `graph.stream()`，但它会把每个 chunk 原样打印出来。

### 3.3 测试策略

所有测试都是**黑盒测试 + 白盒断言**的混合：
- **黑盒**：调 `graph.invoke()` 或 `graph.stream()`，不 mock 内部
- **白盒断言**：检查 state 里的具体值、checkpoint 历史的 step 数、异常类型

例如：
```python
def test_asymmetric_fanout_runs_target_twice():
    """一短一长两条路径 -> 目标节点触发两次"""
    graph = build_fanout_continuation_graph()
    result = graph.invoke({"filename": ".env", "findings": [], "verdict": "pass"})
    # 断言：`extra_step` 只在长路径上跑，但 `reduce` 看到了它的输出
    assert "extra: security branch needed a follow-up" in result["findings"]
```

不测试的东西：
- LangGraph 内部实现细节（不 mock `_run_step` 之类的私有方法）
- 性能（这是教学课程，不是 benchmark）
- 边缘 case（只测核心机制 + 常见坑）

---

## 4. 后续课程规划 / Roadmap for L2-L6

### L2 - Control Flow

**核心概念：**
- `add_conditional_edges`：路由表写法 vs 返回 node 名列表（fan-out 并行）
- 循环终止：条件边返回 `END` + `recursion_limit` / `GraphRecursionError`
- `Command(goto=..., update=...)` 取代"条件边 + 状态更新"两步
- `Send` 做 map-reduce（对列表里每个元素 fan-out 一个节点实例）

**Demo：**
1. 手写 for 循环 vs 图循环，对比可观测性
2. 用条件边 + `recursion_limit` 实现一个 ReAct 骨架（a=模型，b=工具，用假模型）
3. `Command` 简化"决策 + 路由"
4. `Send` 做 parallel map（对列表里每个元素调一次同一个节点）

**测试：**
- 无终止条件的循环触发 `GraphRecursionError`
- 条件边正确路由到不同分支
- `Send` 的 fan-out 数量 == 输入列表长度

### L3 - Persistence

**核心概念：**
- `InMemorySaver` vs `SqliteSaver` vs `PostgresSaver` 的取舍
- `thread_id` 是恢复的钥匙
- `get_state` / `get_state_history` 的 API
- **Time travel**: `update_state()` 后从历史点分叉重跑
- Durability 三档 `exit / async / sync` 与崩溃恢复的真实语义

**Demo：**
1. 用 `InMemorySaver` 跑完，打印 `get_state_history()` 的每一步
2. 换成 `SqliteSaver`，kill 掉进程，重启后 `resume`
3. `update_state()` 改某个历史 checkpoint，证明它分叉出新分支
4. Durability `exit` mode：进程崩溃时丢失中间步

**测试：**
- `thread_id` 隔离：两个 thread 的 state 不互相影响
- `get_state_history()` 的 step 顺序正确
- `update_state()` 后的分叉分支独立

### L4 - Interrupt

**核心概念：**
- `interrupt()` 的**重放语义**：节点 resume 时从头重跑
- "每节点只调一次 interrupt"规则 + 用条件边做重试循环
- `Command(resume=...)` 传递人类输入
- `interrupt_before/after` 静态断点 vs 动态 `interrupt()` 的区别

**Demo：**
1. 审批流：工具调用前暂停 → 人工改参数 → 恢复
2. 表单收集：循环 prompt 直到输入合法
3. 证明 interrupt 之前的代码只执行一次（用计数器）

**测试：**
- `interrupt()` 触发后 `invoke()` 返回 `__interrupt__` 标记
- `Command(resume=...)` 正确传递值
- 条件边重试循环不会指数级重放

### L5 - Subgraph

**核心概念：**
- 共享 key → 直接把编译好的图 `add_node`
- 不共享 key → 在节点函数里做状态翻译
- `Command(graph=Command.PARENT)` 跨层跳转 + 父图必须给共享 key 定义 reducer
- 子图各自 checkpoint 命名空间导致"父图看不到子图中间态"

**Demo：**
1. Supervisor + 2 workers（共享 `messages` key）
2. 不共享 key 的子图（在节点函数里翻译状态）
3. `Command.PARENT` 从子图跳到父图的兄弟节点

**测试：**
- 子图的 checkpoint 与父图隔离
- 共享 key 的写入被父图看到
- `Command.PARENT` 正确跳转

### L6 - Capstone

**目标：** 把 L1-L5 拼成一个"带审批、可断点续跑、可回滚"的完整 agent。

**场景：** 一个代码审查 agent，包含：
1. 风格检查 + 安全检查（L1 的 fan-out）
2. 循环修复直到通过（L2 的 loop）
3. 审批流（L4 的 interrupt）
4. 崩溃后恢复（L3 的 checkpoint）
5. 用子图隔离检查逻辑（L5 的 subgraph）

**故障注入测试：**
- 在不同节点 kill 进程，验证 resume 正确
- 改历史 checkpoint，验证分叉分支独立
- 审批拒绝，验证循环重试

---

## 5. 设计取舍与风格 / Design tradeoffs

### 5.1 为什么不用真实 LLM？

所有课程都用**假模型**（返回固定字符串的函数），原因：
1. **确定性**：同样的输入永远得到同样的输出，测试才能稳定
2. **无依赖**：不需要 API key、不需要网络、不需要付费
3. **快**：跑一遍测试 < 1 秒

L6 的 capstone 会提供一个"可选的真实 LLM 版本"，但测试还是用假模型。

### 5.2 为什么打印原始 chunk 而不是摘要？

很多教程会把 `stream()` 的输出做成摘要：
```python
# 常见做法（我们不这样做）
print(f"Step {i}: {node} -> {verdict}")
```

我们打印原始 chunk：
```python
# 我们的做法
for mode, chunk in graph.stream(..., stream_mode=["updates", "values"]):
    print(f"  {mode:8} {chunk}")
```

原因：**摘要是解释，chunk 是真相**。如果你想知道"同一 super-step 的节点看不到对方的写"，你需要看到两个 `updates` chunk 在同一轮里出现，而 `values` chunk 在下一轮才包含它们的合并结果。摘要会把这个时序关系抹掉。

### 5.3 为什么用中文 README + 英文代码？

代码和注释用英文（LangGraph 生态的惯例），README 用中文（你的请求）。这个混合风格的好处：
- 代码可以直接贴到 StackOverflow / GitHub issue，不需要翻译
- README 用母语解释"为什么"，降低理解门槛

---

## 6. 贡献指南 / Contributing

如果你要添加 L2-L6 或改进 L1，请遵循：

### 6.1 PR 检查清单
- [ ] 代码可以无依赖运行（`poetry install` 后直接 `poetry run python lessons/lN/main.py`）
- [ ] 所有测试通过（`poetry run pytest lessons/lN -v`）
- [ ] README 包含：核心概念、执行模型、踩坑点、官方文档链接
- [ ] 代码打印完整的运行时输出（用 `lib.trace()`，不做摘要）
- [ ] 至少 3 个"改坏它"练习

### 6.2 测试覆盖要求
- 每个踩坑点都有一个对应的测试（用 `pytest.raises` 或 `assert`）
- 测试名以 `test_` 开头，docstring 说明验证什么
- 所有测试在 CI 里跑（当前还没配 CI，但预留）

### 6.3 文档更新
- 添加新课时，更新 `docs/curriculum.md` 的对应章节
- 如果添加新的 helper，更新 `lib/helpers.py` 的 docstring
- 如果发现官方文档有错，在 README 里注明"官方文档 X 页的说法与实际行为不符"

---

## 7. 已知问题与未来改进 / Known issues & future improvements

### 7.1 已知问题
- L2-L6 还未搭建
- 没有 CI（GitHub Actions）自动跑测试
- `lib.trace()` 在并发节点很多时输出可能乱序（因为 `flush=True` 的时机）

### 7.2 未来改进
- 添加 Mermaid 图的自动截图（用 `mermaid-cli` 生成 PNG）
- 每课添加一个"常见问题"FAQ 章节
- L6 添加一个"生产环境 checklist"（什么时候用 Postgres、什么时候用 Redis、如何监控）

---

## 8. 参考资源 / References

- [LangGraph 官方文档](https://docs.langchain.com/oss/python/langgraph/)
- [LangGraph GitHub](https://github.com/langchain-ai/langgraph)
- [Pregel 论文](https://research.google/pubs/pub37252/)（LangGraph 的 super-step 调度模型来源）

---

**最后更新：** 2026-09-10  
**当前进度：** L1 完成，L2-L6 待搭建
