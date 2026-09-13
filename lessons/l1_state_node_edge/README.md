# L1 - State, Node, Edge：LangGraph 的三个原语

> 官方文档对应页
>
> - Graph API: <https://docs.langchain.com/oss/python/langgraph/graph-api>
> - Use the Graph API: <https://docs.langchain.com/oss/python/langgraph/use-graph-api>
> - Choosing APIs: <https://docs.langchain.com/oss/python/langgraph/choosing-apis>

## 为什么这三个词放在第一课

LangGraph 把"agent 工作流"拆成三样东西，所有后面的能力（循环、检查点、中断、子图）都只是这三样的组合变形：

| 原语      | 本质                                                               | 一句话职责                               |
| --------- | ------------------------------------------------------------------ | ---------------------------------------- |
| **State** | 一组 **channel**（通道），每个 channel 有自己的合并规则（reducer） | 定义"节点之间能传什么、并发写怎么合并"   |
| **Node**  | 一个 `state -> partial_update` 的纯函数                            | 读全部状态，只返回自己**想改的 key**     |
| **Edge**  | 静态或动态的连线                                                   | 决定下一个 super-step 谁跑、几个节点并行 |

理解的关键在于：**节点不直接改变 state，它只是"提交一份更新"**，由运行时在 super-step 结束时按 reducer 合并。这个间接层是后面所有能力的地基。

## 执行模型：super-step

LangGraph 的调度单位叫 **super-step**（超步），执行循环是：

```
1. 找到本轮应该跑的节点集合（由 edge 决定，可能多个）
2. 这些节点并行执行，各自返回一份 partial update
3. 等全部结束（join），对每个 channel 调用 reducer 合并所有更新
4. 写入 checkpoint，回到 1，直到没有节点可跑
```

第 3 步的"等全部结束再合并"叫 **Pregel 式同步屏障**。它带来两个直接后果，本课代码都会亲手验证：

- **同一 super-step 内并发写同一个 channel 会报错**（除非该 channel 有 reducer）——见 `main.py` 的坑 A。
- **同一 super-step 内的节点互相看不见对方的写入**——它们读到的都是上一轮的状态快照。这是初学者最常见的困惑，亲手验证一次就再也不会记错。

## State 的三种写法

`StateGraph` 接受任意类型作为 schema，实际常用的三种：

| 写法                 | 优点                       | 代价                                                       | 什么时候用                           |
| -------------------- | -------------------------- | ---------------------------------------------------------- | ------------------------------------ |
| `TypedDict`          | 零开销、最轻、文档示例最多 | 无运行期校验                                               | 默认选择，性能敏感的图               |
| `pydantic.BaseModel` | 入口处做类型校验与强制转换 | 递归校验慢；**只有第一个节点入参被校验**，输出不是模型实例 | 输入来自外部/LLM、需要兜底时         |
| `dataclass`          | 有默认值、比 pydantic 快   | 无校验                                                     | 想要"类"的写法但不想付 pydantic 的价 |

> 官方明确提示的 pydantic 限制：输出**不会**是模型实例；运行期校验**只发生在图的第一个节点入参**，后续节点与输出不校验；错误信息里看不出是哪个节点挂的。

## Reducer：这一课真正的主角

```python
class State(TypedDict):
    n: int                                              # 无 reducer: 后写覆盖，且并发写会报错
    log: Annotated[list[str], operator.add]             # 有 reducer: 并发写自动合并
```

- **没有 reducer 的 channel** 是 `LastValue` 语义：一轮里只能有一个写入者，多个写入者直接 `InvalidUpdateError`。
- **有 reducer 的 channel** 会把同一轮里每个节点的返回值依次喂给 reducer：`_add([a])`、`_add([b])` → `[a, b]`。
- reducer 是**无状态的二元函数** `(left, right) -> merged`，框架按节点返回顺序左折叠。
- 常见 reducer：`operator.add`（列表拼接 / 数字相加）、`operator.or_`（字典合并）、自定义 `def dedupe(l, r)`。

一句话记忆：**"谁写"由 edge 决定，"怎么合"由 reducer 决定。** 并行 fan-out 能不能用，取决于目标 channel 有没有 reducer。

## 三种 schema 分层

`StateGraph(state, input_schema=..., output_schema=...)` 可以给同一个图配三套 schema：

- `input_schema`：调用方**必须**提供的字段（对 LLM/外部输入的第一道关卡）
- `output_schema`：`invoke()` **返回**哪些字段（内部字段不外泄）
- 私有 schema：节点自己声明额外的 channel，做节点间私密通信

两个官方反复强调的细节：

1. 节点**能写任意**在图中注册过的 channel，哪怕它的入参 schema 里没有这个 key。
2. **私有 channel 在 stream 时不会被隐藏**。`invoke()` 只返回 output_schema，但 `stream_mode="values"` 默认吐**全部** channel。要收口就用 `output_keys=[...]`。这是个真实的安全坑：以为私有就没泄露。

## 本课代码要验证的行为

跑一遍就知道了：

```bash
poetry run python lessons/l1_state_node_edge/main.py
```

1. `START -> intake -> (style_check ‖ security_check) -> reduce -> END`，两个 check 节点在**同一个 super-step** 里跑，`findings` 由 `operator.add` 合并。
2. `intake` 里两个节点看到了什么——打印出来证明它们看不到对方。
3. 故意去掉 `security_check -> reduce` 这条边，看 `reduce` 在只有 style 结果时就跑了（join 语义由**入边**决定，不是自动的）。
4. 坑 A：两个并发节点写无 reducer 的 channel → `InvalidUpdateError`。
5. 私有 channel 通过 stream 泄露 → 用 `output_keys` 收口。

## 自己动手（改坏它）

- 把 `findings: Annotated[list[str], operator.add]` 的 `Annotated` 去掉，跑测试，读那条报错信息。
- 把 `style_check` 和 `security_check` 改成串行（`style_check -> security_check`），观察 super-step 数量的变化。
- 把 schema 换成 pydantic，传一个错误类型进去，确认报错发生在**第一个节点**而不是 `graph.invoke` 入口。

对应测试：

```bash
poetry run pytest lessons/l1_state_node_edge -v
```

## MLflow 实战：把 super-step 看进 trace 里

本课起每课配一个 `mlflow_demo.py`，对着本地 MLflow 服务器验证同一批机制（实验空间 `agent-loop`）：

```bash
docker compose up -d                                            # 起 MLflow（repo 根目录，端口 5000）
poetry run python lessons/l1_state_node_edge/mlflow_demo.py     # 连接 + 打 trace + 记 run + 读回
poetry run pytest lessons/l1_state_node_edge/test_mlflow.py -v  # 服务器不可达时自动 skip
```

demo 分四段：0 连接并确认 `agent-loop` 实验空间；1 autolog 打 trace（一次 `invoke` 一条 trace，一个节点一个 span）；2 显式 `start_run` 记 params / metrics / artifact；3 从服务器读回 runs 和 traces 两个视图。**每段都把服务器返回的原始事实打印出来**，不是只打印我们发出去的东西。

mlflow 3.16 的三个实测事实（写代码前先知道，都是踩出来的）：

1. LangGraph 的追踪入口在 **langchain flavor**：`mlflow.langchain.autolog()` 通过 LangChain callback manager 挂 tracer；没有独立的 `mlflow.langgraph` 模块。
2. trace **默认异步上报**（`MLFLOW_ENABLE_ASYNC_TRACE_LOGGING=True`）：`invoke` 完立刻 search 可能搜不到，读取时要带 `flush=True`。
3. `search_traces(experiment_ids=...)` 已废弃，用 `locations=...`。

以及两个"哪些遥测可信"的教训：

- **span 树的形状可信**：fan-out 的两个节点是同一父 span 下的兄弟；join 屏障表现为 `reduce` 的 span 一定在两个 check 都结束之后才开始。这两条写进了 `test_mlflow.py` 的断言。
- **兄弟 span 的时间重叠不可信**：span 时钟在 LangChain 回调触发时才走，微秒级的真并行节点也可能被记录成串行——demo 连跑两遍能看到 True/False 翻转，所以只打印不断言。

顺带一个 langgraph 1.2 的坑：`stream(mode="updates")` 是**按任务完成逐个吐 chunk**，同一 super-step 的两个并行节点占两个 chunk，所以"数 chunk"≠"数 super-step"。权威计数是 checkpoint 的 `metadata.step`（demo 里的 `count_super_steps`），app.py 走 4 步、.env 走 3 步（条件边直达 END）。

UI 入口：<http://localhost:5000/#/experiments/3>（Runs 与 Traces 两个 tab；run 内还能看到挂在 run 上的关联 trace）
