# L5 - Context, Retrieval and Memory：上下文、检索与记忆

> 官方文档对应页
>
> - Memory: <https://docs.langchain.com/oss/python/langgraph/add-memory>
> - 上下文压缩调研: [横向拆解Claude Code、Codex等六大Agent上下文压缩策略后，我们做了第 7 个](../../docs/references/横向拆解Claude%20Code、Codex等六大Agent上下文压缩策略后，我们做了第%207%20个.md)

## 通用学习目标与前置

L2 回答"一轮行动—观察怎么转"；本课回答"模型每一轮到底看见了什么，以及哪些东西能活过这一轮"。学完本课应能独立解释并验证：

1. **四层分工**：working context（真正递给模型的消息列表）、会话状态（State channels）、checkpoint（按 thread_id 隔离的执行历史）、跨会话 Store（按 namespace/key 寻址，跨所有 thread 存活）——同一事实放进四层，**新 thread 只看得到 Store 层**。
2. **检索与写回**：run 开始时 recall 节点把**经确认**的记忆注入 system prompt；run 结束时 writeback 节点把候选记忆（带 provenance）写回 Store；确认是显式动作，不是"写进去就生效"。
3. **更新 / 冲突 / 过期**：version 与 source 随条目走；冲突候选**双方共存**各带来源，等人工裁决；纠正 = 新版本上位 + 旧版本标记 `superseded_by`（保留审计痕迹，不是删除）；过期是应用层规则。
4. **裁剪 / 摘要 / 选择性加载**：手写 token 估算器（chars/4，声明为近似）、`trim_messages`（system 恒活 + 最近窗口优先）、中段摘要（折叠后**关键 spec 行必须仍在模型可见消息中**）、按 topic 的选择性加载。
5. **长工具结果管理**：200 行表格不进 messages，进 State 的 `artifacts`，模型只看 digest + 引用。
6. **更多上下文 != 更好**：噪声对照实验里 5 条无关记忆把脚本模型的答案单位从 mm 翻成 mil。
7. **前缀缓存与真实命中的区分**：前缀稳定是必要条件；**命中只能由 provider 遥测证明**，本课离线部分只证必要条件，live 部分未配置遥测环境，标记未实测。

前置：L1 的 channel/reducer 与 checkpoint `metadata.step`；L2 的消息 wire format（OpenAI chat-completions dict）与 fake/live 同签名 `respond(messages)`。业务场景沿用 PCB 工艺参数助手。

## 方案取舍

| 选择 | 本课做法 | 为什么 | 生产替代 |
| --- | --- | --- | --- |
| 断言对象 | fake 模型记录**实际收到的消息列表**（`FakeModel.seen`），一切断言写在 `seen` 里 | "记忆生效"的唯一判据是字符串出现在模型收到的消息里，不是 store 里"应该有" | 同左；真实模型用 live 对照 |
| Store | `InMemoryStore` + `compile(store=...)`，节点用 keyword-only `store: BaseStore` 参数接收 | 机制最小、可离线；probed 事实见下节 | PostgresStore 等带 TTL/索引的实现 |
| 记忆元数据 | status/version/source/topics/expires_at 全部放在 value dict 里 | 1.2.11 的 `put` 没有 metadata kwarg（probed）；元数据契约是自己的，不绑死 store 实现 | 索引字段交给 store 的 index 配置 |
| 过期 | 应用层 `expires_at` + recall 过滤链 | `InMemoryStore.put(ttl=...)` 直接 `NotImplementedError`（probed 原文见下） | 支持 TTL 的 store 实现或后台清扫 |
| token 估算 | `chars // 4` | 声明为近似值；确定性保证测试可复算 | 真 tokenizer / provider 计数 |
| 裁剪与摘要 | 全部手写（trim / summarize / selective） | 被研究的就是策略本身，必须可读可改 | langchain 的 trim_messages / summarization 中间件 |
| 噪声对照 | 脚本模型唯一规则 = 从 system 里找 `prefers <unit>` | 确定性替身模拟注意力劫持（DESIGN §1：离线断言确定性行为）；因果链可见 | 真实模型 A/B，L7 评测口径 |
| 前缀缓存 | 只证前缀稳定（必要条件），命中证明交给 usage 遥测 | DESIGN §7 校正：**前缀哈希相等不能证明缓存命中** | provider 遥测字段（live_demo 探测） |

**核心取舍一句话：本课所有"记忆生效"的断言都以模型实际收到的消息为证据；框架能替我们做的（裁剪/摘要/检索），都手写一遍换取可解释性。**

### langgraph 1.2.11 Store 集成 probed 事实（写课件前独立小脚本实测，未凭记忆断言）

1. import 路径：`from langgraph.store.memory import InMemoryStore`；挂载：`builder.compile(store=store)`，可与 `checkpointer=` 同时使用。
2. **普通函数节点接收 store 的方式**：keyword-only 参数、**名字必须叫 `store`、注解必须是 `BaseStore`**（或 `Optional[BaseStore]`）——按"名字 + 注解"从 runtime 对象注入；或在节点里 `from langgraph.config import get_store()`（返回同一对象）。
3. **反例**：`from langgraph.prebuilt import InjectedStore` 服务于 ToolNode 风格的**工具**；注解在普通节点上不会注入，invoke 时报 `TypeError: ...() missing 1 required keyword-only argument: 'store'`（DEMO 2 演示原文）。
4. **静默坑**：compile 不挂 store 时节点拿到的 `store` 是 `None`，`get_store()` 也返回 `None`，**不报错**。
5. API 形状：`put(namespace_tuple, key, value_dict)`（无 metadata kwarg）；`get(namespace, key) -> Item(namespace, key, value, created_at, updated_at)`；`search(namespace_prefix, query=, filter=, limit=, offset=)`（filter 作用于 value 字段）；`list_namespaces()`。
6. `put(..., ttl=...)` 在 InMemoryStore 上抛 `NotImplementedError: TTL is not supported by InMemoryStore. Use a store implementation that supports TTL or set ttl=None.`；同 key 覆盖是整条替换（value/时间戳都换），所以 version 要自己维护。

## 运行前预测

先写下你的预测再运行 `main.py`（每条都能在输出里直接核对）：

1. 同一条 "0.127 mm" 事实分别放进 thread A 的工作上下文、`task_note`、checkpoint 和 Store：thread B 的模型能看到几处？
2. writeback 写入候选偏好后**立即**开新 thread：system 里会出现这条偏好吗？`confirm_memory` 之后再开一个呢？
3. `expires_at` 在 2020 年的条目会被注入吗？它还在 store 里吗？
4. 已确认但错误的 "via diameter 0.5 mm" 被纠正后：新 thread 的模型看到 0.5 还是 0.3？旧条目被删了吗？
5. 预算只够 system + 最后一问一答时：被裁掉的 4 条里，"外层最小线宽 0.1 mm" 以什么形式存活？关键行 "内层 = 0.127 mm（spec v3）" 呢？
6. 200 行工具表：naive 与 managed 两种方式下，模型收到的 tool 消息各多大？
7. 同一问题、同一张图，system 里多 5 条无关记忆：答案会变吗？朝哪个方向变？
8. 三次请求共享同一段稳定前缀：能据此宣称缓存命中吗？

## 执行模型与最小例子

图拓扑（本课主图，挂在 L2 的循环形状上，记忆注入节点在 run 首尾）：

```
START -> recall -> model -> writeback -> END
          ^                     |
          |   Store -> working context (检索/注入)
          |   conversation -> Store (写回候选)
          +-- checkpoint 按 thread_id 隔离; State channels 只在本 thread 的节点间流动
```

四层分工表（DEMO 1 逐项验证）：

| 层 | 本质 | 寻址 | 生命周期 | 模型看得到吗 |
| --- | --- | --- | --- | --- |
| working context | 实际递给模型的 messages 列表 | 无（就是本次调用的参数） | 单次模型调用 | **看得到（定义）** |
| 会话状态 | State channels（`question`/`task_note`/`artifacts`） | channel 名 | 本 thread 内跨 super-step | 看不到，除非节点显式拷进消息 |
| checkpoint | 每 super-step 的完整状态快照 | thread_id | 本 thread 终结后仍可查，但别的 thread 摸不到 | 看不到（是恢复/审计材料） |
| Store | key-value 条目 + namespace | (namespace, key) | **跨所有 thread** | 看不到，除非 recall 注入 |

记忆条目 value schema（全部字段自己维护）：

```python
{"text": ..., "status": "confirmed|candidate|superseded",
 "version": int, "source": "thread:<id>", "topics": "...", "expires_at": "ISO|None"}
```

recall 的**唯一过滤链**（`recall_memories`，journal 记录每条被注入/跳过及原因）：`search -> status 门槛 -> 过期门槛 -> (可选) 相关性门槛`。writeback 只产 candidate；`confirm_memory` 显式晋升（version+1），`supersede_memory` 显式让位（保留 `superseded_by` 审计链）。

裁剪策略（`trim_messages` / `collapse_with_summary`）：system 消息恒活且计入预算；非 system 从最新往旧装填直到预算耗尽；被裁中段交给摘要器（离线 `fake_summarizer` / live 同一契约），折叠结果为 `[system..., [summary], 最近窗口...]`。**摘要是兜底不是保险箱**：数字能活，绑定会丢（DEMO 4 实测："0.1" 活了，"外层"丢了）。

## 关键反例与修复

**反例 1：`InjectedStore()` 用在普通函数节点上。** 1.2.11 实测：`compile()` 不报错，`invoke()` 时节点因缺少 keyword-only 参数而 `TypeError: ...() missing 1 required keyword-only argument: 'store'`（DEMO 2、`test_injectedstore_annotation_fails_on_plain_nodes`）。修复：keyword-only `*, store: BaseStore`（名字 + 注解匹配注入）或 `get_store()`。**更隐蔽的变体**：compile 时根本没挂 store——节点拿到静默 `None`，没有任何异常，只有注入悄悄失效；修复是防御性判空 + 集成测试断言"模型 seen 里出现了记忆"。

**反例 2：把 checkpoint / State 当记忆用。** DEMO 1：A 的工作上下文、`task_note`、执行历史在新 thread 的可见性全部为"不可见"，唯独 Store 层可见。checkpoint 是按 thread_id 隔离的执行历史（恢复与 time travel 的材料），**不能当跨会话知识库**；State channel 不进消息就永远到不了模型（`test_state_channels_never_leak_into_the_model_context`）。

**反例 3：全量注入记忆 = 喂噪声。** DEMO 6：同一任务，多 5 条无关记忆（其中一条恰好提到单位），脚本模型的答案从 "0.127 mm" 翻成 "5.000 mil"，且上下文成本更高。修复方向：选择性加载（topic 门槛）+ 确认门槛（candidate 不注入），而不是"多记总没错"。**注意**：干扰是脚本模型的确定性规则（模拟注意力劫持），真实模型的噪声敏感度要用 live/评测实验另证。

**反例 4：前缀相同就宣称缓存命中。** DEMO 7 只证明三次请求共享 304 chars 稳定前缀——必要条件。命中证据只能是 provider 遥测（`prompt_tokens_details.cached_tokens` / `cache_read_input_tokens` 等），水位线与折扣是各家可变参数。本课无遥测配置，**未实测**；live_demo 配好 .env 后打印 usage 字段，缺失时不做任何宣称。

## 跨场景迁移题（改坏它）

换成你自己的任务（资料检索、代码检查）重定义记忆条目与工具后，做三个破坏实验，每个先预测再运行：

1. **把 `recall_memories` 里的 `status != "confirmed"` 分支删掉**：预测过期与被纠正的条目重新出现在新 thread 的 system 里。验证：`poetry run pytest lessons/l5_context_and_memory -v`（`test_expired_memory_is_not_injected`、`test_wrong_memory_correction_supersedes` 会失败——确认门槛是它们唯一的防线）。
2. **把 `trim_messages` 里"system 恒活"的分支去掉，让 system 也参与从新到旧的装填**：预测折叠后关键 spec 行仍在（它在最后），但 system 丢失时摘要/注入锚点没了；再把预算改成只够最后一条，预测 `test_collapse_summary_preserves_key_recent_information` 如何失败——体会"保谁"是策略决定，不是框架默认。
3. **把 managed 工具节点改回直接 append 全表（`managed=False`）**：预测 `test_long_tool_result_managed_as_digest_and_reference` 中体积断言失败，并算一算 10 轮对话多付多少上下文；再把 digest 里的 min/max 删掉，观察哪条断言先红——digest 不是装饰，是模型下一轮唯一的信息来源。

```bash
poetry run pytest lessons/l5_context_and_memory -v
```

## 运行命令与来源

```bash
poetry run python lessons/l5_context_and_memory/main.py        # 离线机制实验（7 个 DEMO，无需网络）
poetry run pytest lessons/l5_context_and_memory -v            # 13 项机制测试（live 2 项 skip）
poetry run python lessons/l5_context_and_memory/live_demo.py  # 真实摘要质量 + 前缀遥测探针（需 .env）
poetry run pytest lessons/l5_context_and_memory/test_live.py -v # live 测试（未配置 .env 时 skip）
```

- 版本（本地实测）：Python 3.12.8，langgraph 1.2.11，langgraph-checkpoint 4.2.0，langchain-core 1.6.2，pytest 9.1.1。
- 文档：本页顶部官方 Memory 页 + 上下文压缩调研（curriculum §7 L5 条目的原文入口）；L2 的消息 wire format 与 fake/live 契约。
- 本课不使用 MLflow（渐进设计）：观测靠 lib helpers 与模型 seen/journal 直接证据。

## 完成记录

- 2026-09-14，`poetry run python lessons/l5_context_and_memory/main.py`：7 个 DEMO 全部按预期输出——四层可见性 Store 可见 / 其余三层不可见；candidate 不注入、confirm(v2) 后新 thread system 含 `user prefers answers in mm units [v2]`；过期条目 journal 记 `expired`、纠正后只见 0.3 且旧条目留 `superseded_by`；裁剪后关键 spec 行仍在、摘要只剩裸数字 0.1；200 行表 naive 5317 chars vs managed 168 chars（约 31x）；噪声把答案单位从 mm 翻成 mil（system 23 -> 67 tok）；前缀 304 chars 共享（仅必要条件）。
- 2026-09-14，`poetry run pytest lessons/l5_context_and_memory -v`：**13 passed, 2 skipped**（skipped 均为 live 测试，理由 live model not configured）。
- 2026-09-14，`poetry run python lessons/l5_context_and_memory/live_demo.py`：无 `.env`，打印配置摘要后 exit 0——**真实摘要质量与缓存遥测：未验证（未配置 .env）**。
- 未验证 / 未实测清单：真实模型中段摘要的绑定保真度（live 1）；provider 前缀缓存命中率与收益（live 2，DESIGN §7 口径，字段缺失时不宣称）；真实模型噪声敏感度（离线为确定性替身）。
- 探针记录：Store 集成六条事实（import / keyword-only `store: BaseStore` 注入 / InjectedStore 反例报错原文 / 静默 None / put-get-search 形状 / ttl NotImplementedError 原文）在写课件前用独立小脚本于 langgraph 1.2.11 实测，未凭记忆断言。
