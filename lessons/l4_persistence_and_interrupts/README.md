# L4 - Persistence, Interrupts and Reliable Execution：持久化、人机协作与可靠执行

> 官方文档对应页
>
> - Persistence: <https://docs.langchain.com/oss/python/langgraph/persistence>
> - Interrupts: <https://docs.langchain.com/oss/python/langgraph/interrupts>
> - Time travel: <https://docs.langchain.com/oss/python/langgraph/use-time-travel>

## 通用学习目标与前置

L1 回答"一个 super-step 里发生什么"，L2 回答"循环怎么转"，本课回答：**当进程在第 N 步死掉、当人类在第 M 步说不、当你在第 K 步之后想反悔时，runtime 能保证什么、不能保证什么**。核心是三个对象：**checkpoint**（每个 super-step 结束时落盘的状态快照）、**thread**（恢复键 `thread_id`）、**interrupt**（把控制权交给人的暂停点）。学完本课应能独立解释并验证：

1. **checkpointer 与 thread**：同一图换 `thread_id` 即换会话；memory 随进程消失，sqlite 落在文件里——重建 graph + saver 对象后历史仍在。
2. **durability 三档**（`sync` / `async` / `exit`，`invoke`/`stream` 均可传）：中途失败后磁盘上各剩几个 checkpoint、SIGKILL 后各能不能恢复。
3. **动态 interrupt vs 静态断点**：节点内 `interrupt(payload)` 与 `compile(interrupt_before=[...])` 的暂停证据差异（`tasks[].interrupts` 有无 payload），以及各自用什么恢复。
4. **接受 / 修改 / 拒绝三条审核路径**：resume 值本身就是节点的 partial update，reject 通过条件边绕过写入节点。
5. **多中断顺序**：两个节点各带 interrupt、同一节点跨循环 interrupt 两次，恢复顺序稳定（答案按重放位置匹配）。
6. **真崩溃恢复与幂等**：SIGKILL 子进程后在**新进程**里恢复；外部写入按行直接计数；ledger（`thread_id` + 确定性 `task_id`）把副作用收敛回一次。
7. **重放范围**：只有未完成的 super-step 会重跑，已 checkpoint 的节点不再执行。
8. **历史分叉与外部撤销的区别**：time travel 分叉的只是**状态树**，外部世界一页都不翻回去；撤销是你要自己写的补偿操作。

前置：L1 的 super-step / channel / reducer，L2 的 Command 与 stop_reason 词汇。业务场景沿用 PCB 工艺参数：`propose -> review(interrupt) -> apply | closed`，`apply` 对外部 spec store（append-only 日志文件）做一次**非事务、非幂等**的写入——可靠性话题全部围绕这一次写入展开。

| # | 机制（必修） | 对应 DEMO | 对应测试 |
| --- | --- | --- | --- |
| 1 | checkpointer / thread / 续跑 / 重开 | DEMO 1 | `test_same_thread_continues_new_thread_fresh` |
| 2 | durability 三档落盘时机 | DEMO 2 | `test_durability_levels_leave_different_checkpoint_trails`, `test_resume_after_exception_re_runs_only_failed_node` |
| 3 | 动态 interrupt vs 静态断点 | DEMO 3 | `test_dynamic_interrupt_pauses_and_resumes_via_command`, `test_static_breakpoint_resumes_with_none_input` |
| 4 | 接受/修改/拒绝审核 | DEMO 4 | `test_accept_edit_reject_three_paths` |
| 5 | 多中断顺序对照 | DEMO 5 | `test_multiple_interrupts_stable_order` |
| 6 | 真崩溃 + 重启恢复 + 幂等 | DEMO 6 | `test_crash_before_write_recovers_with_single_write`, `test_crash_after_write_duplicates_external_effect`, `test_ledger_makes_replay_idempotent`, `test_durability_exit_sigkill_leaves_no_checkpoint` |
| 7 | 恢复重放范围 | DEMO 7 | `test_crash_after_checkpoint_replays_only_unfinished_work` |
| 8 | 历史查询、分叉、外部撤销 | DEMO 8 | `test_fork_diverges_but_external_undo_is_separate` |

## 方案取舍

| 选择 | 本课做法 | 为什么 | 生产替代 |
| --- | --- | --- | --- |
| 故障注入方式 | **子进程 + SIGKILL**（`crash_runner.py`），而非节点内 raise | 异常会走 Python unwinding（exit 档仍能写出收尾 checkpoint）；只有 SIGKILL 才是"进程突然消失"的真实现场 | k8s pod 驱逐 / OOM kill 复现 |
| 副作用载体 | append-only 日志文件，每行可直接数 | "落了没有、落了几次"用 `wc -l` 级别的事实回答，不从终态倒推 | 真实 DB 行 / HTTP 调用 + 遥测 |
| 幂等方案 | 执行 ledger：`thread_id + task_id` 键的 done-marker | 实测 **task_id 跨进程崩溃+恢复是稳定的**（由父 checkpoint 确定性派生），键天然防误杀 | DB 唯一约束 / 幂等 token / outbox |
| 审核载体 | 动态 `interrupt()`（payload 运行时算）为主，静态 `interrupt_before` 对照 | payload 携带提案内容、resume 值携带人的裁决，是人机协议的自然形状 | 静态断点用于调试；审批 UI 挂在 interrupt payload 上 |
| 观测手段 | `print_state` / `print_history` / 原始 stream chunk / 外部日志行 | 本课按课程渐进设计**不用 MLflow**，lib helpers 已够暴露暂停点与历史 | MLflow trace（L7 系统化） |
| checkpointer | memory 与 sqlite 各用一遍同一图 | 证明"持久化在文件不在对象"，并把崩溃实验架在 sqlite 上 | Postgres（连接池与运维见总纲 §5 深入边界，本课未实测） |

**核心取舍一句话：所有"恰好一次/会不会重跑"的断言都以外部文件的行数为准，框架给的每层保证（checkpoint、durability、interrupt、fork）分别用实验定价。**

## 运行前预测

先写下预测再运行 `main.py`（每条都能在输出里直接核对）：

1. 崩溃点选在**外部写入之后、该步 checkpoint 之前**：重启恢复后 WRITE 行数是几？崩溃点选在写入之前呢？
2. `durability="exit"` 的运行被 SIGKILL（不是异常）：重启进程里 `get_state` 能看到什么？换成 `sync` 呢？
3. reject 路径恢复后，外部文件比 accept/edit 少几行？`closed` 节点的写入来自哪里？
4. 同一个节点在循环里 interrupt 两次，两次 resume 的答案会不会配对错位？顺序由什么决定？
5. 从旧的 paused checkpoint 分叉出一条 edit 分支后：原线程的 `get_state` 还返回 accept 的结果吗？外部文件里现在有几行？
6. 对动态 interrupt 用 `invoke(None, config)`（不带 `Command(resume=...)`）会发生什么？

## 执行模型与最小例子

图：`START -> propose -> review(interrupt) -> (apply | closed) -> END`，`apply` 写外部 spec store。崩溃实验在 `crash_runner.py` 的 `propose -> apply -> report` 上做，父进程轮询 ready 文件后 SIGKILL：

```text
super-step N (node apply):
|-- node starts --|-- WRITE lands --|-- node returns --|-- checkpoint committed --|
                    ^ SIGKILL here: 外部写入已落地、checkpoint 未写
                      -> 恢复时 runtime 无从知晓 -> apply 整个重跑 -> WRITE 第二次

parent (main.py / test)                  child (crash_runner.py)
spawn run(sqlite,log,ready,point)  -->   build graph on sqlite checkpointer
poll ready file                    <--   到达崩溃点：写 ready 文件，sleep
SIGKILL                            --x   进程消失（rc=-9）
spawn resume(同 sqlite, 同 thread)  -->   get_state: next=('apply',) -> invoke(None)
```

checkpoint 事实（1.2.11 实测）：每次 invoke 先写 `source="input"` 的 checkpoint（step=-1），之后每个 super-step 结束写一个 `source="loop"` 的 checkpoint，编号递增；`get_state_history` 按**新→旧**返回；snapshot 上 `next`（待执行节点）、`tasks`（待执行任务的 id/name/interrupts）是调试的第一现场。恢复语义：**resume 会从节点函数开头重跑整个节点**（不是从 `interrupt()` 行继续），答案按重放位置匹配——所以中断前的副作用也会重放，这是"中断前只执行一次"简化不能带进实验的原因。

durability 三档实测对照（中途异常 = 节点 raise；SIGKILL = 真杀）：

| durability | 中途异常后的 checkpoint | SIGKILL 后 | 中途历史可分叉 |
| --- | --- | --- | --- |
| `sync`（默认，`None` 同） | 每步一个（3 个：step -1/0/1） | 从最后完成的 super-step 恢复 | 可以 |
| `async` | 与 sync 相同（**同步 saver 下不可区分**，见未实测） | 同上 | 可以 |
| `exit` | 恰好 1 个（退出时统一写；异常 unwind 也会写） | **0 个**：`HISTORY n=0`，线程无从恢复 | 不可以（中途无 checkpoint） |

fork / time travel 实测机制：**同线程** `update_state(old_snapshot.config, values, as_node="节点名")` 从旧 checkpoint 生出 `source="update"` 分叉点，随后 `invoke(None, fork_cfg)` 走分叉；`invoke(None, old.config)` 是 replay（重放会重执行节点、重触发 interrupt、重放外部副作用）。

## 关键反例与修复

**反例 1：崩溃在"写入后、checkpoint 前"→ 外部副作用双写（本课中心）。** `after_write` 崩溃实测：崩溃时日志已有 1 行 WRITE，恢复进程 `get_state` 显示 `next=('apply',)`（checkpoint 认为 apply 没跑完），`invoke(None)` 重跑 apply → **WRITE 共 2 行**。同一实验 `before_write` 崩溃 → 1 行；`after_checkpoint` 崩溃 → 1 行。**checkpoint 保证的是状态恰好一致，不是副作用恰好一次**（DESIGN §7 校正项的实测注脚）。修复：执行 ledger——写前查 `thread_id:task_id` 键，写后标记；实测恢复后 `node=apply` 仍出现 2 次（重放照旧）但 WRITE 只落 1 次，外加一行 `SKIP task=...`。**幂等挡的是效果，不是执行。**

**反例 2：`durability="exit"` 的两种死法不等价。** exit + 节点异常：unwinding 仍写出 1 个收尾 checkpoint，`next=('boom',)`，可恢复；exit + SIGKILL：什么都没写，`HISTORY n=0`，外部那行 WRITE 成了无主孤魂。**"退出时持久化"依赖进程能退出。**

**反例 3：旧文档的跨线程 fork 写法在 1.2.11 上静默失效。** `{"thread_id": 新, "checkpoint_id": 旧}`：`get_state` 返回空（sqlite 的 get_tuple 要求 thread_id 与 checkpoint_id 同时匹配，memory saver 同样不做跨线程检索）；更危险的是 `update_state` + `invoke` 在这个 config 上**不报错**——它从空状态静默重跑整条链（实测终态 `log` 里连 `propose:` 都没有），外部还多落一行 WRITE。正确分叉 = 同线程 `update_state(old.config, values, as_node=...)`。

**反例 4：同线程 fork 会移动线程 head；恢复 head 靠 replay。** 分叉后 `get_state(thread)` 返回的是**分叉分支**的终态，原分支只剩历史（可用，未丢）。实测恢复 head 的办法：对原分支的 paused checkpoint 重放**相同的** `Command(resume=...)`——注意此时 resume 值会被忽略、**缓存的原答案被重放**（实测传新值无效），恰好复现原分支；而这次 replay 又把 `apply` 重执行了一遍，外部 WRITE 再多一行。**replay 不是缓存读取，是重执行。**

**反例 5：`invoke(None)` 恢复不了动态 interrupt。** 实测：paused 线程 `invoke(None)` 后 `next` 仍是 `('review',)`，同一个 interrupt 原样重发。动态中断必须 `Command(resume=...)`；静态断点则 `None` 与 `Command(resume=True)` 都行（静态暂停的 stream 里也有 `__interrupt__` chunk，但内容是**空元组**，`tasks[].interrupts` 为空——这是区分两种暂停的判据）。

**开发期踩坑（写进代码注释）：** 节点参数的 TypedDict 注解是它的可见/可写 channel 子集——注解里没有的 key，节点写入会被**静默丢弃**（本课两处：`i` 未进 schema、节点注解缺 `decision`）。

## 跨场景迁移题（改坏它）

换成你自己的任务（如资料入库、代码合并审批）重定义 propose/apply 后，做三个破坏实验，每个先预测再运行：

1. **删掉 `crash_runner.py` 里 apply 的 ledger 命中检查**（`if LEDGER and ledger_has(key)` 整块）：预测 `after_write` 崩溃实验的 WRITE 行数回到几、`test_ledger_makes_replay_idempotent` 会怎么失败。
2. **把 review 图的路由改成 reject 也进 `apply`**（`route` 返回值去掉 `"closed"` 分支）：预测 `test_accept_edit_reject_three_paths` 的哪几条断言先炸、外部 WRITE 总数变成几。
3. **把 `main.py` 里 fork 语句的 `as_node="review"` 去掉**：预测 `update_state` 报错还是行为变化，`test_fork_diverges_but_external_undo_is_separate` 在哪一步失败（提示：没有 as_node 时更新算在谁头上）。

```bash
poetry run pytest lessons/l4_persistence_and_interrupts -v
```

## 运行命令与来源

```bash
poetry run python lessons/l4_persistence_and_interrupts/main.py   # 离线机制实验（8 个 DEMO，含 5 次真 SIGKILL）
poetry run pytest lessons/l4_persistence_and_interrupts -v        # 13 项机制测试（5 项含子进程 kill+resume）
```

- 版本（本地实测）：Python 3.12.8，langgraph 1.2.11，langgraph-checkpoint 4.2.0，langgraph-checkpoint-sqlite 3.1.1，langchain-core 1.6.2，pytest 9.1.1。
- 文档：本页顶部三个官方链接。**凡与常见文档说法不一致处（跨线程 fork、exit+SIGKILL、replay 忽略新 resume 值、invoke(None) 不恢复动态中断）均以 1.2.11 本地探针实测为准**，探针脚本未入库，结论已写进上表与反例。
- 本课无 MLflow、无 live 模型（课程渐进设计；观测用 `lib.helpers` 的 print_state / print_history / trace）。

## 完成记录

- 2026-09-14，`poetry run python lessons/l4_persistence_and_interrupts/main.py`：8 个 DEMO 全部按预期输出。关键数字：DEMO 2 中途异常后 checkpoint 数 sync/async/exit = **3/3/1**，三档 `next` 均为 `('boom',)` 可恢复，恢复后 `['a','boom','c']`；DEMO 5 中断顺序 `['material', 'confirm item 0', 'confirm item 1']`（4 次 stream 调用）；DEMO 6 三崩溃点恢复后 WRITE 行数 **1 / 2 / 1**，ledger 版 `node=apply` 2 次 + WRITE 1 次 + SKIP 1 行，exit+SIGKILL 版 `NO-CHECKPOINT`、`HISTORY n=0`；DEMO 7 重放范围 propose 1→1、apply 1→2、report 0→1（resume 进程 `HISTORY n=5`）；DEMO 8 外部文件 1→2→3→4 行（fork、伪 fork、replay 各 +1），手写补偿后回到 1 行。
- 2026-09-14，`poetry run pytest lessons/l4_persistence_and_interrupts -v`：**13 passed**（终版连跑两次核验稳定性：16.73s 与 18.58s，无 flake；其中 5 项为子进程 SIGKILL + 新进程恢复测试，靠 ready 文件握手与超时上限保持可靠）。
- **未实测**：`async` 档在异步 saver（如 AsyncPostgresSaver）上的 fire-and-forget 落盘语义——同步 SqliteSaver 下 async 与 sync 完全不可区分，无法离线观测；Postgres 连接池与分布式恢复运维（总纲 §5 深入边界）；多 worker 并发 resume 同一 thread 的竞态行为。
- 探针记录：durability 取值与签名、`interrupt`/`Command(resume=)` 导入路径、静态断点恢复方式、fork 三种写法、task_id 跨进程稳定性，均在写课件前用独立小脚本在 1.2.11 上实测，未凭记忆断言。
