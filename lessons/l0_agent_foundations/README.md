# L0 - Why Agents：为什么需要 Agent

> 官方文档对应页
>
> - Building effective agents: <https://www.anthropic.com/engineering/building-effective-agents>
> - ReAct: Synergizing Reasoning and Acting in Language Models: <https://arxiv.org/abs/2210.03629>

## 通用学习目标与前置

前置：Python 基础。**本课故意不 import langgraph，也不使用 MLflow**——这些概念先于任何框架存在，L1 起才用 State/Node/Edge 重新表达它们，观测能力也从 L1 起逐步加入。本课全部是纯 Python + `lib` 的打印助手。

学完本课你应该能：

- 说清 **单次模型调用（single shot）、固定 workflow、agent 循环** 三者的边界：谁决定步骤的顺序
- 用 **目标 / 行动 / 观察 / 反馈** 解释 agent 循环每一圈在做什么，并指出哪一环由 runtime 负责
- 区分 **模型、工具、runtime/harness** 的分工，知道什么不该交给模型
- 用**直接计数**（模型调用次数、工具执行次数、循环圈数、模拟延迟单位）比较三种方案，并**指出哪种场景不应增加 Agent 复杂度**

本课的任务是贯穿项目的最小形态：从一段 PCB 工艺文本中抽取三个参数（板厚 `board_thickness`、铜厚 `copper_weight`、阻焊类型 `solder_mask`）并校验单位/范围。三种方案用**同一个任务、同一组工具、同一份评分标准**，唯一变量是控制流在哪里。

agent 循环的一圈（`main.py` 的 `run_agent_loop` 就是这张图的约 30 行实现）：

```
 goal (task + stop conditions)
   |
   v
 +--------+  action: tool_call      +---------+  execute   +-----------+
 |  model | ---------------------> | runtime | ---------> |   tools   |
 |        |                        | harness |            | pure code |
 |        | <--------------------- +---------+ <--------- +-----------+
 +--------+  feedback: observation appended to the message history
   |
   v  loop until: final answer | no progress | budget exhausted
```

## 方案取舍

Anthropic 文章的区分线：**workflow** 是代码预先写好路径、LLM 在其中个别步骤被调用；**agent** 是 LLM 自己决定接下来的过程。单次调用则是最退化的一种——没有过程，只有一问一答。

| 角色 | 职责 | 本课代码里的对应 | 不该让它做的事 |
| --- | --- | --- | --- |
| 模型 (model) | 读上下文，提出下一步行动（tool_call）或给出最终答案；擅长自由文本理解 | `ScriptedModel`（离线脚本）/ `live_client()`（真实调用） | 不要让它承担确定性校验、单位换算或计数——DEMO 1 里单次调用的铜厚误判就是证据 |
| 工具 (tools) | 确定性能力：读原文、正则抽取、规格校验；输出可断言 | `read_source` / `extract_parameters` / `check_spec`（`make_tools`） | 不要在工具里藏状态或再做一层"迷你 agent"；工具是纯函数 |
| runtime / harness | 驱动循环、派发工具、把观察拼回消息历史、**强制终止条件**（成功/无进展/预算） | `run_agent_loop` 的 while 循环与 `seen` / `max_steps` 守卫 | 不要替模型决定行动顺序——那是固定 workflow 的活 |

自主性/质量/延迟/成本的取舍（数字来自 `main.py` DEMO 1 的实际输出，模拟延迟单位：模型调用 5、工具执行 1、代码阶段 1）：

| 维度 | 单次调用 | 固定 workflow | Agent 循环 |
| --- | --- | --- | --- |
| 自主性（谁决定步骤） | 无人决定，一次成型 | 代码写死 | 模型按观察临时决定 |
| 质量（本课实测） | **2/3**：校验靠模型自觉，单位换算错就带错出门 | **3/3**：校验回到代码，确定性 | **3/3**：工具给出精确观察 |
| 模型调用 / 工具执行 | 1 / 0 | 1 / 3 | 6 / 5 |
| 延迟（模拟单位） | 5.0 | 10.0 | 35.0 |
| 成本 | 最低 | 低-中 | 最高，且方差大（见 DEMO 3 的两种失败） |
| 适用场景 | 输入格式稳定、任务一步能说清 | 步骤已知且可枚举 | 步骤无法预先写死、下一步取决于上一步的观察 |

**一句话：自主性是买来的，付款方式是延迟、成本和新的失败模式。** 没有场景证据之前，默认选更简单的那个。

## 运行前预测

跑 `main.py` 之前先写下你的预测（这是课程要求的关键预测练习，答案在下一节）：

1. 单次调用会把铜厚 `1 oz` 的校验做对吗？它最容易在哪里翻车？
2. 固定 workflow 付几次模型调用？校验步骤付几次？
3. agent 循环要几圈才能交出答案？它比固定 workflow 多付了什么？
4. 换成固定格式日志行 `THK=1.6mm|CU=1oz|MASK=LPI`，三种方案（外加纯代码解析）谁的质量/成本最优？
5. 如果模型反复发起**完全相同**的 tool_call，循环谁来叫停？

## 执行模型与最小例子

```bash
poetry run python lessons/l0_agent_foundations/main.py
```

DEMO 1，自由文本任务 `订单备注：板材 FR-4；板厚 1.6 mm；外层铜厚 1 oz；阻焊类型 LPI（液态感光油墨）。`，实测基线：

| approach | model calls | tool execs | loop iters | latency units | quality | stop reason |
| --- | --- | --- | --- | --- | --- | --- |
| single_shot | 1 | 0 | 1 | 5.0 | 2/3 | completed |
| fixed_workflow | 1 | 3 | 0 | 10.0 | 3/3 | completed |
| agent_loop | 6 | 5 | 6 | 35.0 | 3/3 | final_answer |

- **single shot**：抽取 + 校验 + 格式化全在一次调用里。脚本化的模型把 `1 oz` 心算成 "35" 却写成 mm，再拿 35 去对板厚上限 4.0 mm，自信地报 fail——**没有任何下游环节能接住这个错，因为不存在下游**。
- **fixed workflow**：parse（代码）→ extract（模型，仅此一次付费）→ validate（代码，调的正是 agent 也会派的 `check_spec`）→ format（代码）。质量不再依赖模型的心算。
- **agent loop**：每圈打印 action 与 observation；第 2 圈的 `extract_parameters` 参数里嵌着第 1 圈 `read_source` 观察到的原文——**这就是"观察流入下一步行动"的直接证据**，也被 `test_main.py` 断言。

DEMO 2，确定性反例：机器日志行 `THK=1.6mm|CU=1oz|MASK=LPI`，实测基线：

| approach | model calls | tool execs | loop iters | latency units | quality | stop reason |
| --- | --- | --- | --- | --- | --- | --- |
| pure_code_parser | 0 | 4 | 0 | 4.0 | 3/3 | completed |
| single_shot | 1 | 0 | 1 | 5.0 | 3/3 | completed |
| agent_loop | 6 | 5 | 6 | 35.0 | 3/3 | final_answer |

**三方质量同为 3/3，agent 却烧掉 6 次模型调用、5 次工具执行和 35 个延迟单位**，去拿纯代码 0 次调用就能拿到的答案；而且它是唯一可能额外撞上 DEMO 3 两种失败的模式。这就是"不应增加 Agent 复杂度"的场景，用数字说话。

DEMO 3，只有 agent 才有的失败模式（同一个 `run_agent_loop`，换脚本）：

- 卡死模型（重复同一 tool_call）→ `no_progress` 停止：3 次模型调用、2 次工具执行、质量 0/3
- 打转模型（每轮换参数重试）→ `budget_exhausted` 停止（`max_steps=3`）：3 次模型调用、3 次工具执行、质量 0/3

固定 workflow 不会打转，因为它根本不循环；单次调用也不会。**自主性带来的新失败面，就是终止条件设计——它是 agent 的一部分，不是事后补丁。**

## 关键反例与修复

1. **反例：单次调用把校验也交给模型。** 症状：DEMO 1 的 2/3，铜厚 verdict 错但没有任何报错。修复：把校验挪回代码——固定 workflow 复用同一个 `check_spec` 工具后 3/3。教训：**模型负责理解自由文本，规则性判断放代码**。
2. **反例：agent 循环没有 no-progress 守卫。** 症状：模型重复发起相同 tool_call，循环空转烧预算。修复：`seen` 集合按 `(tool, args)` 规范化键判重，第二次出现即停；注意判重发生在**执行之前**（测试断言 tool execs 少 1）。
3. **反例：agent 循环没有预算。** 症状：每轮换一点参数的打转模型让 no-progress 永不触发。修复：`max_steps` 硬上限；预算耗尽时明确返回 `budget_exhausted` 而不是假答案。
4. **反例之王：确定性任务上套 agent。** DEMO 2 的两行数字（0 调用 3/3 vs 6 调用 3/3）就是完整论证，不需要更多修辞。

## 跨场景迁移题

先判断三种场景里**哪种不应增加 Agent 复杂度、各自该停在哪种方案**，再对照参考判断写理由（判断依据应该是"步骤能否预先枚举"和"下一步是否取决于上一步的观察"，不是"听起来高级"）：

1. 每晚批量解析 1000 条固定格式产线日志（形如 `THK=1.6mm|CU=1oz|MASK=LPI`）并校验入库。参考：纯代码解析——DEMO 2 的 0 次调用 3/3 就是答案；加模型已经是浪费，加 agent 是负收益。
2. 客户用自然语言写来料检验备注，字段位置不定，偶尔引用需要查询的规格手册。参考：固定 workflow 起步（模型只付在自由文本抽取那一步）；只有当"下一步取决于上一步观察"的分支多到无法枚举时才升级 agent。
3. 90% 的客服问题命中 FAQ 检索即可回答，10% 需要多步查询（查订单 + 查物流 + 改地址）。参考：单次调用 + 检索覆盖 90%；把复杂 10% 路由给一个小 agent，而不是让 100% 流量进循环。

自己动手（改坏它）：

- 把 `run_agent_loop` 里的 `seen` 集合判重删掉，用 `NO_PROGRESS_SCRIPT` 跑 DEMO 3：原本第 3 轮就叫停的运行现在会烧完预算（budget 兜底）；再把 `max_steps` 调到 100，体会"没有终止设计的自主性"意味着什么。
- 把 `SPEC` 里 `copper_weight` 的 `"unit"` 改成 `"mm"`，跑测试：固定 workflow 的质量从 3/3 掉到 2/3——证明**质量保证来自工具里的正确规则，不是流水线形状本身**。
- 把 DEMO 1 里 agent 的 `max_steps` 设为 2 再跑：`stop_reason` 变成 `budget_exhausted`、质量 0/3——预算不足时自主性直接变成失败。

对应测试：

```bash
poetry run pytest lessons/l0_agent_foundations -v
```

## 运行命令与来源

```bash
# 离线机制实验（纯 Python，无网络、无框架、无 MLflow）
poetry run python lessons/l0_agent_foundations/main.py

# 离线行为断言（直接计数模型调用/工具执行/循环圈数）
poetry run pytest lessons/l0_agent_foundations -v

# 真实模型单次调用（需 repo 根目录 .env 配 OPENAI_API_KEY；未配置则打印 skip 提示并退出 0）
poetry run python lessons/l0_agent_foundations/live_demo.py
```

来源：[Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)（workflow 与 agent 的区分线、常见组合模式）、[ReAct 论文](https://arxiv.org/abs/2210.03629)（行动—观察交替的原始表述）。本课的模拟延迟比例（5:1:1）是声明的教学假设，L7 接入真实遥测后替换为实测值。

## 完成记录

- `poetry run python lessons/l0_agent_foundations/main.py`：通过。DEMO 1 基线 2/3、3/3、3/3（模型调用 1/1/6，延迟 5.0/10.0/35.0）；DEMO 2 纯代码解析 0 次模型调用、质量 3/3，agent 6 次调用同质量；DEMO 3 `no_progress`（3 调用/2 执行/0 分）与 `budget_exhausted`（3/3/0 分）均按预期停止。
- `poetry run pytest lessons/l0_agent_foundations -v`：通过，7 passed, 1 skipped（skip 为 live 测试，符合预期）。
- `poetry run python lessons/l0_agent_foundations/live_demo.py`：未验证（未配置 .env）。打印 `describe_live_config()` 的 skip 提示并以退出码 0 结束。
- 真实模型在同一任务上是否会犯脚本里那种单位换算错误、真实 usage 数字：未验证（未配置 .env）。
