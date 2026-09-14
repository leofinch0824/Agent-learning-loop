# Agent Learning

学习 Agent 核心思想与 LangGraph 运行机制，通过最小实验建立可迁移能力，再以业务 Skill、多模态模型迭代和训练数据挖掘进行实践。

- [课程总纲](docs/curriculum.md)：已确认目标、通用必修、业务迁移、八周安排、掌握标准与阅读资料。
- [实现设计](docs/DESIGN.md)：目标目录、课件约定、业务工作流、实验设计与待校正项。
- [业务术语](CONTEXT.md)：业务知识、Skill、归因假设、优化候选和训练样本的区别。

L0–L7 已有课件（L8–L10 为规划，完整进度由课程总纲维护）。各课 README 按 DESIGN.md 固定章节组织，离线机制实验全部可跑；真实模型（live）实验需在 repo 根目录配置 `.env`，未配置时自动 skip 并在各课完成记录中标注「未验证」。

| 单元 | 课件 | 一句话主题 |
| --- | --- | --- |
| L0 | [lessons/l0_agent_foundations](lessons/l0_agent_foundations/README.md) | 单次调用 / 固定工作流 / Agent 循环：何时值得上 Agent |
| L1 | [lessons/l1_state_node_edge](lessons/l1_state_node_edge/README.md) | State/Node/Edge、reducer、super-step（含 MLflow 基线） |
| L2 | [lessons/l2_agent_loop](lessons/l2_agent_loop/README.md) | 原生循环与图循环、结构化工具调用、四种终止 |
| L3 | [lessons/l3_tools_and_execution](lessons/l3_tools_and_execution/README.md) | 工具契约、权限执行分离、沙箱、MCP 协议边界 |
| L4 | [lessons/l4_persistence_and_interrupts](lessons/l4_persistence_and_interrupts/README.md) | checkpoint/durability、interrupt 审核、真崩溃恢复与幂等 |
| L5 | [lessons/l5_context_and_memory](lessons/l5_context_and_memory/README.md) | 上下文四层分工、Store 记忆、裁剪/摘要、缓存口径 |
| L6 | [lessons/l6_planning_and_subgraphs](lessons/l6_planning_and_subgraphs/README.md) | Plan-and-Execute、Send 扇出、子图与 Command.PARENT |
| L7 | [lessons/l7_observability_and_evaluation](lessons/l7_observability_and_evaluation/README.md) | 结果/轨迹评价、judge 校准、失败归因、反思闭环 |

## 运行课件

离线机制实验（无网络、确定性 fake 驱动）：

```bash
poetry install
poetry run python lessons/l1_state_node_edge/main.py   # 换成任意一课的 main.py
poetry run pytest lessons/ -v                           # 全部课程；或 lessons/l2_agent_loop 指定单课
```

真实模型实验（可选；OpenAI 兼容端点，可指向 Qwen 等服务）——在 repo 根目录建 `.env`：

```bash
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1   # 或兼容服务地址
LIVE_MODEL_NAME=gpt-4o-mini                  # 或业务模型名
```

```bash
poetry run python lessons/l2_agent_loop/live_demo.py    # 各课 live_demo.py
poetry run pytest lessons/l2_agent_loop/test_live.py -v # 未配置时自动 skip
```

本地 MLflow 实验空间为 `agent-loop`，服务数据在被忽略的 `mlflow-data/`。L2 起 live 运行轻量记 trace 与费用，L7 系统化使用：

```bash
docker compose up -d
poetry run python lessons/l7_observability_and_evaluation/mlflow_demo.py
poetry run pytest lessons/l7_observability_and_evaluation/test_mlflow.py -v
```

MLflow 不可达时集成测试会跳过；跳过不代表真实验证通过。离线机制实验与真实模型实验分别记录结果，后者按课件明确模型配置和调用预算。

## 文档与资料

课程内容只在总纲维护，目录与实验约定只在实现设计维护；`docs/references/` 保留四篇背景研究，由总纲按主题导航。本地 `llms.txt` 是被忽略的官方文档索引，可以检索主题后定位原文。

本轮合并前的课程、调研草稿及 MLflow 改动已保存于本地提交 `9587537`。旧方案由 Git 历史保留，工作区维护当前版本。

依赖以 `pyproject.toml` 和实际安装版本为准。MIT。
