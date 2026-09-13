# Agent Learning

学习 Agent 核心思想与 LangGraph 运行机制，通过最小实验建立可迁移能力，再以业务 Skill、多模态模型迭代和训练数据挖掘进行实践。

- [课程总纲](docs/curriculum.md)：已确认目标、通用必修、业务迁移、八周安排、掌握标准与阅读资料。
- [实现设计](docs/DESIGN.md)：目标目录、课件约定、业务工作流、实验设计与待校正项。
- [业务术语](CONTEXT.md)：业务知识、Skill、归因假设、优化候选和训练样本的区别。

当前只有 [L1](lessons/l1_state_node_edge/README.md) 有课件实现，含本地 MLflow 示例；部分概念与断言待校正。其他单元为课程规划，完整进度由课程总纲维护。

## 运行现有课件

```bash
poetry install
poetry run python lessons/l1_state_node_edge/main.py
poetry run pytest lessons/l1_state_node_edge/test_main.py -v
```

本地 MLflow 实验空间为 `agent-loop`，服务数据在被忽略的 `mlflow-data/`：

```bash
docker compose up -d
poetry run python lessons/l1_state_node_edge/mlflow_demo.py
poetry run pytest lessons/l1_state_node_edge/test_mlflow.py -v
```

MLflow 不可达时集成测试会跳过；跳过不代表真实验证通过。离线机制实验与真实模型实验分别记录结果，后者按课件明确模型配置和调用预算。

## 文档与资料

课程内容只在总纲维护，目录与实验约定只在实现设计维护；`docs/references/` 保留四篇背景研究，由总纲按主题导航。本地 `llms.txt` 是被忽略的官方文档索引，可以检索主题后定位原文。

本轮合并前的课程、调研草稿及 MLflow 改动已保存于本地提交 `9587537`。旧方案由 Git 历史保留，工作区维护当前版本。

依赖以 `pyproject.toml` 和实际安装版本为准。MIT。
