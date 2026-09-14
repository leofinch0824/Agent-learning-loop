# L7 - Observability, Evaluation and Feedback：可观测、评测与反馈改进

> 官方文档对应页
>
> - MLflow GenAI（ tracing / evaluation 主线）: <https://mlflow.org/docs/latest/genai/>
> - Agent 评测方法论: <https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents>

## 通用学习目标与前置

本课把 L2 起积累的原始 trace、小型结果集和费用记录系统化：**不再问"跑通了没有"，而是问"怎么知道它好不好、坏在哪、改了有没有变好、花了多少"。**

学完应能独立回答：

| 能力                                                                 | 本课证据                                            |
| -------------------------------------------------------------------- | --------------------------------------------------- |
| 说清 **run / trace / span / step** 的区别并从 trace 还原执行过程      | main DEMO 1 + mlflow_demo section 1/2               |
| 区分**结果评价**与**轨迹评价**，各举一个只有对方能抓到的失败          | main DEMO 2，`test_result_correct_trajectory_bad` 等 |
| 按**来源分组留出**数据，解释行级切分为什么会泄漏                      | main DEMO 3，`test_group_split_*`                   |
| 做失败归因（tool/routing/model/budget/**unknown**）并报告未知         | main DEMO 4，`test_attribution_labels_*`            |
| 使用规则 / 人工 / LLM judge 三种评价并**校准** judge                   | main DEMO 5/6，`test_fake_judge_calibration_numbers` |
| 做**资源预算**检查（按案例、按来源组、按总额）                        | main DEMO 7 + mlflow_demo section 3                 |
| 跑一次**反思（evaluator-optimizer）前后对照**并报告退化               | main DEMO 8/9，`test_reflection_*`                  |

前置：L1（super-step、reducer、MLflow 实战）、L2（agent 循环、FakeModel、四种终止）。本课被测系统**直接 import L2 的 `build_loop_graph` 与 `FakeModel`**，不重新实现。

## 方案取舍

- **观测后端只用 MLflow**（本地 docker，实验空间 `agent-loop`），代码不引入其他追踪后端。
- **离线确定性优先**：`main.py` 不碰网络和服务器，12 个案例全部由脚本化 FakeModel 驱动，任何断言都能复现；服务器相关内容集中在 `mlflow_demo.py`，真实 LLM judge 集中在 `live_demo.py`，两者都按可达性优雅降级。
- **评价器复用同一套规则**：离线 `main.py`、服务器 `mlflow_demo.py` 的 scorers、genai.evaluate 的期望值走同一个 `params_match` / `trajectory_score_from_facts`，保证三处分数一致（test_mlflow 直接断言这一点）。
- **反思是脚本化的**：`reflect()` 输出固定修订方案，`REVISED_SCRIPTS` 是"应用该方案后每个案例的脚本化结果"。这诚实地隔离了变量——本课验证的是**评价-归因-修订-复测的闭环方法**，不是模型真的会自我改进（那是 L9 的业务优化）。
- **judge 的离线 fake 与 live 版共享接口** `judge(case, answer) -> {verdict, score, reason}`，校准逻辑同一套。

## 运行前预测

先写下预测再运行（课程要求每课至少一个独立预测）：

1. c02（重复调用后成功）：**结果分 1.0，轨迹分 < 1.0**；c06/c07（自信错答案）：**结果分 0，轨迹分 1.0**。
2. 只查流畅度的 judge v1 在 12 案例上与真值的一致率：**约 75%（9/12）**；加证据引用的 v2：**约 92%（11/12）**，且 v2 会误杀一个"近似说法"案例。
3. 反思后聚合分会明显上升，但 **vendor-datasheet 组会小幅下降**（修订加了一次防御性验证调用）。
4. c11（标注 0.2 vs 工具表 0.3）：反思**修不好**，归因应报 `unknown` 而不是 model_error。
5. 从 span 树单独定位：c04 能看出 budget 指纹（7 个 model span 收在裸 model 上），c02 能看出重复调用；**c07 的树形与成功案例无法区分**。

## 执行模型与最小例子

```bash
poetry run python lessons/l7_observability_and_evaluation/main.py
```

数据集 12 案例、4 个来源组，失败模式刻意分布：

| 案例  | 组                | 设计的失败模式                         | 结果/轨迹 |
| ----- | ----------------- | -------------------------------------- | --------- |
| c01-03 | specbook-a 前三个 | 正常（c02 = 语义重复调用后成功）       | c02: 1.0 / 0.75 |
| c04   | specbook-a        | 可解案例跑到 budget 停止               | 0.0 / 0.3  |
| c05   | specbook-b        | 连续三个坏参数 -> tool_error           | 0.0 / 0.9  |
| c06   | specbook-b        | 工具正确但答案用心算（200 vs 196.85）  | 0.0 / 1.0  |
| c07   | specbook-b        | **流畅自信的错答案（reward hacking）** | 0.0 / 1.0  |
| c08   | specbook-b        | 同一请求两次 -> no_progress            | 0.0 / 0.5  |
| c09-10 | vendor-datasheet  | 正常（c10 近似说法，v2 会误杀）        | 1.0 / 1.0  |
| c11   | vendor-datasheet  | 标注与工具表冲突 -> **unknown**        | 0.0 / 1.0  |
| c12   | legacy-notes      | 正常                                   | 1.0 / 1.0  |

评价器公式（全部在 `main.py` 可读、在测试里钉死）：

- **结果分**：成功终止且每个期望参数（值 1% 容差 + 单位在场）都匹配才得 1.0。
- **轨迹分**：`1 - 0.2*loops - 0.05*wasted - 0.5*wrong_termination`；loops 按**语义键**（`3` 与 `3.0` 归一化后相同）计重复，wasted 是超出 `min_tool_calls` 的执行数，wrong_termination 是可解案例上的 budget/no_progress 停止。
- **归因顺序**：stop_reason（tool_error / budget）→ 轨迹气味（routing_error）→ 答案与工具值的关系（`faithful_to_tool` 为真且结果错 -> **unknown**，疑似标注；否则 model_error）。
- **分组留出**：对**组名**（不是行）做固定种子 shuffle，按 0.5/0.25/0.25 分配 -> 2/1/1 组。
- **预算**：单案例 250 tokens、全集 1400 tokens（基线实际 1500 -> 超支；反思后 1300 -> 回到限内）。
- **反思**：`failure_summary` -> `reflect()`（文字方案）-> `REVISED_SCRIPTS` 重跑同一数据集。

## 关键反例与修复

1. **高分但任务失败（reward hacking，课程指定反例）**。c07 的答案"根据工艺规范，外层最小线宽为 0.15 mm。该结论基于长期生产经验，可信度高。"——judge v1（只查流畅+格式）给 **0.9（pass，被骗）**；judge v2（要求逐字引用证据 "0.1 mm"）给 **0.1（fail，抓住）**；规则真值 result_score=0。**修复不是换模型，是换判据**：`test_reward_hacking_v1_high_v2_low` 断言同一案例同一答案 v1>=0.8 且 v2<0.8。
2. **行级随机切分泄漏**。同一文档的近重复样本分进 train 和 test，指标虚高。修复 = 整组留出；`test_group_split_is_disjoint_complete_and_deterministic` 断言组不跨集合、切分可复现。
3. **trace 定位的边界**。c04（budget）、c02（loop）能从 span 树**单独**归因；c07 的树形（model->tools->model）与成功案例完全同形——**WHERE（哪个节点）和 WHETHER（任务成没成）是两个问题**，后者必须靠评分。mlflow_demo section 2 把这个诚实的限制打印出来。
4. **mlflow 3.16 评测面（全部实测，勿信博客）**：
   - 经典 `mlflow.evaluate` 自 3.0 起废弃；用 python 函数模型对本服务器**直接挂死**（>100s，predict_fn 从未被调用）-> 记为未实测/不可用，不硬套。
   - `mlflow.genai.make_genai_metric` **不存在**；正确面是 `@mlflow.genai.scorer` 装饰器 + `mlflow.genai.evaluate(data, scorers, predict_fn=...)`。
   - genai.evaluate 会校验 `inputs` 的键等于 predict_fn 参数名、复用活动 run（`l7-` 前缀得以保留）、自动为每次预测建 trace、把分数记成 `<scorer>/mean`。
   - 其 worker 线程的懒加载 import 会**死锁**（faulthandler 抓到 import lock 环）；修复 = 主线程先 warm-up（`_warm_up_lazy_imports`），实测从挂死变 3.7s。
5. **聚合改善下的组级退化**。反思后聚合 0.5927 -> 0.9365，但 vendor-datasheet 0.75 -> 0.7458。**报告必须把退化列出来**，`test_minor_group_regression_is_scripted_and_visible` 钉死它。
6. **未知就是未知**。c11 反思后仍失败，归因 `unknown`（模型忠实引用工具值，标注与源冲突）-> 送人工复核，不臆造解释。

## 跨场景迁移题（改坏它）

1. 把 `split_by_group` 改成行级随机（直接 shuffle 案例再切）——`test_group_split_is_disjoint_complete_and_deterministic` 会失败：同一来源组的案例出现在两个集合。
2. 把 judge v2 的证据检查删掉（`quoted` 恒为 True，v2 退化为 v1）——`test_reward_hacking_v1_high_v2_low` 与 `test_fake_judge_calibration_numbers` 失败：反例重新变得不可见。
3. 把 `attribute_failure` 里 `faithful_to_tool` 分支删掉（结果错一律 model_error）——`test_attribution_labels_constructed_cases` 失败：unknown 消失，c11 被"解释"成一个其实不成立的原因。

```bash
poetry run pytest lessons/l7_observability_and_evaluation -v
```

换一个业务域（如资料检索问答）：保留 L2 循环，重写数据集与 expected/evidence；判断三件事——轨迹评价的 loops/wasted 阈值在新域是否还合理、judge v2 的"逐字引用"在开放问答里是否过严（会制造 c10 式误杀）、来源分组键换成什么（文档？网站？客户？）。

## 运行命令与来源

```bash
docker compose up -d                                                # 本地 MLflow（端口 5000）
poetry run python lessons/l7_observability_and_evaluation/main.py   # 离线全流程（无依赖）
poetry run python lessons/l7_observability_and_evaluation/mlflow_demo.py  # 服务器主线
poetry run python lessons/l7_observability_and_evaluation/live_demo.py   # 真实 LLM judge（需 .env）
poetry run pytest lessons/l7_observability_and_evaluation -v        # 全部测试
```

来源：curriculum §2 L7 行与 §7 L7 阅读入口（上面 blockquote 两条）；L1 `README.md` 的 MLflow 实战与 `lib/mlflow_utils.py` 的三个实测事实；L2 `main.py`（被测系统）。版本：mlflow 3.16.0、langgraph 1.2.11、langchain-core 1.6.2。运行名统一 `l7-` 前缀，便于在共享 `agent-loop` 实验空间里识别与清理。

## 完成记录

- 2026-09-14：`main.py` 全流程跑通；`test_main.py` 20 passed（离线，确定性）。数据集分数、归因、校准（v1 9/12、v2 11/12）、反思前后（0.5927 -> 0.9365，vendor 组 -0.0042）、预算（1500 -> 1300）与 README 表述一致。
- 2026-09-14：`mlflow_demo.py` 五段跑通（服务器在线）；`test_mlflow.py` 7 passed。12 个 `l7-case-*` run 各带一条 linked trace；从 span 树单独定位 c04/c02 成功、c07 明确报告"树形无法解释"；服务器端聚合成 650/450/300/100 tokens；genai.evaluate 记录 `l7_result_match/mean=0.5`、`l7_trajectory_quality/mean=0.8708`，与离线一致。
- **未验证**：`live_demo.py` / `test_live.py`（无 `.env`，3 skipped）——真实 LLM judge 的校准数字与 v2 抓取行为待配置后补记；genai.evaluate 的内置 LLM judges（`mlflow.genai.judges.is_grounded` 等）未实测（本课只跑了自定义规则 scorer）；经典 `mlflow.evaluate` 挂死，记为不可用；`MlflowClient` 无删除 trace 的 API（探到 AttributeError），探针留下的孤儿 trace 无法清理。
- 诚实声明：反思是脚本化的（`REVISED_SCRIPTS`），验证的是闭环方法而非模型自改进；真实优化器对照在 L9。
