# 业务模型迭代 Agent：实际课程路线

更新：2026-09-13。按算法工程师、每周约 13–15 小时、接受付费 API 的条件设计。目标见 [learning-goals.md](learning-goals.md)，目录与工作流见 [course-implementation.md](course-implementation.md)。

当前仅 L1 有课件实现，L2–L10 均待开发；“有代码”“机制验证”“真实业务验证”“独立掌握”分开记录。旧 L1–L6 全文保留在 [历史路线](archive/curriculum-langgraph-baseline.md)。本页课号为新编号，不与旧课号混用。

## 主线与依赖

先获得可观察的 Agent 执行能力，再沉淀业务 Skill；在可信评测和人工归因之后学习自动优化，最后把需要后训练的问题送入数据生产流程。可观测性和小型评测从 L2 起贯穿。

| 单元 / 计划目录 | 学习目标与项目增量 | 关键实验与过关证据 | 预计投入 |
| --- | --- | --- | --- |
| L1 `l1_state_node_edge/`（已有） | 理解 state、reducer、super-step 与汇聚；完成当前基础课校正 | 预测非对称路径；直接记录汇聚调用次数；解释并发快照与更新冲突。能完成迁移题即可跳过重复阅读 | 6–8 h |
| L2 `l2_agent_loop/` | 条件路由、Command、终止条件、真实模型工具调用；读入一份模型结果并输出结构化分析 | fake/live 同接口；参数错误和重复工具调用；MLflow 读回一次 trace；记录固定提示词小基线 | 8–10 h |
| L3 `l3_durable_review/` | SQLite checkpoint、thread、interrupt 与幂等；做可恢复的人工审核队列 | 审核前/后故障重启；拒绝、修改与恢复；证明审核结果不重复应用；区分历史分叉与外部撤销 | 10–12 h |
| L4 `l4_business_skills/` | 业务知识的来源、适用范围、更新；Skill 加载、定制、调用与版本固定；基础子图组合 | 原始规则/候选解释冲突；知识更新后旧运行仍能回放；两个任务 Skill 共用一个业务规则；第二个流程复用同一 Skill | 10–12 h |
| L5 `l5_multimodal_evaluation/` | 图文输入、输出结构、证据定位；数据分组、基线与评价口径 | 漏页/错裁剪/图像顺序错误/标注歧义；比较给定正确证据与实际选证据的端到端结果；人工复核评分 | 10–12 h |
| L6 `l6_failure_attribution/` | 错例筛选、聚类、归因假设与小型对照；区分提示词问题、数据问题和待训练能力 | 分析几类不同成因；只修输入或只换指令作对照；记录确认/否定/未知；生成可行动反馈 | 10–12 h |
| L7 `l7_prompt_optimization/` | 固定输入和评分，自动生成候选；接一种现成优化器；预算、选择与停止 | 固定提示词 vs 单轮反思 vs GEPA；相同验证集和可比预算；最终留出测试；无收益时保留基线 | 12–16 h |
| L8 `l8_skill_evolution/` | 将提示词优化扩展到 Skill 的说明、示例和步骤；复合流程与版本晋升 | 候选→验证→人工审阅→发布/拒绝；固定业务规则与评价器；新旧版本对比；一个真实失败后回退 | 10–12 h |
| L9 `l9_training_data_mining/` | 从离线导出的线上记录/模型结果挖掘数据；去重、优先级、人工纠正、导出 | SFT、偏好、RL 任务素材分别校验；无标签案例不充当真值；与随机抽样对照审核产出；防止同源数据跨集合 | 12–16 h |
| L10 `l10_capstone/` | 完整业务模型迭代助手；集成可靠性、真实验证与学习复盘 | 演示两条工作流；一项留出结果、一次审核恢复、一个拒绝候选、可追溯数据导出；独立新增规则或 Skill | 8–10 h |

总教学估算约 96–120 小时，含实验、阅读与审核；不含大型数据标注或训练集群搭建。课次不等于周次，首个业务的数据准备工作也计入学习时间。

## 8 周建议节奏

| 周 | 重点 | 可检查成果 |
| --- | --- | --- |
| W1 | L1、L2；确定一个业务任务，搜集少量代表样本 | 可运行 Agent 循环、原始 trace、固定提示词输出；通用基础只补诊断发现的缺口 |
| W2 | L3；L4 的固定版本 Skill 与业务依据；L5 最小人工评分 | 审核后可恢复；业务 Skill v0；基线与样本分组 |
| W3 | L5 主体、L6 最小归因 | 图文证据可追溯的评测；明确几种错例的下一步，不全部归入提示词问题 |
| W4 | L7 最小自动优化闭环；必要时只比较基线与一种优化方法 | **4 周交付点**：候选、验证、人工检查、一次留出结果及接受/拒绝结论 |
| W5 | 补 L6/L7 的反馈质量、预算与回归实验 | 修复一个优化失败；解释反馈差或数据偏导致的误改；小集只能作为机制证明 |
| W6 | L4 基础流程组合的迁移练习、L8 Skill 自迭代 | Skill v1 候选与版本报告；发布/拒绝/回退；两个可组合的业务流程 |
| W7 | L9 | 去重和审核队列；三种用途的数据素材；小样本审核效率对照 |
| W8 | L10、缓冲与复盘 | 全链路运行记录、独立迁移、训练数据交付包、未解决问题清单 |

4 周版采用纵向切片，只实现 L4/L6/L7 的最小必要能力；8 周再扩展。若落后，先减少候选轮数、算法数量和界面建设，保留真实评测、审核及数据分离。CLI 加本地审核表即可完成主线，Web UI 不作为前置。

## 每次学习怎么进行

工作日一小时建议：10 分钟回忆/预测，35 分钟实验或项目修改，15 分钟解释结果并留下下一步。周末完成较长的真实模型实验、人工复核和对照报告。API 运行等待可以安排其他学习工作，人工标注时间不能省略不计。

AI 可协助生成代码骨架、检查 trace、提出反例和解释论文；每单元至少一个关键预判和迁移修改由学习者独立完成。通过标准是能发现错误假设并修正工作流，读完资料不代表掌握。

## 主线之外

| 扩展 | 何时加入 |
| --- | --- |
| MCP、复杂多 Agent、动态规划 | 现有工具接入或任务拆分已成为具体瓶颈；基础子图已经在 L4/L8 学习 |
| 向量检索、复杂长期记忆 | 业务知识规模使显式规则/标签检索不足；主线先做来源与版本正确 |
| 分级压缩和 Prompt Cache | 图文/工具上下文确实挤占预算；主线已有证据保留、输出限长与 usage 记录 |
| MIPROv2、OPRO 等更多优化方法 | 能解释 GEPA 与简单反思基线的差异后，选一种对照；不逐个复刻优化平台 |
| SFT/RL 训练作业接入 | 有数据、训练脚本及算力；把模型版本与相同评价口径接回项目，真实训练另记证据 |
| Postgres、部署、沙箱、在线流量 | 原型需要长期服务、多人并发或更广执行能力时进入 |

## 按课阅读的第一手资料

先读支持本课实验的段落，接口以实际安装版本为准。

- L1–L2：[Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api)、[Workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents)。
- L3：[Persistence](https://docs.langchain.com/oss/python/langgraph/persistence)、[Interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts)。
- L4/L8：[Subgraphs](https://docs.langchain.com/oss/python/langgraph/use-subgraphs)；业务 Skill 数据约定由本项目实验决定，不把特定客户端的 Skill 自动加载语义套到自建 runtime 上。
- L2/L5：[MLflow GenAI](https://mlflow.org/docs/latest/genai/)；复用现有 `lib/mlflow_utils.py`。
- L7：[GEPA 论文](https://arxiv.org/abs/2507.19457)、[GEPA FAQ](https://gepa-ai.github.io/gepa/guides/faq/)、[优化 API](https://gepa-ai.github.io/gepa/api/core/optimize/)。先理解候选—执行—反馈—选择，再接 API。
- L9：[TRL 数据格式](https://huggingface.co/docs/trl/dataset_formats)、[GRPOTrainer](https://huggingface.co/docs/trl/grpo_trainer)。按目标训练器核验导出内容和图像处理约定。

原 [调研草稿](lesssons-specify-draft.md) 与 `docs/references/` 保留背景；其中引用错配和泛化结论不能直接作为课件断言，校正清单见 [落地设计](course-implementation.md)。
