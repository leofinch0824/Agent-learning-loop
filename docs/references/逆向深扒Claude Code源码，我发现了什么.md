# 逆向深扒Claude Code源码，我发现了什么！？

🔬 Claude Code v2.1.88 源代码深度解析报告

基于 GitHub 泄露源代码的逆向工程分析 分析版本: @anthropic-ai/claude-code v2.1.88 | 约 1,884 个 TypeScript 文件 | ~512,664 行代码 | 12MB 编译后 bundle

Executive Summary

Claude Code 之所以优秀，不在于单一技术突破，而在于 12 层渐进式工程包装（Progressive Harness） 将一个最小 Agent Loop 逐层升级为工业级自主编码代理。其核心竞争力可归纳为三个维度：

| **维度**           | **核心洞察**                                           | **竞品差距**                   |
| ------------------ | ------------------------------------------------------ | ------------------------------ |
| **🏗️ 工程化**      | 编译时 DCE + 分层 prompt cache + 会话持久化 + 流式执行 | 多数竞品还在单文件 prompt 阶段 |
| **🤖 Agent 架构**  | 多模态子代理 + Fork 隔离 + Swarm 协调 + Worktree 并行  | 竞品大多只有单层 Agent         |
| **⚡ Skills 系统** | 按需注入 + CLAUDE.md 惰性加载 + 远程 Skill 搜索        | 竞品大多是静态 prompt 注入     |

# 01

架构全景

![](./assets/17892859513140.14940137925626729.png)

最大的文件是 query.ts，约 785KB。 这不是代码混乱的标志——它是整个 Agent Loop 的心脏，所有的流式处理、工具调度、上下文压缩、子代理管理都在这个文件中协调。这种"胖核心"设计是刻意的：保持核心循环的原子性，避免分散到多个模块导致的协调复杂度。

# 02

核心 Agent Loop

Claude Code 的核心是一个极其简洁的 while-true 循环：

![](./assets/17892859513470.49355018813624285.png)

这就是全部。 这个循环的简洁性是 Claude Code 架构优雅的根本原因。

为什么这么简单就够了？

[HIGH CONFIDENCE] 因为复杂性被推到了两个地方：

1. 工具层 — 每个工具自己处理输入验证、权限检查、并发安全性、渲染等
2. 包装层（Harness） — 12 层机制在这个循环外部叠加 production 特性

核心循环本身从不改变。添加一个新功能 = 添加一个新工具或一个新的包装层。这是经典的 Open-Closed Principle（开闭原则）在 Agent 系统中的完美实践。

# 03

12 层渐进式包装机制

这是 Claude Code 最核心的工程洞察：一个 production AI agent 需要在基础循环之上叠加 12 层机制。每一层都建立在前一层之上。

![](./assets/17892859513530.7598130876007884.png)

各层详细解读

| **层级** | **机制**            | **关键源码**                                  | **工程洞察**                                          |
| -------- | ------------------- | --------------------------------------------- | ----------------------------------------------------- |
| **s01**  | Agent Loop          | `query.ts`                                    | while-true + tool_use 检查 + append                   |
| **s02**  | Tool Dispatch       | `Tool.ts` + `tools.ts`                        | `buildTool()` 工厂 + 统一注册表                       |
| **s03**  | Planning            | `EnterPlanModeTool` / `TodoWriteTool`         | 先列步骤再执行，**完成率翻倍**                        |
| **s04**  | Sub-Agents          | `AgentTool` + `forkSubagent.ts`               | 子代理获得全新 messages[]，保持主对话干净             |
| **s05**  | Knowledge On Demand | `SkillTool` + `memdir/`                       | 通过 tool_result 注入，不污染 system prompt           |
| **s06**  | Context Compression | `services/compact/`                           | 三层策略：autoCompact + snipCompact + contextCollapse |
| **s07**  | Persistent Tasks    | `TaskCreate/Update/Get/List`                  | 基于文件的任务图，带状态追踪和依赖关系                |
| **s08**  | Background Tasks    | `DreamTask` + `LocalShellTask`                | 守护线程运行命令，完成时注入通知                      |
| **s09**  | Agent Teams         | `TeamCreate/Delete` + `InProcessTeammateTask` | 持久化队友 + 异步邮箱                                 |
| **s10**  | Team Protocols      | `SendMessageTool`                             | 一个请求-响应模式驱动所有代理间协商                   |
| **s11**  | Autonomous Agents   | `coordinator/coordinatorMode.ts`              | 空闲周期 + 自动认领，无需 lead 逐个分配               |
| **s12**  | Worktree Isolation  | `EnterWorktreeTool` / `ExitWorktreeTool`      | 任务管理目标，worktree 管理目录，通过 ID 绑定         |

s03 Planning 的回报率惊人。 从源码注释看，仅仅添加 "先列步骤再执行" 的机制就使任务完成率翻倍。这与学术界 Plan-and-Execute Agent 的研究结论一致。

# 04

工具系统

   4.1 Tool Interface 设计

每个工具通过 buildTool() 工厂函数创建，实现统一的生命周期接口：

![](./assets/17892859513620.22917429707504144.png)

   4.2 工具全景

![](./assets/17892859513710.3432678392639831.png)

   4.3 关键设计决策

buildTool() 工厂的 fail-closed 默认值 — 这是安全设计的典范

```
const TOOL_DEFAULTS = {  isEnabled: () => true,  isConcurrencySafe: () => false,    // 默认：不安全 → 串行执行  isReadOnly: () => false,            // 默认：有写入 → 需要权限  isDestructive: () => false,  checkPermissions: () => 'allow',    // 默认：交给通用权限系统  toAutoClassifierInput: () => '',    // 默认：跳过安全分类器
```

这意味着：

- 新增工具时忘记声明并发安全性？→ 自动串行执行（安全的默认）
- 忘记声明只读？→ 自动触发权限检查（安全的默认）
- 安全相关工具必须显式覆盖 toAutoClassifierInput

   4.4 并行工具执行

StreamingToolExecutor 是性能关键组件：

```
Claude API 返回多个 tool_use blocks        │        ▼StreamingToolExecutor 分区：├── isConcurrencySafe() == true  → 并行执行 ──→ Promise.all()└── isConcurrencySafe() == false → 串行执行 ──→ 顺序 await        │        ▼结果 append 到 messages[]，回到 Agent Loop
```

这解释了为什么 Claude Code 在文件搜索这类操作上感觉很快——多个 GlobTool / GrepTool 调用被并行执行。

# 05

Prompt 工程

   5.1 分层缓存架构

Claude Code 的 system prompt 被精心分割为静态前缀和动态后缀，这是 prompt cache 命中率的关键优化。

```
System Prompt 结构:┌─────────────────────────────────────────────┐│  STATIC PREFIX（全局可缓存）                  ││  ├── Identity + Intro                       ││  ├── System Rules                           ││  ├── Doing Tasks Guidelines                 ││  ├── Actions Safety Section                 ││  ├── Using Your Tools Section               ││  ├── Tone & Style                           ││  └── Output Efficiency                      ││                                             ││  ═══ DYNAMIC BOUNDARY MARKER ═══           ││  __SYSTEM_PROMPT_DYNAMIC_BOUNDARY__         ││                                             ││  DYNAMIC SUFFIX（会话特定，不缓存）          ││  ├── Session-specific Guidance              ││  ├── Memory (CLAUDE.md)                     ││  ├── Environment Info                       ││  ├── Language / Output Style                ││  ├── MCP Server Instructions               ││  └── Scratchpad / FRC / Budget              │└─────────────────────────────────────────────┘
```

关键约束（源码注释原文）：

```
WARNING: Do not remove or reorder this marker without updating cache logic in:- src/utils/api.ts (splitSysPromptPrefix)- src/services/api/claude.ts (buildSystemPromptBlocks)
```

这意味着 Anthropic 在 API 侧对 system prompt 做了基于 Blake2b 哈希的前缀缓存。静态部分跨用户/跨会话共享缓存，极大降低了每次 API 调用的 prompt token 成本。

   5.2 System Prompt 的代码级编程智慧

从 prompts.ts 源码中提取的关键 prompt engineering 技巧：

   5.2.1 极简主义编码风格指令

```
❌ "Don't add features, refactor code, or make improvements beyond what was asked"❌ "Don't add error handling for scenarios that can't happen"❌ "Don't create helpers or abstractions for one-time operations"❌ "Three similar lines of code is better than a premature abstraction"
```

这四条规则反映了 Anthropic 团队从实践中发现的核心问题：LLM 天然倾向于过度工程化（over-engineering）。Claude Code 通过 prompt 约束来对抗这个倾向。

   5.2.2 工具调用优先级指令

```
"Do NOT use Bash to run commands when a relevant dedicated tool is provided.""To read files use FileReadTool instead of cat, head, tail, or sed""To edit files use FileEditTool instead of sed or awk"
```

这不是偏好，是架构上的要求。使用专用工具而非 Bash，让权限系统和 UI 渲染工作正常，让用户能精确看到每个操作。

   5.2.3 内外有别的指令差异

```
// Anthropic 内部用户额外获得：if (process.env.USER_TYPE === 'ant') {  // 1. 更严格的注释规范  "Default to writing no comments. Only add one when the WHY is non-obvious"    // 2. 更强的完成验证要求  "Before reporting a task complete, verify it actually works: run the test"
```

这揭示了一个重要事实：外部用户看到的 Claude Code 可能不是最优版本。Anthropic 内部用户获得更严格的质量控制 prompt、更好的验证机制（独立 Verification Agent）、以及针对 model 弱点的专门对策（如 Capybara v8 的 29-30% 错误声明率对策）。

   5.2.4 安全行为雕塑

CYBER_RISK_INSTRUCTION 是整个 prompt 中的安全基石：

```
"Assist with authorized security testing, defensive security, CTF challenges""Refuse requests for destructive techniques, DoS attacks, mass targeting""Dual-use security tools require clear authorization context"
```

短短一段话，但经过 Safeguards 团队精心校准和评估。源码注释明确要求修改必须获得专门团队审批。

   5.2.5 行动安全框架

```
"Carefully consider the reversibility and blast radius of actions""The cost of pausing to confirm is low, while the cost of an unwanted action (lost work, unintended messages sent, deleted branches) can be very high"
```

设计哲学：量两次，切一次（Measure twice, cut once）。

将操作分为三类：

- 🟢 可自由执行：编辑文件、运行测试等本地可逆操作
- 🟡 需确认：git push、创建 PR、发消息到外部服务
- 🔴 高度警惕：rm -rf、force-push、删除分支、drop table

# 06

权限系统

四层防御纵深

![](./assets/17892859513780.9558672708722716.png)

关键特征：

- Hook 系统 允许用户编写 shell 脚本在工具执行前/后触发
- Pattern Matching 支持 Bash(git \*) 这样的模式匹配
- Session Memory 记住用户本次会话中的授权决定
- YOLO 模式 (bypass permissions) 可以跳过所有检查，但有独立的安全分类器来兜底

# 07

上下文管理

三重压缩策略

```
Context Window 预算分配:┌─────────────────────────────────────────────────────┐│  System Prompt (tools, permissions, CLAUDE.md)       ││  ═══════════════════════════════════════════════     ││                                                     ││  Conversation History                                ││  ┌─────────────────────────────────────────────┐    ││  │ [older messages → 压缩摘要]                  │    ││  │ ═══════════════════════════════════════════  │    ││  │ [compact_boundary marker]                    │    ││  │ ─────────────────────────────────────────── │    ││  │ [recent messages — 完整保真度]               │    ││  └─────────────────────────────────────────────┘    ││                                                     ││  Current Turn                                        │└─────────────────────────────────────────────────────┘
```

| **策略**            | **触发条件**         | **机制**                         | **Feature Flag**   |
| ------------------- | -------------------- | -------------------------------- | ------------------ |
| **autoCompact**     | token count 超过阈值 | 调用 Claude API 对旧消息生成摘要 | 默认启用           |
| **snipCompact**     | 僵尸消息堆积         | 移除无效消息和过期标记           | `HISTORY_SNIP`     |
| **contextCollapse** | 上下文结构性低效     | 重构上下文组织方式               | `CONTEXT_COLLAPSE` |

对用户的认知： System prompt 告诉模型 "The conversation has unlimited context through automatic summarization" — 用户和模型都认为对话没有长度限制，但实际上是三层压缩在透明地工作。

Function Result Clearing (FRC)

```
// 只保留最近 N 个工具结果，更早的被自动清除`Old tool results will be automatically cleared from context to free up space.  The ${config.keepRecent} most recent results are always kept.`
```

系统 prompt 还额外提醒模型：

```
"Write down any important information you might need later in your response, as the original tool result may be cleared later."
```

这是一个精妙的设计——让模型在工具结果还新鲜时主动"记笔记"，防止信息在压缩时丢失。

# 08

子代理与多代理架构

![](./assets/17892859513890.21373035340181634.png)

四种 Spawn 模式

| **模式**     | **实现**            | **隔离级别**                  | **使用场景**         |
| ------------ | ------------------- | ----------------------------- | -------------------- |
| **default**  | 同进程              | messages[] 共享               | 简单委派             |
| **fork**     | 独立进程            | 全新 messages[]，共享文件缓存 | 研究性任务、多步实现 |
| **worktree** | Git Worktree + fork | 独立目录 + 全新消息           | 并行修改不同功能     |
| **remote**   | Bridge to Container | 完全隔离                      | Claude Code Remote   |

Fork Agent 的精妙之处

```
// 源码注释中的设计倡导："creates a fork, which runs in the background and keeps its tool output out of your context — so you can keep chatting with the user while it works"
```

核心价值：Fork 代理的工具输出不会回到主代理的上下文窗口。这解决了 Agent 系统中最常见的问题——上下文污染（context pollution）。主代理可以继续与用户对话，而 Fork 代理在后台完成研究或实现工作。

Swarm 模式（feature-gated）

```
Lead Agent  ├── Teammate A ──→ claims Task 1  ├── Teammate B ──→ claims Task 2  └── Teammate C ──→ claims Task 3
共享: task board, message inbox隔离: messages[], file cache, cwd
```

特征：

- 自主认领：队友扫描任务板，自动认领可用任务
- Send Message Protocol：统一的请求-响应模式驱动所有协商
- AsyncLocalStorage：在共享进程中实现每代理上下文隔离

# 09

Skills 系统

   9.1 按需注入 vs 预加载

Claude Code 的 Skills 系统最根本的设计决策是：通过 tool_result 注入知识，而不是通过 system prompt 预加载。

```
传统方式（大多数竞品）:System Prompt = Base Prompt +ALL Skills +ALL Memory→ 每次 API 调用都携带全部知识 → 浪费 token、占据上下文
Claude Code 方式:System Prompt = Base Prompt +Dynamic Boundary + Env InfoSkills = SkillTool.call() → tool_result → 当需要时注入Memory = CLAUDE.md → 按目录惰性加载→ 知识只在需要时出现在上下文中
```

   9.2 CLAUDE.md 惰性记忆系统

```
~/.claude/CLAUDE.md                  ← 全局级别<project-root>/CLAUDE.md             ← 项目级别<project-root>/.claude/CLAUDE.md     ← 项目配置级别<current-dir>/CLAUDE.md              ← 目录级别
```

这些文件在 loadMemoryPrompt() 时被加载到 system prompt 的动态区域，但只在当前工作目录相关时才被加载。

   9.3 Experimental Skill Search

108 个缺失模块中最引人注目的是 EXPERIMENTAL_SKILL_SEARCH 系列：

| **模块**                           | **功能**       |
| ---------------------------------- | -------------- |
| `skillSearch/featureCheck.js`      | 特征检查       |
| `skillSearch/remoteSkillLoader.js` | 远程技能加载器 |
| `skillSearch/remoteSkillState.js`  | 远程技能状态   |
| `skillSearch/localSearch.js`       | 本地技能搜索   |
| `skillSearch/prefetch.js`          | 技能预取       |
| `skillSearch/telemetry.js`         | 技能搜索遥测   |

系统 prompt 中的相关指令：

```
"Relevant skills are automatically surfaced each turn as 'Skills relevant to your task:' reminders. If you're about to do something those don't cover, call DiscoverSkillsTool with a specific description."
```

这表明 Anthropic 正在构建一个类似 "App Store for Agent Skills" 的系统——远程技能可以被动态发现、加载和执行。

# 10

会话持久化与恢复

```
~/.claude/projects/<hash>/sessions/└── <session-id>.jsonl           ← append-only 日志    ├── {"type":"user",...}    ├── {"type":"assistant",...}    ├── {"type":"progress",...}    └── {"type":"system","subtype":"compact_boundary",...}
```

持久化策略

| **消息类型**       | **写入方式**                         | **原因** |
| ------------------ | ------------------------------------ | -------- |
| User messages      | `await write` (阻塞)                 | 崩溃恢复 |
| Assistant messages | fire-and-forget (非阻塞，保序队列)   | 性能     |
| Progress           | inline write                         | 去重     |
| Flush              | on result yield / cowork eager flush | 一致性   |

恢复机制

```
--continue    → 恢复当前目录的最后一个会话--resume <id> → 恢复指定会话--fork-session → 新 ID，复制历史
```

# 11

MCP 集成

```
MCPConnectionManager  ├── Server Discovery (来自 settings.json)  │     ├── stdio  → 生成子进程  │     ├── sse    → HTTP EventSource  │     ├── http   → Streamable HTTP  │     ├── ws     → WebSocket  │     └── sdk    → 进程内传输  │  ├── Authentication  │     ├── OAuth 2.0 flow  │     ├── Cross-App Access (XAA / SEP-990)  │     └── API key via headers  │  └── Tool Registration        ├── mcp__<server>__<tool> 命名约定        ├── 来自 MCP 服务器的动态 schema        ├── 权限穿透到 Claude Code        └── 资源列表 (ListMcpResourcesTool)
```

MCP 工具与内置工具在权限系统、渲染系统中完全统一处理——对用户来说，MCP 工具和内置工具没有区别。

# 12

编译时特征门控

```
Anthropic 内部 Monorepo              发布的 npm 包══════════════════════               ═════════════════feature('DAEMON') → true  ──build──→  feature('DAEMON') → false    ↓                                     ↓daemon/main.js  ← INCLUDED  ──bundle──→  daemon/main.js  ← DELETED (DCE)
```

Bun 的 feature() 是编译时内在函数（compile-time intrinsic）：

- 内部构建：true → 代码保留在 bundle 中
- 外部构建：false → Dead Code Elimination 移除

已确认的 Feature Flags

```
COORDINATOR_MODE         → 多代理协调器HISTORY_SNIP             → 激进的历史裁剪CONTEXT_COLLAPSE         → 上下文重构DAEMON                   → 后台守护进程KAIROS                   → 推送通知、文件发送、自主模式PROACTIVE                → 主动行为VOICE_MODE               → 语音输入/输出WEB_BROWSER_TOOL         → 浏览器自动化EXPERIMENTAL_SKILL_SEARCH → 技能发现CACHED_MICROCOMPACT      → 缓存微压缩FORK_SUBAGENT            → Fork 子代理VERIFICATION_AGENT       → 验证代理TOKEN_BUDGET             → Token 预算控制
```

运行时门控

```
process.env.USER_TYPE === 'ant'  → Anthropic 内部员工功能GrowthBook feature flags         → A/B 实验
```

GrowthBook flags 使用随机词对来混淆目的（如 tengu_frond_boric、tengu_hive_evidence、tengu_onyx_plover），防止外部人员从 flag 名称推断功能。

# 13

未来路线图

从源码中提取的未来发展方向：

   13.1 新模型

| **代号**               | **对应**         | **状态**                                |
| ---------------------- | ---------------- | --------------------------------------- |
| **Numbat（袋食蚁兽）** | 下一代模型       | 确认（注释中明确提及）                  |
| **Opus 4.7**           | 高端模型         | 开发中                                  |
| **Sonnet 4.8**         | 平衡模型         | 开发中                                  |
| Capybara v8            | 当前 Sonnet 版本 | 已服役，有已知问题（29-30% 错误声明率） |
| Fennec                 | Opus 4.6         | 已服役                                  |

   13.2 KAIROS — 全自主代理模式

这是最大的未发布特性。从 System Prompt 源码：

```
你正在自主运行。你会收到 <tick> 提示让你保持活跃 — 把它们当作"你醒了，现在要做什么？"多个 tick 可能被批量放在一条消息中。这很正常——只处理最新的一个。
## 节奏控制使用 SleepTool 控制等待时间。每次唤醒花费一次 API 调用，但 prompt cache 在 5 分钟不活动后过期 — 需要平衡。
## 偏向行动Reading files, searching code, making code changes, committing，不需要询问。如果两个合理方案举棋不定，选一个就行。随时可以纠正方向。
## 终端焦点- 未聚焦: 用户离开了 → 大幅倾向自主行动- 聚焦: 用户在看 → 更协作
```

   13.3 未上线但已实现的功能

| **功能**            | **描述**                                                     |
| ------------------- | ------------------------------------------------------------ |
| **Voice Mode**      | Push-to-talk 语音输入，连接 Anthropic voice_stream WebSocket |
| **WebBrowserTool**  | 内置浏览器自动化（代号: bagel）                              |
| **Buddy System**    | 虚拟宠物伙伴，18 个物种、5 档稀有度、7 种帽子                |
| **Dream Task**      | 后台记忆整固子代理，AI 空闲时处理和整理记忆                  |
| **Workflow Tool**   | 执行预定义工作流脚本                                         |
| **Undercover Mode** | 在公开 repo 中隐藏 AI 身份                                   |

# 14

关键设计模式与工程哲学总结

设计模式一览

| **模式**                     | **实现位置**                       | **目的**                      |
| ---------------------------- | ---------------------------------- | ----------------------------- |
| **AsyncGenerator Streaming** | `QueryEngine`, `query()`           | API 到消费端的全链路流式传输  |
| **Builder + Factory**        | `buildTool()`                      | 工具定义的安全默认值          |
| **Branded Types**            | `SystemPrompt`, `asSystemPrompt()` | 防止 string/array 混淆        |
| **Feature Flags + DCE**      | `feature()` from `bun:bundle`      | 编译时 dead code elimination  |
| **Discriminated Unions**     | `Message` types                    | 类型安全的消息处理            |
| **Observer + State Machine** | `StreamingToolExecutor`            | 工具执行生命周期追踪          |
| **Snapshot State**           | `FileHistoryState`                 | 文件操作的 undo/redo          |
| **Ring Buffer**              | Error log                          | 长会话的有界内存              |
| **Fire-and-Forget Write**    | `recordTranscript()`               | 非阻塞持久化 + 保序           |
| **Lazy Schema**              | `lazySchema()`                     | Zod schema 的延迟求值提升性能 |
| **Context Isolation**        | `AsyncLocalStorage`                | 共享进程中的每代理上下文隔离  |

工程哲学

```
1. "One Loop & Bash is all you need"   → 核心循环极简，复杂性推到工具层和包装层
2. "Adding a tool = adding one handler"   → Open-Closed Principle 的 Agent 实践
3. "An agent without a plan drifts"   → Planning 翻倍完成率
4. "Load knowledge when you need it"   → 按需注入 > 预加载
5. "Context fills up; make room"   → 三层透明压缩，用户无感知
6. "Measure twice, cut once"   → 行动安全框架，可逆性优先
7. "Fail closed, not open"   → buildTool 默认值总是选择安全的一侧
```

# 15

对我们的启示

   15.1 如果你在构建 AI Coding Agent

| **优先级** | **行动**                           | **Claude Code 启发**         |
| ---------- | ---------------------------------- | ---------------------------- |
| **P0**     | 实现基础 Agent Loop + Bash Tool    | 这就是最小可行产品           |
| **P0**     | 添加 Planning 机制                 | 完成率翻倍的"免费午餐"       |
| **P1**     | 实现 Sub-Agent + Context Isolation | 避免上下文污染               |
| **P1**     | 实现三层权限系统                   | rules + hooks + interactive  |
| **P1**     | 按需注入 Skills，不要预加载        | 节省 token，提升相关性       |
| **P2**     | 实现 Context Compression           | auto-compact 为核心          |
| **P2**     | System Prompt 分层缓存             | 静态/动态分界 → cache 命中率 |
| **P3**     | Task Persistence                   | JSONL append-only 日志       |
| **P3**     | Multi-Agent / Swarm                | 只在单代理证明不够时         |

   15.2 Claude Code 的三级跳

```
Level 1: CLI Tool      → 一个 loop + bash + 几个工具Level 2: Agent Product  → 权限 + 压缩 + 持久化 + 子代理Level 3: Autonomous Platform → KAIROS + Voice + Skill Market + Swarm
```

Claude Code 正在从一个编程助手进化为一个全天候自主开发代理。

108个缺失模块（internal-only）意味着我们看到的只是冰山一角。真正的Anthropic内部版本有守护进程、协调器、主动通知、语音模式、远程技能搜索等完整能力。外部发布版本是经过大量feature-gating后的精简版。

-End-
