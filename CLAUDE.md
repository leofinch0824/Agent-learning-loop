# agent-learning

LangGraph 教学项目：把官方文档拆成可跑、可验证的分阶课程。

## 项目地图

```
docs/curriculum.md    # 学习路线图（L1-L6）：每课的目标、踩坑点、对应官方文档
docs/DESIGN.md        # 设计原则、目录约定、新课件怎么加
lib/helpers.py        # 教学辅助（trace / print_state / print_history / build_checkpointer）
lessons/lN_*/         # 每课三件套：README.md + main.py + test_main.py
```

## 官方文档索引（本地）

`llms.txt` 是 LangChain 官方文档的全量索引（176 个 section，含 43 页 LangGraph Python 文档），**已在 .gitignore 中，不入库但保存在本地**。

查文档时优先读它，别猜 URL：

```bash
# 找某主题的页面
grep -n 'langgraph' llms.txt

# 拉取某个 section 的完整页面列表（比本地 grep 更全）
curl -s https://docs.langchain.com/oss/python/langgraph/llms.txt
```

页面 URL 规律：`https://docs.langchain.com/oss/python/langgraph/<page>.md`（追加 `.md` 拿到 Markdown 原文）。带链接的课程映射表在 `docs/curriculum.md`。

## 命令

```bash
poetry install
poetry run python lessons/l1_state_node_edge/main.py   # 跑单课，打印完整执行过程
poetry run pytest lessons/ -v                          # 全部测试
```

## 约定

- 每课确定性、无 API key、无网络：假模型函数代替真实 LLM
- `main.py` 用 `lib.trace()` 打印原始 chunk，不做摘要
- 每个踩坑点在 `test_main.py` 里有对应断言
- 新增课程后更新 `docs/curriculum.md` 的进度与文档映射
