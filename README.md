# LangGraph Agent Learning Curriculum

> A hands-on, progressive LangGraph learning path grounded in the official docs, designed for engineers who want to **understand how it works** rather than just how to call it.

## 快速开始 / Quick start

```bash
# Clone and setup
git clone <this-repo>
cd agent-learning
poetry install

# Run lesson 1
poetry run python lessons/l1_state_node_edge/main.py

# Run tests
poetry run pytest lessons/l1_state_node_edge -v
```

## 设计理念 / Philosophy

Three principles shape this curriculum:

1. **Core primitives first**: State, Node, Edge are the foundation. Everything else (loops, checkpoints, interrupts, subgraphs) is a combination of these three.
2. **Progressive dependency**: Each lesson builds on the previous one's mechanics. You can't understand interrupts without checkpoints, and you can't understand checkpoints without edges.
3. **Verify, don't trust**: Every lesson has runnable code + tests that *prove* the behavior by printing raw runtime output. No API key needed. Every README cross-references the official docs with page links.

## 课程地图 / Curriculum

| Lesson | Topic | Key concept | Official docs |
|--------|-------|-------------|---------------|
| **L1** | State / Node / Edge | Super-step execution model, reducer, fan-out trigger rule | [Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api) |
| **L2** | Control flow | Conditional edges, loops, `Command`, `Send` | [Create loops](https://docs.langchain.com/oss/python/langgraph/use-graph-api#create-and-control-loops) |
| **L3** | Persistence | Checkpointers, `thread_id`, time travel, durability modes | [Persistence](https://docs.langchain.com/oss/python/langgraph/persistence) |
| **L4** | Interrupt | Human-in-the-loop, replay semantics, `Command(resume=...)` | [Interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts) |
| **L5** | Subgraph | Shared/separate state, `Command.PARENT`, checkpoint namespaces | [Use subgraphs](https://docs.langchain.com/oss/python/langgraph/use-subgraphs) |
| **L6** | Capstone | Full agent with approval + crash recovery + rollback | [Graph API overview](https://docs.langchain.com/oss/python/langgraph/graph-api) |

Full syllabus (with Chinese commentary): [`docs/curriculum.md`](./docs/curriculum.md)

## 学完 L1 你会知道 / After L1 you will know

- Why concurrent writes to a no-reducer channel raise `InvalidUpdateError`
- Why two nodes in the same super-step cannot see each other's writes
- Why an asymmetric fan-out (one short path, one long path) causes a join node to fire **twice**
- How LangGraph schedules nodes: the **trigger rule** (OR over incoming edges)
- Why private schemas leak during `stream(mode="values")` and how to contain them

Run it:

```bash
poetry run python lessons/l1_state_node_edge/main.py
poetry run pytest lessons/l1_state_node_edge -v
```

## 项目状态 / Status

- ✅ **L1 (State, Node, Edge)**: Complete with 7 tests, full trace output, and "break it" exercises
- ⏳ **L2 (Control flow)**: Planned
- ⏳ **L3 (Persistence)**: Planned
- ⏳ **L4 (Interrupt)**: Planned
- ⏳ **L5 (Subgraph)**: Planned
- ⏳ **L6 (Capstone)**: Planned

## 贡献指南 / Contributing

Each lesson follows a three-part pattern:

1. **Runnable minimal code** (`main.py`) that prints raw runtime output
2. **Test suite** (`test_main.py`) that asserts the behavior with pytest
3. **README** explaining "why this design" + common pitfalls + links to official docs

All lessons are deterministic and run without API keys. Submit PRs that keep this pattern.

## 依赖 / Requirements

- Python ≥3.12
- `langgraph` ≥1.2.11
- `langgraph-checkpoint-sqlite` (for L3 persistence)

## 证书 / License

MIT

---

**Acknowledgments**: This curriculum is built on the [official LangGraph docs](https://docs.langchain.com/oss/python/langgraph/). Every lesson cross-references the corresponding official page. If code and docs disagree, trust the code and file an issue upstream.
