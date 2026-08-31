# Deep Research 项目文档索引

> 2026-08-31 文档整理：删除已废弃的 V1 改进系列（01-09）、V1 实测报告（14）、已执行完毕的中间计划（16/17）、refactor 中已完成阶段的审计/计划/进度文档（00/02/03/04/PROGRESS），以及根目录一次性 E2E 调试脚本。处置理由见文末。

## 现存文档

| 文档 | 性质 | 状态 | 说明 |
|------|------|------|------|
| [README.md](../README.md) | 主设计文档 | 与代码对齐 | 系统架构、链路、运行方式的权威来源 |
| [refactor/01_target_architecture.md](refactor/01_target_architecture.md) | 设计蓝图 | 目标态 | V2「Evidence-grounded Research Gap Discovery and Idea Validation System」核心理念、目标主流程与数据模型；实际实现以 README/代码为准 |
| [10-rag-fulltext-pipeline.md](10-rag-fulltext-pipeline.md) | 设计+决策记录 | 已实现 | RAG 全文检索管线：PDF→解析切分→embedding→ChromaDB；含 OpenScholar/STORM 调研与 ChromaDB 选型决策 |
| [11-llm-wiki.md](11-llm-wiki.md) | 设计+决策记录 | 已实现 | LLM Wiki 知识编译：放弃 GraphRAG 改用 Karpathy LLM Wiki 范式的完整决策对比 |
| [12-literature-map.md](12-literature-map.md) | 设计文档 | 待实现（roadmap） | 文献地图：引文网络+语义聚类双视角，业界方案调研与算法选型（Adamic/Adar + Louvain） |
| [15_从研究方向到Idea的难点分析与优化建议.md](15_从研究方向到Idea的难点分析与优化建议.md) | 探索性分析 | 诊断框架仍有效 | V2 长链路 7 道硬闸门的系统性诊断：为何"研究方向→Idea"难。其中 O3-O9 缓解已落地，核心张力（闸门串联 × 上游供给 × 分级产出）仍是主线 |

## 2026-08-31 整理处置记录

- **01-09（V1 改进系列）**：V1 链路已被 V2 整体取代。已实现部分（01 检索重试→现为源级 cooldown+TokenBucket、02-05 随 V1 废弃、09-P0）实现细节以代码为准；未实现部分（06 gap 分析 / 07 元数据 / 08 PostgreSQL 迁移）方案与 V2 代码不匹配，需要时重写成本很低。
- **14（V1 链路优化分析报告，2026-07-20）**：V1 终局实证（GNN 任务 633 篇、5 ideas 全 reject），是 V2 重构的动因记录，结论已沉淀于 refactor/01 与 15。
- **16 / 17（2026-08-05 优化计划 / 现存不足）**：文中八项差距与缺陷 1-7 项均已实施完毕（17 号文末自述"第 1-7 项已全部实施完成"），属已完成的中期计划，保留会误导读者以为仍是未修项。
- **refactor/00 / 02 / 03 / 04 / PROGRESS**：重构前架构审计、数据库迁移/实施计划、阶段审查与进度清单。Phase 0-3A 已全部完成（PROGRESS 中 3B/3C 标注"待实施"严重过时——mine_gaps/audit_gaps 实际早已上线并经 run8-12 多轮 E2E 验证）。
