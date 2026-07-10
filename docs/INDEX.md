# Deep Research 项目改进索引

> 基于 2026-07-07 实测数据分析制定，2026-07-08 更新（新增文献地图设计）。按优先级排列，逐项实现。

## 目录

| 编号 | 文档 | 优先级 | 状态 | 涉及文件 |
|------|------|--------|------|----------|
| 01 | [检索源重试与退避](01-retrieval-retry.md) | P0 | **已实现** | `paper_sources/*.py`, `search_service.py` |
| 02 | [Ideas 来源追溯](02-ideas-traceability.md) | P0 | **已实现** | `prompts.py`, `runner.py`, `schemas.py`, `models.py` |
| 03 | [评分权重调整](03-scoring-weights.md) | P1 | **已实现** | `runner.py` |
| 04 | [报告引用追溯](04-report-citations.md) | P1 | **已实现** | `prompts.py`, `runner.py` |
| 05 | [幻觉防护（prompt层）](05-hallucination-guard.md) | P1 | **已实现** | `prompts.py`, `runner.py` |
| 06 | [Gap 分析改进](06-gap-analysis.md) | P2 | 待实现 | `prompts.py`, `runner.py` |
| 07 | [元数据补充](07-metadata-enrichment.md) | P2 | 待实现 | `paper_sources/*.py`, `scoring_service.py` |
| 08 | [PostgreSQL 迁移](08-postgres-migration.md) | P3 | 待实现 | `session.py`, `config.py`, `.env` |
| 09 | [架构升级](09-architecture-upgrade.md) | — | **P0已实现**，P1-A 详见 10 | `runner.py`, `prompts.py`, `schemas.py` |
| **10** | [**RAG 全文检索管线**](10-rag-fulltext-pipeline.md) | **P1** | **已实现** | `rag_service.py`, `runner.py`, `models.py`, `prompts.py` |
| **11** | [**LLM Wiki 知识编译**](11-llm-wiki.md) | **P2** | **已实现** | `wiki_service.py`(新建), `runner.py`, `models.py`, `schemas.py`, `prompts.py` |
| **12** | [**文献地图**](12-literature-map.md) | **P1** | **设计文档** | `literature_map_service.py`(新建), `maps.py`(新建), `runner.py`, `models.py`, 前端 `LiteratureMapView.tsx` |

## 实现顺序

```
已完成:
  P0: 01 → 02                    （检索质量 + ideas 可追溯）
  P1: 03 → 04 → 05               （评分 + 报告 + 幻觉防护 prompt层）
  09-P0: 自反馈迭代 + 新颖性检查 + idea质量5层改进

下一步:
  P1: 10（RAG全文检索 — 幻觉根因修复，ChromaDB向量数据库，为LLM Wiki提供素材）✅
  P1: 12（文献地图 — 引文网络+语义聚类双层可视化，复用doc 10 embedding，用户可见价值高）
  P2: 11（LLM Wiki知识编译 — 增量编译wiki页面替代GraphRAG，预编译交叉引用+矛盾检测）✅
  P2: 06 → 07                    （gap + 元数据，体验优化）
  P2: 09-P1B（多视角提问搜索，参考STORM）
  P2: 09-P2A（交叉编码器重排序）
  P3: 08                         （PostgreSQL迁移，可延后）
```

## 问题概览

| 阶段 | 问题 | 影响 | 解决方案 | 状态 |
|------|------|------|----------|------|
| 检索 | S2/OpenAlex 429 限流直接跳过 | 缺少高引用经典论文 | 01 指数退避重试 | ✅ |
| 检索 | CORE 来源元数据缺失 | 评分不准 | 07 元数据补充 | 待实现 |
| 评分 | authority 权重过低 | 低质量论文排高位 | 03 权重调整 | ✅ |
| 评分 | method_extract 仅从摘要提取 | 方法信息不准/幻觉 | **10 RAG全文检索** | 待实现 |
| 报告 | prompt 只传 title+year | 无法溯源 | 04 编号引用+DOI | ✅ |
| Ideas | related_paper_titles 存标题不存 ID | 无法追溯 | 02 改存 paper_id | ✅ |
| 幻觉 | LLM 从有限摘要信息推断方法 | 编造模型/数据集 | 05 prompt约束 + **10 RAG根因修复** | 05✅ 10待实现 |
| Idea质量 | 概念模糊、编造数据集 | idea无价值 | 5层改进 + **10 RAG提供真实段落** | 5层✅ 10待实现 |
| Gap | Round 2 gaps 与 Round 1 相同 | 搜索无方向改进 | 06 gap分析改进 | 待实现 |
| 可视化 | 论文列表平铺，无关系视图 | 用户无法全局把握领域结构 | **12 文献地图**（引文网络+语义聚类） | 设计文档 |

## 关键决策记录

### 2026-07-07: RAG vs 纯摘要提取
- **问题**：method_extract 从摘要（~200词）提取模型/算法/数据集，摘要通常不含 Method 细节
- **调研**：OpenScholar 用全文段落检索，幻觉率 0%（GPT-4o 无检索 78-90%）
- **决策**：实现 RAG 管线（PDF下载→解析→embedding→检索），仅对高优先级论文，无PDF降级用摘要
- **规模**：15-30篇论文 × ~100段落 = 1500-3000 chunks
- **详见**：[10-rag-fulltext-pipeline.md](10-rag-fulltext-pipeline.md)

### 2026-07-07: 向量数据库选型 — ChromaDB
- **问题**：numpy余弦相似度不支持GraphRAG扩展，无法持久化
- **选型**：ChromaDB（嵌入式，零配置，metadata过滤，持久化）
- **原因**：为GraphRAG预留实体/关系collection扩展；嵌入式与SQLite理念一致；原生metadata过滤
- **embedding模型**：BAAI/bge-base-en-v1.5（768维，质量最佳，GraphRAG兼容）
- **详见**：[10-rag-fulltext-pipeline.md](10-rag-fulltext-pipeline.md)

### 2026-07-07: GraphRAG → LLM Wiki 路线变更
- **原方案**：GraphRAG（实体/关系抽取→社区检测→Local/Global/Hybrid Search）
- **问题**：~220次LLM调用、社区检测不可解释、矛盾检测困难、知识不累积
- **新方案**：LLM Wiki（Karpathy 范式）— LLM 作为 wiki 编辑器，增量编译结构化 Markdown 页面
- **核心优势**：知识累积（越用越丰富）、交叉引用预编译、矛盾自动检测、人类可读、LLM调用仅4-7次
- **页面类型**：concept（替代聚类）、method、dataset、model、synthesis（跨主题分析）
- **详见**：[11-llm-wiki.md](11-llm-wiki.md)

### 2026-07-08: 文献地图设计
- **问题**：论文列表平铺，无关系视图；LLM聚类不持久化；用户无法全局把握领域结构
- **调研**：Connected Papers（共引+书目耦合）、Inciteful（PageRank+Adamic/Adar+Salton）、Open Knowledge Maps（语义聚类）、ResearchRabbit（推荐+引文网络）
- **方案**：双层关系模型（引文网络层 + 语义相似层）→ Louvain社区检测 → LLM命名聚类 → 力导向图可视化
- **关键决策**：引文相似度用Adamic/Adar+Salton（参考Inciteful）；语义相似度复用ChromaDB embedding（复用doc 10）；社区检测用Louvain（与doc 11 GraphRAG共用）
- **维护**：增量更新（每轮检索后语义边增量）+ 全量重建（报告生成前引文+聚类）；版本管理+引文缓存
- **与GraphRAG关系**：文献地图是GraphRAG的论文级简化版，先落地，GraphRAG升级时聚类层可替换
- **详见**：[12-literature-map.md](12-literature-map.md)
