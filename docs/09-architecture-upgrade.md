# 架构升级规划（基于业界对比）

> 参考 STORM、OpenScholar、GPT Researcher、AI-Scientist、Idea2Paper、KGGen 等开源项目
> 更新于 2026-07-07：P0 全部已实现，P1-A 升级为独立文档 [10-rag-fulltext-pipeline.md](10-rag-fulltext-pipeline.md)

## P0: 自反馈迭代 + 新颖性检查 ✅ 已实现

### P0-A: 报告自反馈迭代（参考 OpenScholar）✅
- 生成报告后 LLM 自评（完整性/组织/缺失信息）
- 根据反馈精炼，最多迭代3轮
- 文件: `runner.py` `_generate_report`, `prompts.py`（REPORT_FEEDBACK_SYSTEM/USER, REPORT_REFINE_SYSTEM/USER）

### P0-B: Ideas 新颖性检查（参考 AI-Scientist）✅
- 每个 idea 生成后，用标题+描述搜索已有论文
- LLM 判断是否已有类似工作
- 已有的降低 novelty 分数（惩罚 0.1）
- 文件: `runner.py` `_generate_and_score_ideas`, `prompts.py`（NOVELTY_CHECK_SYSTEM/USER）

### P0-C: Ideas 自反馈迭代 ✅
- 生成 ideas 后 LLM 自评（创新性/可行性/与论文关联度）
- 根据反馈重新生成不够好的 ideas，最多 3 轮
- 3 轮后分数仍 <0.70 则自动提升前 3 个（分数≥0.55）为 go
- 文件: `runner.py` `_generate_and_score_ideas`

### P0-D: Idea 质量 5 层改进 ✅
1. `method_extract`：评分时提取具体模型/算法/数据集
2. `technique_details`：聚类时提取精确技术细节
3. 强约束 prompt：描述必须以"基于[聚类X]的[具体方法]"开头
4. 评分惩罚：模糊方法 ≤0.5，编造概念 ≤0.3
5. 数据流：method_sketch 必须包含模型/算法/数据集/指标/基线

## P1: 段落级检索 + 多视角搜索

### P1-A: 段落级检索（参考 OpenScholar）→ 详见 [10-rag-fulltext-pipeline.md](10-rag-fulltext-pipeline.md)

**已升级为独立文档**，完整方案包含：
- PDF 下载（PyMuPDF）+ 全文解析 + 章节切分（250词/段）
- sentence-transformers embedding + SQLite 存储
- numpy 余弦相似度检索（无需向量数据库）
- 下游全链路集成：method_extract / 聚类 / 报告 / ideas / 引用验证
- 降级策略：无 PDF 用摘要，embedding 失败退回关键词检索

原方案（FAISS/ChromaDB）已废弃——我们的规模（1500-3000 chunks）不需要向量数据库。

### P1-B: 多视角提问驱动搜索（参考 STORM）
- 从不同专家视角生成搜索问题
- 每个视角独立搜索+综合
- 文件: `prompts.py`, `runner.py`

## P2: 交叉编码器重排序 + 引用验证

### P2-A: 交叉编码器重排序（替代 LLM 逐篇评分）
- 用 BGE-reranker 批量重排序
- 引用数加权
- 文件: `scoring_service.py`, `runner.py`

### P2-B: 引用验证（增强版）
- 生成后检查每个论断是否有引用支撑
- 缺失引用的补充
- **RAG 增强**：验证 idea 中的模型/数据集是否在检索段落中存在（详见 [10-rag-fulltext-pipeline.md](10-rag-fulltext-pipeline.md) 引用验证部分）
- 文件: `runner.py`, `prompts.py`, `rag_service.py`
