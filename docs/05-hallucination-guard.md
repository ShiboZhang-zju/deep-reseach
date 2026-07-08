# 05 - 幻觉防护（prompt 层）

> **注意**：本文档是 prompt 层的防线（已实现）。幻觉的**根因修复**见 [10-rag-fulltext-pipeline.md](10-rag-fulltext-pipeline.md)——通过 PDF 全文检索提供真实段落，使 LLM 无需推断。

## 问题
- 报告/ideas 无约束 LLM 仅引用提供的论文，可能编造引用
- related_paper_titles 可能包含未检索到的论文标题
- 报告中的"关键趋势"可能基于 LLM 自身知识而非检索论文
- **根因**：method_extract 从摘要提取，摘要不含 Method 细节，LLM 被迫推断 → 幻觉

## 方案

### 1. Prompt 约束
所有涉及引用的 prompt 增加硬约束：
```
IMPORTANT: Only reference papers from the provided list. 
Do not invent or reference any paper not in the list.
If you need a paper not in the list, state "no relevant paper found" instead.
```

### 2. 生成后验证
- 报告：提取所有 [Px] 引用，验证编号在范围内
- Ideas：验证 related_paper_ids 全部在提供的论文列表中
- 不匹配的剔除并记录 warning

### 3. Ideas prompt 传入论文 ID 映射
```
Available papers (ONLY use these IDs for related_paper_ids):
P1: <paper_id> - Title
P2: <paper_id> - Title
...
```

## 涉及文件
- `backend/app/agent/prompts.py` — REPORT_SYSTEM, IDEAS_SYSTEM, IDEA_SCORE_SYSTEM
- `backend/app/agent/runner.py` — `_generate_report`, `_generate_and_score_ideas` 添加验证

## 验证
- 报告中无未提供的论文标题
- Ideas 的 related_paper_ids 全部有效
