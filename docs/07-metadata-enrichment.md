# 07 - 元数据补充

## 问题
- CORE 来源论文大量缺少 citation_count、year、venue
- 导致评分不准确（authority 和 recency 维度）
- 影响报告质量（无法提供年份和引用数）

## 方案

### 1. 后置元数据补充
论文入库后，异步查询 S2/OpenAlex 补充元数据：
- 用 DOI 查 S2 获取 citation_count、venue、year
- 用标题查 OpenAlex 获取 citation_count、venue

### 2. 评分前补充
在 `_score_papers` 之前，检查论文元数据是否完整，不完整的批量补充：
```python
async def _enrich_paper_metadata(db, papers):
    """Enrich papers missing citation_count/year/venue via S2/OpenAlex."""
    for paper in papers:
        if paper.citation_count == 0 and paper.year is None:
            # Query S2 by DOI or title
            ...
```

### 3. 来源优先级
论文元数据以 S2 > OpenAlex > Crossref > CORE 为优先级合并。

## 涉及文件
- `backend/app/services/scoring_service.py` — 添加 enrich 函数
- `backend/app/paper_sources/semantic_scholar.py` — 添加 by_doi / by_title 查询
- `backend/app/agent/runner.py` — 评分前调用 enrich

## 验证
- CORE 来源论文的 citation_count 填充率 > 50%
- 高优先级论文中 citation_count > 0 的比例提升
