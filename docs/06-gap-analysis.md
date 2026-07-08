# 06 - Gap 分析改进

## 问题
- Round 2 的 gaps 与 Round 1 几乎相同
- Gap 分析没有有效指导下一轮搜索策略调整
- queries 生成没有基于 gap 做针对性变化

## 方案

### 1. Gap 去重
生成 gap 后与上一轮 gap 对比，相似度 > 0.7 的标记为 "unresolved"，要求 LLM 换角度：
```
ROUND_SUMMARY_SYSTEM 增加：
- If a gap is similar to a previous gap, mark it as "unresolved" and suggest a different search angle
- Focus on NEW gaps not covered by previous rounds
```

### 2. Gap 驱动 query 生成
QUERIES prompt 中明确要求针对 unresolved gaps 生成不同角度的 query：
```
QUERIES_USER 增加：
Unresolved gaps (need different search angles):
{unresolved_gaps}
Generate queries specifically targeting these gaps from NEW angles.
```

### 3. Gap 分类
将 gap 分为 "coverage gap"（缺少某子领域论文）和 "method gap"（缺少某类方法），指导不同搜索策略。

## 涉及文件
- `backend/app/agent/prompts.py` — ROUND_SUMMARY_SYSTEM/USER, QUERIES_SYSTEM/USER
- `backend/app/agent/runner.py` — 传 unresolved_gaps

## 验证
- Round 2 的 gaps 应与 Round 1 有显著差异
- Round 2 的 queries 应针对 Round 1 的 unresolved gaps
