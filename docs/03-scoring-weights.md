# 03 - 评分权重调整

## 问题
- `final_score = 0.35*relevance + 0.20*authority + 0.15*recency + 0.15*novelty + 0.15*idea_potential`
- relevance 占 35%，authority 仅 20%
- citation_count 缺失时（=0），LLM 无法正确评估 authority
- 导致高相关但低质量的论文可能排高位

## 方案

### 1. 调整权重
```python
final_score = 0.30*relevance + 0.25*authority + 0.15*recency + 0.15*novelty + 0.15*idea_potential
```

### 2. citation 缺失降权
当 citation_count = 0 且 year = None 时，authority 乘以 0.7 惩罚系数：
```python
if paper.citation_count == 0 and paper.year is None:
    authority_adjusted = score.authority * 0.7
else:
    authority_adjusted = score.authority
```

### 3. venue 加成
预定义顶会/顶刊列表，venue 匹配时 authority 加 0.1：
```python
TOP_VENUES = ["ICML", "NeurIPS", "ICLR", "CVPR", "ACL", "EMNLP", 
              "AAAI", "IJCAI", "KDD", "WWW", "SIGIR", "TSE", "TACL"...]
```

## 涉及文件
- `backend/app/agent/runner.py` — `_score_papers` 修改 final_score 计算

## 验证
- 高优先级论文中 citation_count > 50 的比例应提升
