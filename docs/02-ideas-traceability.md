# 02 - Ideas 来源追溯

## 问题
- `related_paper_titles` 只存论文标题字符串，不存 ID
- 标题可能被 LLM 改写或截断，无法匹配数据库记录
- 无法从 idea 跳转到具体论文
- 可能引用不存在的论文（幻觉）

## 方案

### 1. Schema 改为存论文 ID
```python
# schemas.py - IdeaItem
related_paper_ids: list[str]  # 论文 ID 列表
related_paper_titles: list[str]  # 保留标题用于显示
```

### 2. Prompt 传入论文 ID + 标题
```python
# prompts.py - IDEAS_USER
High-priority papers (use paper_id for related_paper_ids):
[paper_id] Title
[paper_id] Title
...
```

### 3. 生成后验证
LLM 生成 ideas 后，检查 `related_paper_ids` 是否都在提供的论文列表中。不在的剔除并记录 warning。

### 4. 前端展示
IdeasView 中点击 related paper 可跳转到论文详情。

## 涉及文件
- `backend/app/schemas/schemas.py` — IdeaItem 添加 related_paper_ids
- `backend/app/db/models.py` — ResearchIdea 添加 related_paper_ids_json 字段
- `backend/app/agent/prompts.py` — IDEAS_USER 传入论文 ID
- `backend/app/agent/runner.py` — _generate_and_score_ideas 传 ID、验证
- `frontend/src/components/IdeasView.tsx` — 展示关联论文
- `frontend/src/types/index.ts` — Idea 类型添加 related_paper_ids

## 验证
- Ideas 的 related_paper_ids 全部存在于数据库
- 前端可点击跳转
