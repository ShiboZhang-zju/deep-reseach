# 01 - 检索源重试与退避

## 问题
- Semantic Scholar / OpenAlex 429 限流后直接跳过，不重试
- GNN 任务中 S2 和 OpenAlex 贡献 0 篇论文，导致缺少高引用经典论文
- Crossref 也频繁 429

## 方案

### 1. 指数退避重试
每个 paper source 在收到 429 时，按 `1s → 2s → 4s` 退避重试，最多 3 次。

### 2. 请求间隔
各源维护最小请求间隔，避免短时间内密集请求：
- Semantic Scholar: 1s/req
- OpenAlex: 0.5s/req
- Crossref: 1s/req

### 3. 结果缓存
对相同 query 的搜索结果做内存缓存（TTL 1 小时），避免重复请求。

## 涉及文件
- `backend/app/paper_sources/base.py` — 基类添加重试逻辑
- `backend/app/paper_sources/semantic_scholar.py`
- `backend/app/paper_sources/openalex.py`
- `backend/app/paper_sources/crossref.py`
- `backend/app/services/search_service.py` — 添加缓存层

## 验证
- GNN 任务重新搜索，S2 和 OpenAlex 应贡献论文
- 高优先级论文中应出现 citation_count > 100 的论文
