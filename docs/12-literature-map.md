# 12 - 文献地图（Literature Map）设计与维护方案

> 将检索到的论文组织成可视化文献地图，支持引文网络 + 语义聚类双视角
> 状态：**设计文档**，待实现

## 一、背景与目标

### 当前痛点

Deep Research 当前已检索并存储论文（含评分、优先级、聚类），但：
1. **论文关系不可见**：论文列表是平铺的，无法看到论文间的引用关系和相似关系
2. **聚类不持久**：`_build_paper_clusters` 每次运行 Ideas 生成时临时调用 LLM 聚类，结果不持久化，不供前端展示
3. **缺少全局视图**：用户无法快速判断"这个方向有哪些流派""哪些论文是核心枢纽""哪些区域是研究空白"
4. **检索方向不可视**：gap-driven 检索的"方向感"无法直观体现

### 目标

构建一个 **文献地图**，将任务内所有论文组织为可视化图谱：
- **节点** = 论文，大小表示引用数/评分，颜色表示年份/优先级
- **边** = 论文间关系（引用/共引/语义相似）
- **聚类区域** = 研究主题分组
- **交互**：点击节点查看详情，拖拽布局，按维度筛选

---

## 二、业界方案调研

### 2.1 方案分类

业界文献地图工具分为两大流派：

| 流派 | 代表工具 | 核心方法 | 优势 | 劣势 |
|------|---------|---------|------|------|
| **引文网络派** | Connected Papers, Inciteful, ResearchRabbit, Litmaps | 共引分析 + 书目耦合 + PageRank | 能发现"使用不同术语但属于同一学术对话"的论文 | 需要引文数据，新论文可能无引用 |
| **语义聚类派** | Open Knowledge Maps | Embedding 语义相似度聚类 | 不依赖引文数据，适合新领域/交叉学科 | 可能遗漏结构居中但语义不同的论文 |

**最佳实践**（来自 Tesify 2026 对比文章）：先用语义聚类做主题定位，再用引文网络做深度探索。**我们的方案应同时支持两种视角。**

### 2.2 核心算法对比

| 工具 | 算法 | 数据需求 | 适合场景 |
|------|------|---------|---------|
| **Connected Papers** | 共引 + 书目耦合（具体权重未公开） | Semantic Scholar 引用数据 | 从单篇论文出发探索相关论文 |
| **Inciteful** | PageRank（重要性）+ Adamic/Adar（书目耦合）+ Salton 索引（共引） | Semantic Scholar + OpenAlex 引用数据 | 找桥接论文、核心论文 |
| **Open Knowledge Maps** | 语义相似度聚类（非关键词频率） | BASE/PubMed 元数据 + 摘要 | 关键词搜索 → 主题概览 |
| **ResearchRabbit** | 推荐算法（类 Spotify）+ 引文网络 | Semantic Scholar | 迭代式论文发现 |
| **Litmaps** | 时间轴 + 引文链追踪 | OpenAlex + Crossref | 领域演进可视化 |

### 2.3 Inciteful 技术细节（最值得参考）

Inciteful 将文献建模为 **有向无环图（DAG）**，从种子论文出发做两层深度搜索：

```
节点0: 种子论文
节点1: 种子论文引用的论文 + 引用种子论文的论文（一跳邻居）
节点2: 对节点1再做一跳搜索（两跳邻居）
```

三种核心算法：

1. **PageRank（重要性排名）**：不只看引用数，还看引用来源的重要性 → 找奠基性论文
2. **Adamic/Adar（书目耦合相似度）**：两篇论文共同引用的论文越多越相似，但共同引用的小众论文权重更高
   - `Score(a,b) = Σ 1/log(degree(w))`，w 为 a 和 b 共同引用的论文
3. **Salton 索引（共引相似度）**：两篇论文被同一组论文引用的余弦相似度
   - `Salton(u,v) = |Γ(u) ∩ Γ(v)| / √(|Γ(u)| × |Γ(v)|)`

---

## 三、方案设计

### 3.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                    文献地图构建管线                           │
│                                                             │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │ 论文入库  │───→│ 引文数据补充  │───→│ 关系图构建       │  │
│  │ (已有)    │    │ (新增)       │    │ (新增)           │  │
│  └──────────┘    └──────────────┘    └────────┬─────────┘  │
│                                               │             │
│                    ┌──────────────────────────┘             │
│                    ▼                                        │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │ 引文网络层   │    │ 语义相似层   │    │ 聚类层       │  │
│  │ (引文边)     │    │ (Embedding边)│    │ (社区检测)   │  │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘  │
│         │                   │                   │           │
│         └───────────────────┼───────────────────┘           │
│                             ▼                               │
│                    ┌──────────────┐                         │
│                    │ 文献地图存储  │                         │
│                    │ (SQLite)     │                         │
│                    └──────┬───────┘                         │
│                           │                                 │
│                           ▼                                 │
│                    ┌──────────────┐                         │
│                    │ 前端可视化    │                         │
│                    │ (力导向图)   │                         │
│                    └──────────────┘                         │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 双层关系模型

文献地图包含两层关系，用户可切换/叠加：

#### 层 1：引文网络层（Citation Network）

| 边类型 | 说明 | 来源 |
|--------|------|------|
| `cites` | 论文 A 引用论文 B（直接引用） | Semantic Scholar / OpenAlex API |
| `co-cited` | 论文 A 和 B 被同一篇论文引用（共引） | 从引用数据推导 |
| `biblio-coupled` | 论文 A 和 B 引用了相同的参考文献（书目耦合） | 从引用数据推导 |

#### 层 2：语义相似层（Semantic Similarity）

| 边类型 | 说明 | 来源 |
|--------|------|------|
| `semantic-similar` | 论文 A 和 B 的摘要/全文 embedding 余弦相似度 > 阈值 | ChromaDB（复用 RAG 管线的 embedding） |

### 3.3 引文数据获取

当前系统只存储论文元数据，**没有引用关系**。需要新增引文数据获取步骤：

```python
# 在 paper_sources 层新增引文获取能力
# Semantic Scholar Graph API: GET /graph/v1/paper/{paper_id}/references
# Semantic Scholar Graph API: GET /graph/v1/paper/{paper_id}/citations
# OpenAlex: GET /works/{work_id}?select=referenced_works,cited_by_count

async def fetch_citation_graph(paper: Paper, max_depth: int = 1) -> CitationGraph:
    """获取论文的引用和被引论文（一跳）"""
    # 1. 调用 S2 / OpenAlex API 获取 references + citations
    # 2. 对引用的论文也做 normalize + upsert（入库但标记为 citation_only）
    # 3. 返回边列表: [(source_id, target_id, "cites")]
    ...
```

**获取策略**：
- 仅对 **高优先级 + 中优先级** 论文获取引文数据（控制 API 调用量）
- 引用的论文中，已在库内的建立 `cites` 边；不在库内的创建轻量节点（`is_citation_node=True`，只存 title/year/doi）
- 一跳深度，不做两跳（Inciteful 做两跳但规模达 10K-20K，我们 15-30 篇核心论文 × 一跳 ≈ 200-500 篇引用节点，够用）

### 3.4 相似度计算

#### 引文相似度（参考 Inciteful）

```python
def compute_citation_similarity(papers: list[Paper], edges: list[CitationEdge]) -> SimilarityMatrix:
    """计算论文间的引文相似度"""
    # 1. 书目耦合（Adamic/Adar）: 两篇论文共同引用的论文越多越相似
    #    Score(a,b) = Σ 1/log(degree(w))，w 为共同引用的论文
    biblio_coupling = compute_adamic_adar(papers, edges)

    # 2. 共引相似度（Salton 索引）: 两篇论文被同一组论文引用的余弦相似度
    #    Salton(u,v) = |Γ(u) ∩ Γ(v)| / √(|Γ(u)| × |Γ(v)|)
    co_citation = compute_salton_index(papers, edges)

    # 3. 融合: 引文相似度 = 0.5 × biblio_coupling + 0.5 × co_citation
    citation_sim = 0.5 * biblio_coupling + 0.5 * co_citation
    return citation_sim
```

#### 语义相似度（复用 RAG embedding）

```python
def compute_semantic_similarity(paper_ids: list[str]) -> SimilarityMatrix:
    """复用 ChromaDB 中的论文 embedding 计算语义相似度"""
    # 1. 从 ChromaDB 获取所有论文的 embedding（论文级，取 chunk 均值或取摘要 embedding）
    # 2. 计算余弦相似度矩阵
    # 3. 过滤: 仅保留 similarity > 0.7 的边（稀疏化）
    ...
```

#### 融合相似度

```python
def compute_fused_similarity(citation_sim, semantic_sim, alpha=0.6):
    """融合引文相似度和语义相似度"""
    # alpha 控制引文 vs 语义的权重
    # 有引文数据的论文: fused = alpha * citation + (1-alpha) * semantic
    # 无引文数据的论文: fused = semantic（降级）
    ...
```

### 3.5 聚类算法

当前系统用 LLM 做聚类（`_build_paper_clusters`），存在两个问题：
1. 不持久化（每次 ideas 生成时重新算）
2. 不基于数据驱动（LLM 猜测分组）

**改进方案：双层聚类**

#### 方案 A：社区检测（数据驱动，推荐）

```python
import networkx as nx
from community import community_louvain

def detect_paper_communities(papers, similarity_edges) -> dict[int, list[str]]:
    """基于相似度图的社区检测"""
    G = nx.Graph()
    for p in papers:
        G.add_node(p.id, title=p.title, year=p.year, priority=p.priority)
    for edge in similarity_edges:
        G.add_edge(edge.source, edge.target, weight=edge.similarity)

    # Louvain 社区检测（与 GraphRAG doc 11 一致）
    partition = community_louvain.best_partition(G, resolution=1.0)

    communities = {}
    for paper_id, comm_id in partition.items():
        communities.setdefault(comm_id, []).append(paper_id)
    return communities
```

#### 方案 B：embedding 聚类（无引文数据时降级）

```python
from sklearn.cluster import KMeans  # 或 HDBSCAN

def cluster_by_embedding(paper_embeddings, n_clusters=5):
    """用 embedding 做 K-Means 聚类（降级方案）"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(paper_embeddings)
    return labels
```

#### 方案 C：LLM 命名（持久化现有逻辑）

社区检测产出分组后，调用 LLM 为每个聚类生成：
- `cluster_name`: 聚类名称
- `core_method`: 核心方法
- `summary`: 聚类摘要
- `key_findings`: 关键发现

### 3.6 数据库设计

新增 3 张表：

```sql
-- 论文引用关系
CREATE TABLE paper_citations (
    id TEXT PRIMARY KEY,
    source_paper_id TEXT NOT NULL REFERENCES papers(id),
    target_paper_id TEXT NOT NULL REFERENCES papers(id),
    relation_type TEXT NOT NULL,  -- 'cites' | 'co_cited' | 'biblio_coupled'
    weight FLOAT DEFAULT 1.0,     -- 相似度权重（对于 co_cited/biblio_coupled）
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source_paper_id, target_paper_id, relation_type)
);
CREATE INDEX idx_citations_source ON paper_citations(source_paper_id);
CREATE INDEX idx_citations_target ON paper_citations(target_paper_id);

-- 文献地图（每个任务一张地图，支持版本/快照）
CREATE TABLE literature_maps (
    id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES research_tasks(id),
    version INTEGER DEFAULT 1,          -- 增量更新时的版本号
    node_count INTEGER DEFAULT 0,
    edge_count INTEGER DEFAULT 0,
    cluster_count INTEGER DEFAULT 0,
    graph_json TEXT,                    -- 完整图数据 JSON（nodes + edges + clusters）
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_litmaps_task ON literature_maps(task_id);

-- 论文聚类（持久化聚类结果）
CREATE TABLE paper_clusters (
    id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES research_tasks(id),
    cluster_index INTEGER NOT NULL,     -- 聚类序号
    cluster_name TEXT,                  -- LLM 生成的聚类名
    core_method TEXT,                   -- 核心方法
    summary TEXT,                       -- 聚类摘要
    key_findings TEXT,                  -- 关键发现
    paper_ids_json TEXT NOT NULL,       -- 聚类内论文 ID 列表
    paper_count INTEGER DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_clusters_task ON paper_clusters(task_id);
```

### 3.7 图数据结构（API 返回）

```python
class LiteratureMapData(BaseModel):
    """前端可视化所需的完整图数据"""
    nodes: list[MapNode]
    edges: list[MapEdge]
    clusters: list[MapCluster]
    stats: MapStats

class MapNode(BaseModel):
    id: str                    # paper_id
    title: str
    year: int | None
    priority: str              # high / medium / low / citation_only
    final_score: float | None
    citation_count: int
    cluster_index: int | None  # 所属聚类
    is_seed: bool              # 是否为任务直接检索到的论文（vs 引文扩展节点）

class MapEdge(BaseModel):
    source: str
    target: str
    type: str                  # cites / co_cited / biblio_coupled / semantic
    weight: float

class MapCluster(BaseModel):
    index: int
    name: str
    core_method: str
    summary: str
    paper_count: int
    color: str                 # 前端渲染颜色

class MapStats(BaseModel):
    total_papers: int
    seed_papers: int
    citation_nodes: int
    total_edges: int
    cluster_count: int
    density: float             # 图密度
```

---

## 四、维护策略

### 4.1 增量更新机制

文献地图不是一次构建就固定的，随着检索轮次增加需要增量更新：

```
Round 1: 检索到 50 篇 → 构建初始地图 (v1)
Round 2: 新增 30 篇 → 增量更新地图 (v2)
  - 新论文加入图
  - 重新计算受影响的相似度边
  - 增量社区检测（可选：全量重算，规模小）
Report 阶段: 最终地图 (v_final)
  - 获取高/中优先级论文的引文数据
  - 全量重算相似度 + 聚类
  - LLM 命名聚类
  - 持久化
```

### 4.2 更新触发时机

| 时机 | 操作 | 耗时预估 |
|------|------|---------|
| 每轮检索后 | 语义相似度增量更新（新论文 vs 已有论文） | ~2s（50 篇 × embedding 比较） |
| 报告生成前 | 全量引文获取 + 相似度重算 + 社区检测 + LLM 命名 | ~30-60s |
| 用户反馈后 | 如果触发新检索轮，同"每轮检索后" | ~2s |

### 4.3 版本管理

```python
# 每次更新创建新版本，保留历史版本供对比
def update_literature_map(task_id: str, incremental: bool = True):
    """更新文献地图"""
    latest = get_latest_map(task_id)
    new_version = (latest.version + 1) if latest else 1

    if incremental and latest:
        # 增量: 只处理新论文
        new_paper_ids = get_papers_since(task_id, latest.updated_at)
        ...
    else:
        # 全量: 重新构建
        ...

    # 保存新版本
    save_map(task_id, new_version, graph_data)
```

### 4.4 引文数据缓存

引文 API 调用是主要瓶颈，需要缓存：

```python
# 引文数据全局缓存（跨任务复用）
# key: paper_id (DOI / S2 ID / arXiv ID)
# value: {references: [...], citations: [...], fetched_at: timestamp}
# TTL: 7 天（引文数据不会频繁变化）

# 存储位置: SQLite paper_citations 表（已有引用关系的不再重复获取）
# 或 Redis（如果后续引入）
```

### 4.5 降级策略

| 场景 | 降级方案 |
|------|---------|
| S2/OpenAlex 引文 API 限流 | 仅用语义相似度，跳过引文层 |
| ChromaDB embedding 不可用 | 用 TF-IDF / 标题关键词重叠作为相似度降级 |
| 社区检测失败（论文太少） | 退回 LLM 聚类（现有逻辑） |
| 图太大（>1000 节点） | 只展示高+中优先级论文，citation_only 节点折叠 |

---

## 五、实现计划

### Phase 1: 引文数据获取 + 存储（~3h）

1. `paper_sources/semantic_scholar.py` 新增 `fetch_references()` 和 `fetch_citations()`
2. `paper_sources/openalex.py` 新增引文获取
3. `models.py` 新增 `PaperCitation` 表
4. `paper_repo.py` 新增引文存储/查询方法
5. `runner.py` 在搜索循环结束后、报告生成前，批量获取高/中优先级论文的引文数据

### Phase 2: 相似度计算 + 聚类（~3h）

1. `services/literature_map_service.py`（新建）：
   - `compute_citation_similarity()`: Adamic/Adar + Salton
   - `compute_semantic_similarity()`: 复用 ChromaDB embedding
   - `compute_fused_similarity()`: 融合两层
   - `detect_communities()`: NetworkX + Louvain
2. `models.py` 新增 `LiteratureMap` + `PaperCluster` 表
3. LLM 聚类命名（复用现有 `CLUSTER_SYSTEM` prompt，改为对社区检测结果命名）

### Phase 3: API + 前端可视化（~4h）

1. `api/routes/maps.py`（新建）：
   - `GET /api/tasks/{id}/map` → 返回 `LiteratureMapData`
   - `GET /api/tasks/{id}/map/clusters` → 返回聚类详情
   - `POST /api/tasks/{id}/map/refresh` → 手动刷新地图
2. SSE 事件：`map_updated`（地图更新时推送）
3. 前端 `LiteratureMapView.tsx`（新建）：
   - 用 `react-force-graph-2d` 或 `d3-force` 渲染力导向图
   - 节点: 圆形，大小 = citation_count，颜色 = year 渐变，边框 = priority
   - 边: 引用边（实线箭头）/ 语义边（虚线）/ 共引边（点线）
   - 聚类: 半透明色块包围同聚类节点
   - 交互: 点击节点 → 侧边栏显示论文详情；hover → 高亮邻居；滚轮缩放
4. `TaskDetail.tsx` 新增"文献地图"标签页

### Phase 4: 增量维护 + 优化（~2h）

1. 每轮检索后增量更新语义相似度边
2. 报告生成前全量重建（引文 + 语义 + 聚类）
3. 引文数据缓存（paper_citations 表查重）
4. 图稀疏化（语义边只保留 top-K 邻居，避免边爆炸）

---

## 六、与现有系统的集成点

### 6.1 与 RAG 管线（doc 10）的关系

| RAG 管线产出 | 文献地图复用 |
|-------------|------------|
| ChromaDB 论文 chunk embedding | 取论文所有 chunk embedding 的均值作为论文级 embedding，用于语义相似度计算 |
| PaperChunk 表 | 知道哪些论文有全文、哪些只有摘要 |

### 6.2 与 GraphRAG（doc 11）的关系

| GraphRAG 产出 | 文献地图复用 |
|-------------|------------|
| KG 实体/关系 | 知识图谱社区可作为文献地图的聚类层（替代 Louvain） |
| KG 社区 | 直接映射为文献地图的聚类区域 |
| NetworkX 图结构 | 共用图存储和算法库 |

**实现顺序建议**：doc 10 (RAG) → **doc 12 (文献地图)** → doc 11 (GraphRAG)。文献地图的社区检测是 GraphRAG 的简化版，可以先落地，GraphRAG 后续升级时直接替换聚类层。

### 6.3 与现有聚类的替换

当前 `_build_paper_clusters`（runner.py L716-764）用 LLM 聚类，将被文献地图的社区检测替代：

```python
# 改造前（当前）:
async def _build_paper_clusters(db, state, llm, task_id):
    # LLM 聚类，临时结果，不持久化
    ...

# 改造后:
async def _build_paper_clusters(db, state, llm, task_id):
    # 1. 从 literature_maps 表读取最新聚类（如果已有）
    clusters = load_clusters_from_map(db, task_id)
    if clusters:
        return clusters
    # 2. 如果还没有地图，触发构建
    await build_literature_map(db, task_id, llm)
    clusters = load_clusters_from_map(db, task_id)
    return clusters
```

### 6.4 与 Ideas 生成的集成

Ideas 生成时，除了传入高优先级论文列表，还传入文献地图的聚类信息：

```python
# runner.py _generate_and_score_ideas 中
map_data = get_literature_map(db, task_id)
if map_data:
    cluster_context = format_map_clusters_for_ideas(map_data)
    # 传入 IDEAS prompt，让 LLM 看到结构化的聚类视图
    ...
```

---

## 七、前端可视化设计

### 7.1 布局

```
┌─────────────────────────────────────────────────────────┐
│  文献地图                                        [刷新]  │
├──────────────┬──────────────────────────────────────────┤
│  控制面板     │  图谱画布                                │
│              │                                          │
│  □ 引用边    │     ●───●       ●───●                   │
│  □ 共引边    │     │   │       │   │                   │
│  □ 语义边    │     ●───●───────●───●                   │
│              │      \     /       │                     │
│  优先级:     │       ●───●        ●                     │
│  ☑ 高       │        聚类A      聚类B                  │
│  ☑ 中       │                                          │
│  ☐ 低       │                                          │
│  ☐ 引文节点  │                                          │
│              │                                          │
│  聚类: 4个   │                                          │
│  论文: 87篇  │                                          │
│  边: 234条   │                                          │
├──────────────┴──────────────────────────────────────────┤
│  选中论文详情:                                           │
│  [标题] [年份] [评分] [优先级] [引用数]                 │
│  [摘要...]                                               │
│  [关联论文: 5篇]  [所在聚类: 基于Transformer的...]      │
└─────────────────────────────────────────────────────────┘
```

### 7.2 交互

- **点击节点**：侧边栏显示论文详情 + 高亮邻居
- **双击节点**：跳转到论文列表页并定位该论文
- **hover 聚类区域**：显示聚类名称和摘要
- **拖拽节点**：力导向布局自动重排
- **滚轮缩放**：缩放图谱
- **切换边类型**：勾选控制面板中的边类型
- **筛选节点**：按优先级/年份/聚类筛选

### 7.3 技术选型

- **react-force-graph-2d**：基于 d3-force 的 React 力导向图组件，支持大规模节点
- 备选：**@antv/G6**（蚂蚁图可视化引擎，中文生态好）
- 备选：**cytoscape.js**（成熟稳定，布局算法丰富）

---

## 八、新增依赖

```python
# requirements.txt 新增
networkx==3.4.2              # 图结构（与 doc 11 GraphRAG 共用）
python-louvain==0.16         # 社区检测（与 doc 11 GraphRAG 共用）
# react-force-graph-2d       # 前端（npm install react-force-graph-2d）
# ChromaDB 和 sentence-transformers 已在 doc 10 中引入
```

---

## 九、LLM 调用量预估

| 步骤 | LLM 调用数 | 说明 |
|------|-----------|------|
| 引文获取 | 0 | 纯 API 调用（S2/OpenAlex） |
| 相似度计算 | 0 | 纯算法（Adamic/Adar / 余弦相似度） |
| 社区检测 | 0 | 纯算法（Louvain） |
| 聚类命名 | 3-6 | 每个社区一次 LLM 调用（生成名称+摘要） |
| **总计** | **3-6** | 极少，可忽略 |

> 相比当前 LLM 聚类（1 次调用处理 60 篇论文），社区检测 + LLM 命名的方案调用数相当但质量更高（数据驱动 + 命名精确）。

---

## 十、与 doc 11 GraphRAG 的关系

| 维度 | doc 12 文献地图 | doc 11 GraphRAG |
|------|----------------|-----------------|
| **目标** | 论文级可视化 + 主题聚类 | 实体级知识图谱 + 跨论文推理 |
| **节点** | 论文 | 模型/数据集/算法等实体 |
| **边** | 引用/共引/语义相似 | uses/evaluates_on/outperforms |
| **聚类** | Louvain 社区检测（论文级） | Leiden 社区检测（实体级） |
| **检索** | 可视化浏览 | Local/Global/Hybrid Search |
| **依赖** | doc 10（复用 embedding） | doc 10（复用 chunks + embedding） |
| **优先级** | **先实施**（用户可见价值） | 后实施（idea 生成增强） |

**演进路径**：doc 12 文献地图 → doc 11 GraphRAG 时，文献地图的论文级社区检测可升级为实体级社区检测，文献地图的聚类区域可映射 GraphRAG 的社区。
