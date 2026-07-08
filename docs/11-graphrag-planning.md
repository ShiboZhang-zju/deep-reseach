# 11 - GraphRAG 知识图谱检索规划

> 基于 doc 10 RAG 管线的升级路线：从语义检索到知识图谱推理
> 状态：**规划中**，待 doc 10 实现完成后启动

## 背景

### 为什么需要 GraphRAG

RAG（doc 10）解决了"从论文全文检索相关段落"的问题，但有局限：

| RAG 局限 | GraphRAG 优势 |
|---------|--------------|
| 只能检索语义相似的段落 | 能发现跨论文的实体关联（非显而易见的连接） |
| 无法回答"所有论文中用BERT的有哪些" | 实体查询，直接遍历知识图谱 |
| 无法做全局摘要（跨所有论文） | 社区检测 + 社区摘要，map-reduce 全局分析 |
| 聚类靠 LLM 猜测 | 社区检测算法（Leiden）数据驱动聚类 |
| Idea 生成只看相似段落 | 能发现"论文A的方法 + 论文B的数据集"的组合 |

### 参考项目

| 项目 | 方法 | 关键特点 |
|------|------|---------|
| **Microsoft GraphRAG** | LLM实体抽取→社区检测→社区摘要→全局/局部搜索 | 工业级，Parquet存储，Leiden算法 |
| **KGGen (Stanford)** | 两步法：实体→关系三元组→迭代聚类合并 | 聚类去重，MINE基准66% |
| **LightRAG** | 轻量GraphRAG，双层检索（实体+关系） | 比MS GraphRAG快，适合中小规模 |
| **Idea2Paper** | 摘要→pattern提取→Idea/Pattern/Domain/Paper四类节点 | 三路径检索+两阶段排序 |

**选型**：参考 **LightRAG** + **KGGen** 的轻量方案，不直接用 Microsoft GraphRAG（太重，需要大量LLM调用做社区摘要）。

## 架构

```
doc 10 RAG 输出                GraphRAG 构建（离线）              GraphRAG 检索（在线）
┌──────────────┐    ┌──────────────────────────┐    ┌─────────────────────────┐
│ paper_chunks │───→│ 1. LLM 实体抽取           │───→│ Local Search            │
│ (ChromaDB)   │    │    (模型/数据集/算法/指标) │    │   查询实体 → 关联实体   │
│              │    │ 2. LLM 关系抽取           │    │   → 关联段落            │
│              │    │    (uses/evaluates_on/    │    │                         │
│              │    │     outperforms/...)      │    │ Global Search           │
│              │    │ 3. 实体聚类去重 (KGGen式)  │    │   社区级 map-reduce     │
│              │    │ 4. 社区检测 (Leiden)      │    │   跨论文全局摘要        │
│              │    │ 5. 社区摘要 (LLM, 可选)   │    │                         │
│              │    │ 6. 存入 ChromaDB +        │    │ Hybrid Search           │
│              │    │    NetworkX graph         │    │   RAG + GraphRAG 融合   │
└──────────────┘    └──────────────────────────┘    └─────────────────────────┘
```

## Phase 1: 实体与关系抽取

### 实体类型

```python
class EntityType:
    MODEL = "model"          # BERT, GPT-4, Transformer-XL
    DATASET = "dataset"      # MMLU, GLUE, MultiWOZ
    ALGORITHM = "algorithm"  # RLHF, contrastive learning, DPO
    METRIC = "metric"        # F1, BLEU, accuracy
    TASK = "task"            # machine translation, sentiment analysis
    METHOD = "method"        # chain-of-thought, few-shot
    BASELINE = "baseline"    # GPT-3.5, T5-base
```

### 关系类型

```python
class RelationType:
    USES = "uses"                    # paper → model (论文使用了某模型)
    EVALUATES_ON = "evaluates_on"    # paper → dataset (在数据集上评估)
    OUTPERFORMS = "outperforms"      # model → baseline (优于基线)
    PROPOSES = "proposes"            # paper → method (提出了某方法)
    EXTENDS = "extends"              # method → method (扩展了某方法)
    COMBINES = "combines"            # method → method (组合了方法)
    MEASURED_BY = "measured_by"      # task → metric (用指标衡量)
```

### 抽取 Prompt（参考 KGGen 两步法）

```python
ENTITY_EXTRACT_SYSTEM = """You are a knowledge graph extractor.
Extract entities from the given paper passage. Only extract entities of these types:
- model: specific model names (e.g., "BERT", "GPT-4", "Transformer-XL")
- dataset: dataset names (e.g., "MMLU", "GLUE", "MultiWOZ")
- algorithm: algorithm/technique names (e.g., "RLHF", "contrastive learning")
- metric: evaluation metrics (e.g., "F1", "BLEU", "accuracy")
- task: research tasks (e.g., "machine translation", "sentiment analysis")
- method: specific methods (e.g., "chain-of-thought", "few-shot prompting")
- baseline: baseline systems compared against

Return JSON: {"entities": [{"name": "...", "type": "..."}]}
Be CONCRETE. Do not extract generic terms like "neural network" or "deep learning".
Output in the entity's original language (usually English)."""

RELATION_EXTRACT_SYSTEM = """You are a relation extractor.
Given entities and the source passage, extract subject-predicate-object triples.
Valid predicates: uses, evaluates_on, outperforms, proposes, extends, combines, measured_by

Return JSON: {"relations": [{"subject": "...", "predicate": "...", "object": "..."}]}
Only extract relations explicitly stated in the passage. Do not infer."""
```

### 抽取流程

```python
async def extract_knowledge_graph(chunks: list[PaperChunk], llm) -> tuple[list[Entity], list[Relation]]:
    """从论文段落中抽取实体和关系"""
    all_entities = []
    all_relations = []
    
    for chunk in chunks:
        # Step 1: 实体抽取
        entity_result = await llm.chat_json([
            {"role": "system", "content": ENTITY_EXTRACT_SYSTEM},
            {"role": "user", "content": chunk.text},
        ], EntityList)
        
        # Step 2: 关系抽取（传入已抽取的实体）
        relation_result = await llm.chat_json([
            {"role": "system", "content": RELATION_EXTRACT_SYSTEM},
            {"role": "user", "content": f"Entities: {entity_result.entities}\n\nPassage: {chunk.text}"},
        ], RelationList)
        
        all_entities.extend(entity_result.entities)
        all_relations.extend(relation_result.relations)
    
    return all_entities, all_relations
```

## Phase 2: 实体聚类去重（参考 KGGen）

不同段落/论文中抽取的同一实体可能有不同写法（"GPT-4" vs "GPT4" vs "gpt-4"）。

```python
async def cluster_entities(entities: list[Entity], llm) -> dict[str, str]:
    """聚类合并同义实体，返回 {original_name → canonical_name}"""
    # Step 1: 规则预处理（小写、去标点）
    normalized = {}
    for e in entities:
        key = e.name.lower().replace("-", "").replace("_", "").replace(" ", "")
        normalized.setdefault(key, []).append(e.name)
    
    # Step 2: LLM 验证模糊匹配（参考 KGGen 聚类验证）
    mapping = {}
    for key, names in normalized.items():
        if len(names) == 1:
            mapping[names[0]] = names[0]
        else:
            # LLM 判断是否同一实体，选择最规范写法
            canonical = await llm_canonical_name(names)
            for n in names:
                mapping[n] = canonical
    
    return mapping
```

## Phase 3: 社区检测

```python
import networkx as nx
from community import community_louvain  # python-louvain (Leiden算法)

def detect_communities(entities: list[Entity], relations: list[Relation]) -> dict:
    """构建图并检测社区"""
    G = nx.Graph()
    
    # 添加节点
    for e in entities:
        G.add_node(e.id, name=e.name, type=e.type, papers=e.paper_ids)
    
    # 添加边（关系作为边）
    for r in relations:
        if G.has_edge(r.subject, r.object):
            G[r.subject][r.object]["weight"] += 1
        else:
            G.add_edge(r.subject, r.object, 
                       relation=r.predicate, weight=1)
    
    # Leiden/Louvain 社区检测
    communities = community_louvain.best_partition(G)
    
    # 组织社区
    community_groups = {}
    for node, comm_id in communities.items():
        community_groups.setdefault(comm_id, []).append(node)
    
    return {
        "graph": G,
        "communities": community_groups,
        "num_communities": len(community_groups),
    }
```

**社区含义**：每个社区代表一个研究主题群（如"基于Transformer的对话系统"社区包含 BERT/GPT/MultiWOZ/对话理解等实体）。

## Phase 4: 存储

### ChromaDB（实体 embedding）

```python
# 新增 ChromaDB collection
entity_collection = chroma_client.get_or_create_collection(
    name="entity_embeddings",
    metadata={"hnsw:space": "cosine"},
)

# 实体描述 embedding（用于实体检索）
entity_collection.add(
    ids=[e.id for e in entities],
    embeddings=embed_entity_descriptions(entities),  # bge-base, 768维
    documents=[f"{e.name} ({e.type}): {e.description}" for e in entities],
    metadatas=[{"name": e.name, "type": e.type, "papers": e.paper_ids} for e in entities],
)
```

### NetworkX 图（关系结构）

```python
# 持久化为 GraphML
nx.write_graphml(G, "./graph_data/paper_kg.graphml")

# 或序列化为 JSON（更易调试）
graph_data = nx.node_link_data(G)
with open("./graph_data/paper_kg.json", "w") as f:
    json.dump(graph_data, f, ensure_ascii=False, indent=2)
```

### SQLite（实体/关系元数据）

```sql
CREATE TABLE kg_entities (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT NOT NULL,
    canonical_name TEXT NOT NULL,
    paper_ids_json TEXT,          -- 哪些论文提到了这个实体
    description TEXT,
    community_id INTEGER,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE kg_relations (
    id TEXT PRIMARY KEY,
    subject_entity_id TEXT REFERENCES kg_entities(id),
    predicate TEXT NOT NULL,
    object_entity_id TEXT REFERENCES kg_entities(id),
    paper_ids_json TEXT,
    weight INTEGER DEFAULT 1,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE kg_communities (
    id INTEGER PRIMARY KEY,
    summary TEXT,                 -- LLM 生成的社区摘要
    entity_ids_json TEXT,
    paper_ids_json TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

## Phase 5: 检索策略

### Local Search（实体级）

```python
async def graphrag_local_search(query: str, top_k: int = 10) -> list:
    """查询实体 → 关联实体 → 关联段落"""
    # 1. 查询编码
    query_emb = embed(query)
    
    # 2. 从 entity_collection 检索相关实体
    entity_results = entity_collection.query(
        query_embeddings=[query_emb], n_results=5
    )
    
    # 3. 从图中扩展关联实体（1跳邻居）
    related_entities = []
    for ent_id in entity_results["ids"][0]:
        neighbors = list(G.neighbors(ent_id))
        related_entities.extend(neighbors)
    
    # 4. 获取关联段落（通过 entity → paper_ids → chunks）
    paper_ids = set()
    for ent_id in entity_results["ids"][0] + related_entities:
        paper_ids.update(G.nodes[ent_id].get("papers", []))
    
    # 5. 从 paper_chunks 检索段落
    chunks = rag_retrieve(query, paper_ids=list(paper_ids), top_k=top_k)
    
    return {"entities": entity_results, "chunks": chunks}
```

### Global Search（社区级）

```python
async def graphrag_global_search(query: str) -> str:
    """社区级 map-reduce 全局搜索"""
    # 1. 获取所有社区摘要
    communities = load_all_communities()
    
    # 2. Map: 每个社区根据query生成局部回答
    partial_answers = []
    for comm in communities:
        answer = await llm.chat([
            {"role": "system", "content": "Based on this research community summary, answer the query."},
            {"role": "user", "content": f"Community: {comm.summary}\nQuery: {query}"},
        ])
        partial_answers.append(answer)
    
    # 3. Reduce: 综合所有局部回答
    final_answer = await llm.chat([
        {"role": "system", "content": "Synthesize partial answers into a comprehensive response."},
        {"role": "user", "content": f"Partial answers:\n{chr(10).join(partial_answers)}"},
    ])
    
    return final_answer
```

### Hybrid Search（RAG + GraphRAG）

```python
async def hybrid_search(query: str, top_k: int = 10) -> dict:
    """融合 RAG 语义检索 + GraphRAG 知识图谱检索"""
    # RAG 检索
    rag_results = rag_retrieve(query, top_k=top_k)
    
    # GraphRAG 检索
    graph_results = await graphrag_local_search(query, top_k=top_k)
    
    # 融合排序（RAG score + graph centrality boost）
    fused = fuse_results(rag_results, graph_results)
    
    return fused
```

## 下游集成点

### 1. Idea 生成增强（核心价值）

```python
# 发现跨论文方法组合
def find_method_combinations(graph) -> list[dict]:
    """从知识图谱中发现可组合的方法"""
    combinations = []
    
    # 找出"extends"和"combines"关系，推断新组合
    for node in graph.nodes:
        if graph.nodes[node].get("type") == "method":
            # 方法A的论文 + 方法B的数据集
            method_papers = graph.nodes[node].get("papers", [])
            for neighbor in graph.neighbors(node):
                if graph.nodes[neighbor].get("type") == "dataset":
                    ds_papers = graph.nodes[neighbor].get("papers", [])
                    # 方法A还未在数据集B上评估过
                    if not set(method_papers) & set(ds_papers):
                        combinations.append({
                            "method": node,
                            "dataset": neighbor,
                            "method_papers": method_papers,
                            "dataset_papers": ds_papers,
                        })
    
    return combinations
```

### 2. 聚类增强（数据驱动替代 LLM 猜测）

```python
# 用社区检测替代现有 _build_paper_clusters
def cluster_by_community(graph, papers) -> list[Cluster]:
    """基于知识图谱社区对论文聚类"""
    paper_communities = {}
    for node in graph.nodes:
        comm_id = graph.nodes[node].get("community_id")
        for paper_id in graph.nodes[node].get("papers", []):
            paper_communities.setdefault(comm_id, set()).add(paper_id)
    
    return [Cluster(theme=comm.summary, paper_ids=list(pids)) 
            for comm, pids in paper_communities.items()]
```

### 3. 新颖性检查增强

```python
def check_novelty_via_graph(idea, graph) -> float:
    """检查idea中的方法组合是否已存在于知识图谱"""
    # 提取idea中的实体
    idea_entities = extract_entities_from_idea(idea)
    
    # 检查实体间的组合关系是否已存在
    existing_combos = 0
    for method in idea_entities.get("methods", []):
        for dataset in idea_entities.get("datasets", []):
            if graph.has_edge(method, dataset):
                existing_combos += 1
    
    # 已存在的组合越多，新颖性越低
    novelty = 1.0 - (existing_combos / max(len(idea_entities.get("methods", [])) * len(idea_entities.get("datasets", [])), 1))
    return novelty
```

## 新增依赖

```python
# requirements.txt 新增（在doc 10基础上）
networkx==3.4.2              # 图结构
python-louvain==0.16         # 社区检测（Leiden/Louvain算法）
# ChromaDB 和 sentence-transformers 已在 doc 10 中引入
```

## 涉及文件

| 文件 | 改动 |
|------|------|
| `backend/requirements.txt` | 新增 networkx, python-louvain |
| `backend/app/db/models.py` | 新增 KGEntity, KGRelation, KGCommunity 模型 |
| `backend/app/services/graphrag_service.py` | **新建**：实体抽取、关系抽取、社区检测、检索 |
| `backend/app/services/rag_service.py` | 新增 entity_collection 管理 |
| `backend/app/agent/runner.py` | `_build_paper_clusters` 改用社区检测；`_generate_and_score_ideas` 接入方法组合发现 |
| `backend/app/agent/prompts.py` | 新增 ENTITY_EXTRACT_SYSTEM, RELATION_EXTRACT_SYSTEM |
| `backend/app/schemas/schemas.py` | 新增 Entity, Relation, Community, EntityList, RelationList |

## LLM 调用量预估

| 步骤 | LLM 调用数（15篇论文, ~100 chunks） | 说明 |
|------|--------------------------------------|------|
| 实体抽取 | ~100 | 每个 chunk 一次 |
| 关系抽取 | ~100 | 每个 chunk 一次 |
| 实体聚类验证 | ~10-20 | 模糊匹配对 |
| 社区摘要 | ~3-5 | 每个社区一次（可选） |
| **总计** | **~220-230** | 用小模型（qwen3.6-35b），约2-3分钟 |

> **优化**：可批量处理多个 chunk（一次传入5个），减少调用次数到 ~40-50。

## 实现步骤

### Step 1: 实体/关系抽取（~3h）
1. 定义 Entity/Relation schema
2. 实现 ENTITY_EXTRACT_SYSTEM / RELATION_EXTRACT_SYSTEM prompt
3. 实现 `extract_knowledge_graph` 函数
4. 单元测试：对3篇论文抽取，验证实体质量

### Step 2: 图构建与社区检测（~2h）
1. 实现 NetworkX 图构建
2. 实体聚类去重（KGGen式）
3. Leiden 社区检测
4. 图持久化（GraphML + SQLite）

### Step 3: 检索引擎（~2h）
1. 实现 Local Search（实体级）
2. 实现 Global Search（社区级）
3. 实现 Hybrid Search（RAG + GraphRAG 融合）

### Step 4: 下游集成（~3h）
1. `_build_paper_clusters` 改用社区检测
2. `_generate_and_score_ideas` 接入方法组合发现
3. 新颖性检查接入图谱验证
4. 报告生成接入 Global Search

## 与 doc 10 的关系

```
doc 10 (RAG):           doc 11 (GraphRAG):
  PDF → chunks            chunks → 实体/关系 → 知识图谱
  → ChromaDB              → ChromaDB (entities) + NetworkX (graph)
  → 语义检索               → 实体检索 + 社区检索
  → method_extract        → 方法组合发现
  → 聚类(LLM)             → 聚类(社区检测, 数据驱动)
  → idea生成               → idea生成(跨论文组合, 新颖性验证)
```

**依赖关系**：doc 11 必须在 doc 10 完成后实施，因为：
1. 实体抽取需要 paper chunks（doc 10 的 PDF解析输出）
2. 实体 embedding 复用 doc 10 的 bge-base 模型和 ChromaDB client
3. Hybrid Search 需要 doc 10 的 `rag_retrieve` 函数
