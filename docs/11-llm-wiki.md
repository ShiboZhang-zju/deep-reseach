# 11 - LLM Wiki 知识编译管线

> 替换原 GraphRAG 规划（doc 11），采用 Karpathy LLM Wiki 范式
> 状态：**已实现**

## 背景

### 为什么放弃 GraphRAG，改用 LLM Wiki

原 GraphRAG 方案（实体抽取→关系抽取→社区检测→检索）有以下问题：

| GraphRAG 问题 | LLM Wiki 解决方案 |
|--------------|----------------|
| 每次查询都要重新派生知识（实体遍历→合成） | 知识在 ingest 阶段一次性编译，后续直接读取 |
| 实体/关系抽取质量不稳定（LLM 两步法容易遗漏） | LLM 直接编写结构化 wiki 页面，质量更高 |
| 社区检测（Leiden）结果不可解释 | 概念页面即聚类，天然可读 |
| 交叉引用需图遍历才发现 | 预编译 `[[wikilinks]]`，直接可用 |
| 矛盾检测困难 | ingest 时自动捕获，存入 contradictions 字段 |
| 需要 NetworkX + python-louvain 额外依赖 | 纯 LLM + SQLite，零额外依赖 |
| 人类不可读（图结构） | wiki 页面本身是可读的 Markdown 制品 |
| ~220 次 LLM 调用（实体+关系抽取） | ~15-30 次（每批5篇一次，15篇≈3批） |

### LLM Wiki 核心理念

> "The wiki is a persistent compounding artifact. The cross-references are already there.
> The contradictions are already flagged. The synthesis already reflects all the material."

LLM 不再是检索器，而是 **wiki 编辑器**。每个交叉引用、矛盾标记、合成结果在 ingest 阶段就已完成，后续查询直接复用。

## 架构

```
doc 10 RAG 输出              LLM Wiki 构建（离线）              下游使用（在线）
┌──────────────┐    ┌──────────────────────────┐    ┌─────────────────────────┐
│ paper_chunks │───→│ 1. 批量 Ingest (5篇/批)  │───→│ 报告生成               │
│ (ChromaDB)   │    │ 2. LLM 生成 wiki actions  │    │   注入 wiki 上下文      │
│ paper abstracts│  │    (create/update pages)  │    │   替代原始摘要           │
│ method_extract│   │ 3. 执行 actions           │    │                         │
│              │    │    (merge into existing)  │    │ Ideas 生成              │
│              │    │ 4. 更新 index             │    │   注入 wiki 方法/数据集  │
│              │    │ 5. (可选) Lint 健康检查   │    │   发现跨论文组合          │
└──────────────┘    └──────────────────────────┘    │                         │
                                                     │ 聚类（替代LLM聚类）     │
┌──────────────┐                                    │   wiki concept pages    │
│ wiki_pages   │◄───────────────────────────────────│   = 预编译聚类           │
│ (SQLite)     │                                    └─────────────────────────┘
│  - concept   │
│  - method     │
│  - dataset    │
│  - model      │
│  - synthesis  │
│  - index      │
└──────────────┘
```

### 三层架构

| 层 | 性质 | 读写权限 | 本项目映射 |
|---|---|---|---|
| Raw | 不可变真理之源 | LLM 只读 | 论文摘要 + RAG chunks (doc 10) |
| Wiki | 结构化 Markdown | LLM 写，用户读 | `wiki_pages` 表 (SQLite) |
| Schema | 维护规则 | 用户+LLM 演进 | `WIKI_INGEST_SYSTEM` prompt |

## 页面类型

```python
class WikiPageType:
    CONCEPT = "concept"      # 研究概念/主题（替代聚类）
    METHOD = "method"        # 具体方法/算法
    DATASET = "dataset"      # 数据集
    MODEL = "model"          # 模型
    SYNTHESIS = "synthesis"  # 跨主题综合分析
    INDEX = "index"          # 自动维护的索引
```

### Concept 页面示例

```markdown
# 记忆增强的大语言模型

## Summary
通过为 LLM 添加外部记忆机制，解决上下文窗口有限的限制，使模型能访问长程历史信息。

## Papers
- [P1] MemGPT: LLM 管理分层记忆池，通过 summarization 压缩旧记忆
- [P3] A-Mem: agentic memory 实现自主记忆管理，基于 Llama-3
- [P7] MemoryBank: 基于间隔重复的记忆遗忘机制

## Technical Details
- MemGPT: 主记忆 + 归档记忆 + 检索记忆三层架构，summarization 压缩
- A-Mem: 基于 Llama-3-8B，agentic 自主决策记忆操作
- MemoryBank: Ebbinghaus 遗忘曲线 + 间隔重复算法

## Strengths
- 不改变模型参数，纯推理时增强
- 可支持超长上下文（百万token级）

## Limitations
- 记忆检索延迟增加推理时间 [P1]
- 仅限文本模态，不支持多模态记忆 [P3]
- 记忆一致性难以保证 [P7]

## Cross-references
[[Llama-3]], [[RAG]], [[Summarization]]
```

## 核心操作

### 1. Ingest（论文摄入）

```python
async def ingest_papers_to_wiki(db, papers, llm, task_id):
    """批量摄入论文到 wiki。
    
    流程:
    1. 按优先级排序论文（high 优先）
    2. 分批处理（每批5篇）
    3. 每批:
       a. 构建论文上下文（标题+摘要+方法提取+RAG段落）
       b. 列出现有 wiki 页面
       c. LLM 生成 actions（create/update）
       d. 执行 actions（merge 到已有页面）
    4. 重新生成 index
    """
```

**关键设计**：每次 ingest 时，LLM 能看到已有 wiki 页面列表，因此：
- 已有主题 → `update`（合并新信息）
- 新主题 → `create`（新页面）
- 一次摄入5篇论文，可能触及10-15个页面更新

### 2. Query（查询）

```python
def get_wiki_context(db, task_id, page_types=None):
    """获取 wiki 页面作为报告/idea 生成的上下文。
    
    - 报告生成: 传入所有页面类型
    - Idea生成: 重点传入 method/dataset/concept 页面
    - 聚类: 仅传入 concept 页面
    """
```

### 3. Lint（健康检查，可选）

```python
async def lint_wiki(db, task_id, llm):
    """检查 wiki 健康度:
    - contradictions: 页面间矛盾
    - orphan: 无入站链接的孤立页面
    - stale: 陈旧信息
    - missing_link: 应建立但未建立的交叉引用
    """
```

## 数据模型

```sql
CREATE TABLE wiki_pages (
    id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES research_tasks(id),
    page_type TEXT NOT NULL,         -- concept/method/dataset/model/synthesis/index
    title TEXT NOT NULL,
    content_markdown TEXT DEFAULT '',
    paper_ids_json TEXT DEFAULT '[]',    -- 引用的论文 ID
    links_json TEXT DEFAULT '[]',         -- 出站 wikilinks
    contradictions_json TEXT DEFAULT '[]', -- 发现的矛盾
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_wiki_task_type ON wiki_pages(task_id, page_type);
CREATE INDEX idx_wiki_task_title ON wiki_pages(task_id, title);
```

## 下游集成

### 1. 聚类（替代 LLM 聚类 + GraphRAG 社区检测）

```python
# 旧: _build_paper_clusters → LLM 一次性聚类所有论文
# 新: get_wiki_clusters → 从 wiki concept 页面提取聚类

def get_wiki_clusters(db, task_id):
    """将 wiki concept 页面转换为 ClusterList 格式。
    
    每个 concept 页面 = 一个聚类：
    - cluster_name = 页面标题
    - core_method = Technical Details 段落
    - limitations = Limitations 段落
    - representative_papers = paper_ids 对应的论文标题
    - cross_cluster_gaps = synthesis 页面 + contradictions
    """
```

**优势**：
- 聚类在 ingest 阶段就已完成（增量累积），不需要报告前临时聚类
- 聚类结果可解释（wiki 页面本身就是解释）
- 支持跨论文方法组合发现（synthesis 页面）

### 2. 报告生成增强

```python
# _generate_report 中注入 wiki 上下文
wiki_context = get_wiki_context(db, task_id)
# 传入 outline 生成和 section 生成 prompt
```

**效果**：报告 LLM 不再只看到原始论文摘要，而是看到预编译的方法对比、技术细节、已知矛盾。

### 3. Idea 生成增强

```python
# _generate_and_score_ideas 中注入 wiki 上下文
wiki_context = get_wiki_context(db, task_id)
# 传入 idea 生成 prompt
```

**效果**：
- method 页面提供具体方法名（防幻觉）
- dataset 页面提供真实数据集名
- synthesis 页面提供跨论文组合机会（替代 GraphRAG 的图遍历）
- contradictions 提供研究空白方向

## LLM 调用量对比

| 步骤 | GraphRAG (原方案) | LLM Wiki (新方案) |
|------|------------------|------------------|
| 实体抽取 | ~100 次（每chunk一次） | 0 |
| 关系抽取 | ~100 次（每chunk一次） | 0 |
| 实体聚类 | ~10-20 次 | 0 |
| 社区摘要 | ~3-5 次 | 0 |
| **Wiki ingest** | 0 | **~3-6 次**（每批5篇，15篇≈3批） |
| Wiki lint（可选） | 0 | ~1 次 |
| **总计** | **~220 次** | **~4-7 次** |

> **优化**：LLM Wiki 调用量仅为 GraphRAG 的 2-3%，且每次调用产出更有价值（完整 wiki 页面 vs 单个三元组）。

## 涉及文件

| 文件 | 改动 |
|------|------|
| `backend/app/services/wiki_service.py` | **新建**：Wiki 引擎（ingest/query/lint/clusters） |
| `backend/app/db/models.py` | 新增 `WikiPage` 模型 |
| `backend/app/schemas/schemas.py` | 新增 `WikiAction`, `WikiActionList`, `WikiLintIssue`, `WikiLintResult` |
| `backend/app/agent/prompts.py` | 新增 `WIKI_INGEST_SYSTEM/USER`, `WIKI_LINT_SYSTEM/USER` |
| `backend/app/agent/runner.py` | RAG 后添加 wiki ingest；`_build_paper_clusters` 改用 wiki；报告/idea 注入 wiki 上下文 |

## 实现步骤

### Step 1: 数据层 ✅
1. 定义 `WikiPage` ORM 模型
2. 定义 `WikiAction`/`WikiActionList` Pydantic schemas
3. 定义 `WikiLintIssue`/`WikiLintResult` schemas

### Step 2: Wiki 引擎 ✅
1. 实现 `ingest_papers_to_wiki`（批量 ingest + actions 执行）
2. 实现 `get_wiki_context`（查询上下文）
3. 实现 `get_wiki_clusters`（ClusterList 兼容）
4. 实现 `_merge_page`（增量合并）
5. 实现 `_regenerate_index`（自动索引）
6. 实现 `lint_wiki`（健康检查）

### Step 3: Prompt 设计 ✅
1. `WIKI_INGEST_SYSTEM/USER`：wiki 编辑器指令
2. `WIKI_LINT_SYSTEM/USER`：健康审计指令

### Step 4: Runner 集成 ✅
1. RAG 索引后添加 wiki ingest 步骤
2. `_build_paper_clusters` 优先使用 wiki，LLM 聚类作为 fallback
3. `_generate_report` 注入 wiki 上下文
4. `_generate_and_score_ideas` 注入 wiki 上下文

## 与 GraphRAG 的能力对比

| 能力 | GraphRAG | LLM Wiki |
|------|----------|----------|
| 跨论文实体关联 | ✅（图遍历） | ✅（concept 页面 + [[wikilinks]]） |
| 全局摘要 | ✅（社区检测 + map-reduce） | ✅（synthesis 页面，预编译） |
| 方法组合发现 | ✅（图遍历推断） | ✅（synthesis 页面，LLM 在 ingest 时分析） |
| 矛盾检测 | ❌（困难） | ✅（ingest 时自动标记） |
| 知识累积 | ❌（每次查询重新派生） | ✅（增量累积，越用越丰富） |
| 人类可读 | ❌（图结构） | ✅（Markdown 页面） |
| LLM 调用量 | ~220次 | ~4-7次 |
| 额外依赖 | networkx, python-louvain | 无 |

## 与 doc 10/12 的关系

```
doc 10 (RAG):           doc 11 (LLM Wiki):       doc 12 (文献地图):
  PDF → chunks            chunks → wiki pages      paper embeddings → 引文网络
  → ChromaDB              → SQLite wiki_pages      → 力导向图可视化
  → 语义检索               → 知识合成 + 交叉引用     → 社区检测 + LLM命名
  → method_extract        → 聚类(wiki concept)     → 可视化聚类
  → 聚类(LLM)             → idea生成(跨论文组合)    → (复用 doc 10 embedding)
  → idea生成              → 新颖性验证(wiki)
```

**依赖关系**：
- doc 11 必须在 doc 10 完成后实施（需要 RAG chunks 作为 wiki ingest 的素材）
- doc 12 的聚类层可升级为使用 wiki concept 页面（替代 LLM 聚类）
