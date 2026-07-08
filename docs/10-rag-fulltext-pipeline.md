# 10 - RAG 全文检索管线

> 根因方案：解决 method_extract 仅依赖摘要导致的幻觉问题
> 架构升级（2026-07-07）：numpy 方案改为 ChromaDB 向量数据库，为 GraphRAG（doc 11）预留扩展

## 背景

### 现状问题
- `method_extract` 从标题+摘要（前1000字符）提取模型/算法/数据集
- 摘要通常不含 Method/Experiment 细节，论文的 Method 章节才是技术细节来源
- 强制 LLM 从有限信息推断 → 产生幻觉（编造模型名/数据集）
- `Paper.pdf_url` 字段已存储但从未使用
- 下游聚类/报告/idea 生成都基于不准确的 method_extract → 错误传播

### 调研结论（2026-07-07）

| 项目 | 内容来源 | 防幻觉机制 | 幻觉率 |
|------|----------|-----------|--------|
| **OpenScholar** | 全文切250词段落，2.34亿段落 | 双编码器检索+交叉编码器重排+引用验证 | **0%** |
| **STORM** | 互联网搜索（非论文库） | 多视角提问+对话式收集+来源过滤 | N/A |
| **Idea2Paper** | 摘要（同我们） | 知识图谱结构化 | 有局限 |
| **KGGen** | 任意文本 | 两步提取(实体→关系)+迭代聚类 | — |

**核心结论**：OpenScholar 证明全文段落检索可将幻觉率降至 0%。我们的规模（15-30篇高优先级论文）远小于 OpenScholar（4500万），可以用极简架构实现同等效果。

## 方案架构

```
Phase 1: PDF获取    →  Phase 2: 多模态解析切分    →  Phase 3: Embedding  →  Phase 4: RAG检索
  下载PDF(PyMuPDF)     文字: PyMuPDF(主力)            bge-base-en-v1.5       ChromaDB向量检索
  并发3路(Semaphore)   图片: 提取+VLM描述             768维                  metadata过滤
  降级:无PDF用摘要      表格: pdfplumber→markdown      存ChromaDB持久化         top_k返回
                       扫描页: PaddleOCR兜底           (含image描述embedding)   含image_path
```

> **技术选型决策**：使用 ChromaDB 而非 numpy 余弦相似度，原因：
> 1. 为 GraphRAG（doc 11）预留扩展——实体/关系 embedding 可存入独立 collection
> 2. ChromaDB 嵌入式零配置，与 SQLite 理念一致，无需额外服务
> 3. 原生 metadata 过滤（paper_id/section），无需手动加载全量矩阵
> 4. 持久化存储，进程重启后无需重新 embedding

### Phase 1: PDF 获取

```python
async def fetch_paper_pdfs(papers: list[Paper], max_concurrent: int = 3):
    """对高优先级论文异步下载PDF"""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def download_one(paper: Paper) -> str | None:
        if not paper.pdf_url:
            return None  # 降级：用摘要
        async with semaphore:
            try:
                resp = await httpx.AsyncClient(timeout=60).get(paper.pdf_url, follow_redirects=True)
                if resp.status_code == 200 and resp.headers.get("content-type", "").startswith("application/pdf"):
                    return resp.content
            except Exception:
                pass
        return None
    
    results = await asyncio.gather(*[download_one(p) for p in papers])
    return dict(zip([p.id for p in papers], results))
```

**PDF 可用性预估**：
| 来源 | 开放获取比例 | pdf_url 来源 |
|------|------------|-------------|
| arXiv | 100% | `link[href]` pdf |
| Semantic Scholar | ~35% | `openAccessPdf.url` |
| CORE | ~70% | `downloadUrl` |
| OpenAlex/Crossref/IEEE | 0% | 无 |

综合约 40-50% 高优先级论文可获取 PDF。

### Phase 2: 多模态解析与切分（图片内联在文字流中）

**核心原则**：图片/表格保留在原始文字位置，不单独提取为 chunk。按阅读顺序重建页面，图片/表格作为内联标记 `[FIGURE: {path}] {描述}` 插入文字流，然后统一切分为 ~250 词的 chunk。这样每个 chunk 包含完整的上下文（文字 + 图片描述），检索时不会丢失图文关联。

```
PDF 页面 → 按坐标排序所有元素（文字块/图片/表格/矢量图）
  → 文字正常输出
  → 遇到图片 → 提取保存 → VLM描述 → 插入 [FIGURE: {path}] {描述}
  → 遇到表格 → pdfplumber提取 → 插入 [TABLE] {markdown}
  → 遇到扫描页 → PaddleOCR兜底
  → 最终文本流（含内联标记）→ 按250词切分为 chunk
```

```
PDF → PyMuPDF 按坐标排序页面元素
  ├─ 文字块 → 正常输出文字
  ├─ 图片区域（嵌入式位图 + 矢量图形密集区）
  │    → 提取保存到 ./paper_assets/{paper_id}/
  │    → VLM 生成描述
  │    → 内联插入: [FIGURE: {path}] {描述}
  ├─ 表格区域 → pdfplumber 提取 → 内联插入: [TABLE] {markdown}
  ├─ 扫描页 → PaddleOCR PP-Structure 兜底（同上内联逻辑）
  → 最终文本流（含内联标记）→ 章节检测 + 250词切分 → chunk
```

#### 2a. 页面元素收集与排序（核心）

```python
import fitz  # PyMuPDF
import os, base64, re

ASSETS_DIR = "./paper_assets"

async def parse_pdf_to_chunks(pdf_bytes: bytes, paper_id: str, llm) -> list[PaperChunk]:
    """解析PDF，按阅读顺序重建文字流，图片/表格内联插入，然后切分为chunk"""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    paper_dir = os.path.join(ASSETS_DIR, paper_id)
    os.makedirs(paper_dir, exist_ok=True)

    full_text_stream = ""  # 完整文字流（含内联标记）
    current_section = "unknown"

    for page_num, page in enumerate(doc):
        page_text = page.get_text("text")
        is_scanned = len(page_text.strip()) < 50

        if is_scanned:
            # 扫描页 → PaddleOCR 兜底（返回内联格式的文字流）
            page_stream = await _paddleocr_page_to_stream(page, paper_id, paper_dir, page_num, llm)
        else:
            # Born-digital → PyMuPDF 按坐标排序
            page_stream = await _pymupdf_page_to_stream(doc, page, paper_id, paper_dir, page_num, llm)

        full_text_stream += "\n" + page_stream

    # 章节检测 + 250词切分
    chunks = _split_into_chunks(full_text_stream, paper_id)
    return chunks


async def _pymupdf_page_to_stream(doc, page, paper_id, paper_dir, page_num, llm) -> str:
    """按坐标排序页面元素，图片/表格内联插入文字流"""
    elements = []

    # 1. 收集文字块
    for block in page.get_text("dict")["blocks"]:
        if "lines" not in block:
            continue
        text = " ".join(
            span["text"] for line in block["lines"] for span in line["spans"]
        ).strip()
        if not text:
            continue
        max_size = max(span["size"] for line in block["lines"] for span in line["spans"])
        is_bold = any("bold" in span["font"].lower() for line in block["lines"] for span in line["spans"])
        elements.append({
            "type": "text",
            "bbox": block["bbox"],  # (x0, y0, x1, y1)
            "text": text,
            "font_size": max_size,
            "is_bold": is_bold,
        })

    # 2. 收集嵌入式位图（带页面坐标）
    for img_info in page.get_images(full=True):
        xref = img_info[0]
        for rect in page.get_image_rects(xref):
            if rect.width < 50 or rect.height < 50:
                continue  # 跳过小图标
            elements.append({
                "type": "image_embedded",
                "bbox": (rect.x0, rect.y0, rect.x1, rect.y1),
                "xref": xref,
            })

    # 3. 收集矢量图形密集区（架构图/流程图）
    drawings = page.get_drawings()
    if len(drawings) > 20:
        # 计算矢量图形的包围盒
        all_x = [d["rect"][0] for d in drawings] + [d["rect"][2] for d in drawings]
        all_y = [d["rect"][1] for d in drawings] + [d["rect"][3] for d in drawings]
        vec_bbox = (min(all_x), min(all_y), max(all_x), max(all_y))
        # 避免与已检测的嵌入式图片重复
        if not _bbox_overlaps_existing(vec_bbox, elements):
            elements.append({
                "type": "vector_figure",
                "bbox": vec_bbox,
            })

    # 4. 按阅读顺序排序（先按Y坐标，再按X坐标；两栏论文自动处理）
    page_width = page.rect.width
    elements.sort(key=lambda e: (
        int(e["bbox"][1] / 10),  # Y坐标量化（每10px一行，避免微小偏移）
        e["bbox"][0] if e["bbox"][0] < page_width / 2 else e["bbox"][0],  # X坐标
    ))

    # 5. 构建文字流，图片/表格内联插入
    stream = ""
    for elem in elements:
        if elem["type"] == "text":
            stream += elem["text"] + " "
        elif elem["type"] in ("image_embedded", "vector_figure"):
            # 提取/保存图片
            img_path = await _save_figure(doc, page, elem, paper_dir, page_num)
            if img_path:
                # VLM 描述
                description = await _vlm_describe_image(img_path, llm)
                # 内联插入到文字流当前位置
                stream += f"\n[FIGURE: {img_path}] {description}\n"

    return stream


def _bbox_overlaps_existing(bbox, elements) -> bool:
    """检查矢量图形包围盒是否与已有图片重叠"""
    x0, y0, x1, y1 = bbox
    for elem in elements:
        if elem["type"] in ("image_embedded",):
            ex0, ey0, ex1, ey1 = elem["bbox"]
            # 计算重叠面积
            ox = max(0, min(x1, ex1) - max(x0, ex0))
            oy = max(0, min(y1, ey1) - max(y0, ey0))
            if ox * oy > 0.5 * (x1 - x0) * (y1 - y0):  # 重叠超过50%
                return True
    return False
```

#### 2b. 图片保存

```python
async def _save_figure(doc, page, elem, paper_dir, page_num) -> str | None:
    """提取并保存图片，返回路径"""
    if elem["type"] == "image_embedded":
        xref = elem["xref"]
        pix = fitz.Pixmap(doc, xref)
        if pix.n > 4:  # CMYK → RGB
            pix = fitz.Pixmap(fitz.csRGB, pix)
        img_path = os.path.join(paper_dir, f"fig_{page_num}_{xref}.png")
        pix.save(img_path)
        pix = None
        return img_path

    elif elem["type"] == "vector_figure":
        # 渲染矢量图形区域为图片
        bbox = elem["bbox"]
        clip = fitz.Rect(bbox)
        mat = fitz.Matrix(2, 2)  # 2x zoom = 144 DPI
        pix = page.get_pixmap(matrix=mat, clip=clip)
        img_path = os.path.join(paper_dir, f"vec_{page_num}.png")
        pix.save(img_path)
        return img_path

    return None
```

#### 2c. VLM 图片描述

```python
FIGURE_VLM_PROMPT = """Describe this figure from an academic paper concisely.
Focus on:
1. Figure type (architecture diagram / results plot / flowchart / example / comparison)
2. Key components, methods, or data shown
3. Main takeaways
Keep under 100 words. Output in Chinese."""

async def _vlm_describe_image(image_path: str, llm) -> str:
    """用 VLM (gpt-4o vision) 生成图片描述"""
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": FIGURE_VLM_PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
        ]},
    ]
    return await llm.chat(messages, temperature=0.3)
```

#### 2d. 表格提取（pdfplumber，内联插入）

```python
import pdfplumber

async def _extract_tables_inline(pdf_bytes: bytes, page_num: int) -> str:
    """提取该页表格，返回 markdown 格式（内联用）"""
    tables_text = ""
    with pdfplumber.open(pdf_bytes) as pdf:
        if page_num >= len(pdf.pages):
            return ""
        page = pdf.pages[page_num]
        for table in page.extract_tables():
            if not table or len(table) < 2:
                continue
            header = table[0]
            rows = table[1:]
            md_lines = ["| " + " | ".join(str(c or "") for c in header) + " |"]
            md_lines.append("| " + " | ".join("---" for _ in header) + " |")
            for row in rows:
                md_lines.append("| " + " | ".join(str(c or "") for c in row) + " |")
            tables_text += "\n[TABLE]\n" + "\n".join(md_lines) + "\n[/TABLE]\n"
    return tables_text
```

#### 2e. PaddleOCR 兜底（扫描页，返回内联文字流）

```python
async def _paddleocr_page_to_stream(page, paper_id, paper_dir, page_num, llm) -> str:
    """扫描页用 PaddleOCR PP-Structure，返回内联格式的文字流"""
    try:
        from paddleocr import PPStructure
        import cv2, numpy as np
    except ImportError:
        logger.warning("PaddleOCR not installed, skipping scanned page %d", page_num)
        return ""

    engine = PPStructure(show_log=False)
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    img = cv2.imdecode(np.frombuffer(pix.tobytes(), np.uint8), cv2.IMREAD_COLOR)

    result = engine(img)
    # PP-Structure 已按位置排序，直接拼接
    stream = ""
    for region in result:
        rtype = region.get("type", "")
        if rtype == "text":
            stream += region.get("res", "") + " "
        elif rtype == "table":
            html = region.get("res", {}).get("html", "")
            if html:
                stream += f"\n[TABLE]\n{html}\n[/TABLE]\n"
        elif rtype == "figure":
            # 提取图片区域
            bbox = region.get("bbox", [0, 0, 0, 0])
            clip = fitz.Rect(bbox)
            fig_pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=clip)
            img_path = os.path.join(paper_dir, f"ocr_fig_{page_num}.png")
            fig_pix.save(img_path)
            description = await _vlm_describe_image(img_path, llm)
            stream += f"\n[FIGURE: {img_path}] {description}\n"
    return stream
```

#### 2f. 章节检测 + 切分

```python
SECTION_KEYWORDS = {
    "method": ["method", "methodology", "approach", "model", "architecture", "framework"],
    "experiment": ["experiment", "evaluation", "results", "setup", "implementation"],
    "introduction": ["introduction", "background", "related work", "preliminar"],
    "conclusion": ["conclusion", "discussion", "future work", "limitation"],
}

def _split_into_chunks(text_stream: str, paper_id: str) -> list[PaperChunk]:
    """将含内联标记的文字流按章节+250词切分为chunk"""
    chunks = []
    current_section = "unknown"
    chunk_buffer = ""
    chunk_index = 0

    for line in text_stream.split("\n"):
        # 章节检测
        text_lower = line.lower().strip()
        is_header = len(text_lower) < 80 and any(
            kw in text_lower for kws in SECTION_KEYWORDS.values() for kw in kws
        )
        if is_header:
            if chunk_buffer:
                chunks.append(_make_chunk(paper_id, chunk_index, current_section, chunk_buffer))
                chunk_index += 1
                chunk_buffer = ""
            for section, keywords in SECTION_KEYWORDS.items():
                if any(kw in text_lower for kw in keywords):
                    current_section = section
                    break
            continue

        chunk_buffer += " " + line
        # [FIGURE] 和 [TABLE] 标记不计入词数，但占用 buffer
        word_count = len(re.sub(r'\[FIGURE:.*?\]|\[TABLE\]|\[/TABLE\]', '', chunk_buffer).split())
        if word_count >= 250:
            chunks.append(_make_chunk(paper_id, chunk_index, current_section, chunk_buffer))
            chunk_index += 1
            chunk_buffer = ""

    if chunk_buffer.strip():
        chunks.append(_make_chunk(paper_id, chunk_index, current_section, chunk_buffer))
    return chunks

def _make_chunk(paper_id, idx, section, text) -> PaperChunk:
    """创建chunk，提取其中的图片路径列表"""
    # 从文本中提取所有 [FIGURE: {path}] 的路径
    figure_paths = re.findall(r'\[FIGURE: (.+?)\]', text)
    return PaperChunk(
        paper_id=paper_id,
        chunk_index=idx,
        section=section,
        chunk_type="text",  # 统一为text，图片/表格是内联标记
        text=text.strip(),
        image_paths_json=json.dumps(figure_paths),  # 该chunk包含的图片路径列表
        page_number=0,
        word_count=len(re.sub(r'\[FIGURE:.*?\]|\[TABLE\]|\[/TABLE\]', '', text).split()),
        extraction_method="pymupdf_inline",
    )
```

#### 无 PDF 的降级

```python
def chunk_from_abstract(paper: Paper) -> list[PaperChunk]:
    """无PDF时，用摘要作为单个chunk"""
    if not paper.abstract:
        return []
    return [PaperChunk(
        paper_id=paper.id, chunk_index=0, section="abstract",
        chunk_type="text", text=paper.abstract, image_paths_json="[]",
        page_number=0, word_count=len(paper.abstract.split()),
        extraction_method="abstract_fallback",
    )]
```

### Phase 3: Embedding + 向量存储

```python
from sentence_transformers import SentenceTransformer
import chromadb

# 全局单例，避免重复加载模型
_embedding_model: SentenceTransformer | None = None
_chroma_client: chromadb.PersistentClient | None = None
_paper_collection = None  # ChromaDB collection for paper chunks

def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        # bge-base: 768维，质量最佳，为GraphRAG实体embedding预留兼容
        _embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    return _embedding_model

def get_chroma_collection():
    """获取ChromaDB paper_chunks collection（持久化）"""
    global _chroma_client, _paper_collection
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path="./chroma_db")
    if _paper_collection is None:
        _paper_collection = _chroma_client.get_or_create_collection(
            name="paper_chunks",
            metadata={"hnsw:space": "cosine"},  # 余弦相似度
        )
    return _paper_collection

def embed_and_store_chunks(chunks: list[PaperChunk]) -> None:
    """批量生成embedding并存入ChromaDB"""
    model = get_embedding_model()
    collection = get_chroma_collection()
    
    texts = [c.text for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=False, batch_size=32)
    
    # 批量插入ChromaDB
    collection.add(
        ids=[c.id for c in chunks],
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=[
            {
                "paper_id": c.paper_id,
                "section": c.section,
                "chunk_index": c.chunk_index,
                "page_number": c.page_number,
            }
            for c in chunks
        ],
    )
```

**模型选择**：
| 模型 | 维度 | 大小 | 速度 | 质量 | GraphRAG兼容 |
|------|------|------|------|------|-------------|
| all-MiniLM-L6-v2 | 384 | ~80MB | 快 | 良好 | 一般 |
| BAAI/bge-small-en-v1.5 | 384 | ~120MB | 快 | 更好 | 良好 |
| **BAAI/bge-base-en-v1.5** | **768** | **~400MB** | **中** | **最佳** | **✅ 推荐** |

推荐 `BAAI/bge-base-en-v1.5`：768维为 GraphRAG 实体/关系 embedding 预留兼容性，质量最佳，CPU批量编码3000段落约60-90秒。

### Phase 4: RAG 检索

```python
def rag_retrieve(
    query: str,
    top_k: int = 10,
    paper_ids: list[str] | None = None,
    section_filter: list[str] | None = None,  # ["method", "experiment"]
) -> list[RetrievedChunk]:
    """从ChromaDB检索最相关段落"""
    model = get_embedding_model()
    collection = get_chroma_collection()
    
    query_emb = model.encode([query])[0]
    
    # 构建metadata过滤条件
    where = {}
    if paper_ids:
        where["paper_id"] = {"$in": paper_ids}
    if section_filter:
        where["section"] = {"$in": section_filter}
    
    results = collection.query(
        query_embeddings=[query_emb.tolist()],
        n_results=top_k,
        where=where if where else None,
    )
    
    return [
        RetrievedChunk(
            chunk_id=results["ids"][0][i],
            text=results["documents"][0][i],
            section=results["metadatas"][0][i].get("section", "unknown"),
            paper_id=results["metadatas"][0][i].get("paper_id"),
            score=1 - results["distances"][0][i],
        )
        for i in range(len(results["ids"][0]))
        if results["distances"][0][i] < 0.7  # 过滤低相关
    ]
```

**性能**：ChromaDB HNSW索引，3000 chunks 检索 <5ms。

**内联图片渲染**：检索返回的 `text` 中包含 `[FIGURE: {path}] {描述}` 标记，前端解析正则 `\[FIGURE: (.+?)\]` 提取图片路径，在对应位置渲染 `<img>` 标签。图片描述作为 alt text。同理 `[TABLE]...[/TABLE]` 标记渲染为表格。

**GraphRAG 扩展预留**：后续可新增 `entity_embeddings` 和 `relation_embeddings` collection，与 `paper_chunks` 共用同一个 ChromaDB client。

## 数据库设计

### SQLite：chunk 元数据（关系查询用）

```sql
CREATE TABLE paper_chunks (
    id TEXT PRIMARY KEY,
    paper_id TEXT NOT NULL REFERENCES papers(id),
    chunk_index INTEGER NOT NULL,
    section TEXT DEFAULT 'unknown',     -- method/experiment/introduction/conclusion/abstract/other
    chunk_type TEXT DEFAULT 'text',     -- 统一为text（图片/表格是内联标记）
    text TEXT NOT NULL,                 -- 文字（含 [FIGURE: {path}] {描述} 和 [TABLE] {markdown} [/TABLE] 内联标记）
    image_paths_json TEXT DEFAULT '[]', -- 该chunk包含的图片路径列表（JSON数组）
    page_number INTEGER DEFAULT 0,      -- 源页码
    word_count INTEGER DEFAULT 0,
    has_pdf BOOLEAN DEFAULT FALSE,
    extraction_method TEXT DEFAULT 'pymupdf_inline',  -- pymupdf_inline/paddleocr/abstract_fallback
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (paper_id) REFERENCES papers(id)
);
CREATE INDEX idx_paper_chunks_paper ON paper_chunks(paper_id);
CREATE INDEX idx_paper_chunks_section ON paper_chunks(section);
```

### ChromaDB：向量索引（语义检索用）

```
Collection: paper_chunks
  - 持久化路径: ./chroma_db/
  - 距离度量: cosine
  - 索引算法: HNSW
  
每条记录:
  - id: chunk_id (与SQLite paper_chunks.id 一致)
  - embedding: 768维 float (bge-base-en-v1.5)
  - document: 段落原文
  - metadata: {paper_id, section, chunk_index}
```

> **设计说明**：chunk 文本同时存 SQLite 和 ChromaDB。SQLite 用于关系查询（如"获取某论文的所有chunk"），ChromaDB 用于语义检索。冗余存储但解耦清晰。

### ORM 模型

```python
class PaperChunk(Base):
    __tablename__ = "paper_chunks"
    
    id = Column(String, primary_key=True, default=_uuid)
    paper_id = Column(String, ForeignKey("papers.id"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    section = Column(Text, default="unknown")
    chunk_type = Column(Text, default="text")          # 统一text（图片/表格内联）
    text = Column(Text, nullable=False)                 # 含 [FIGURE: path] 描述 / [TABLE] markdown [/TABLE] 标记
    image_paths_json = Column(Text, default="[]")       # JSON数组：该chunk包含的图片路径
    page_number = Column(Integer, default=0)
    word_count = Column(Integer, default=0)
    has_pdf = Column(Boolean, default=False)
    extraction_method = Column(Text, default="pymupdf_inline")
    created_at = Column(DateTime, default=_utcnow)
    
    paper = relationship("Paper", backref="chunks")
    # 注意：embedding 不存SQLite，存ChromaDB
```

## 下游集成点

### 1. method_extract 增强（替代现有摘要提取）

```python
# runner.py _score_papers 中
async def _extract_method_via_rag(db, paper: Paper, llm) -> str:
    """从论文全文段落中提取方法信息"""
    # 优先检索 method/experiment 段落
    chunks = rag_retrieve(
        query=f"method architecture algorithm dataset {paper.title}",
        paper_ids=[paper.id],
        section_filter=["method", "experiment"],
        top_k=5,
    )
    if not chunks:
        # 降级：用摘要
        return ""
    
    passage_text = "\n\n".join(c.chunk.text for c in chunks)
    messages = [
        {"role": "system", "content": METHOD_EXTRACT_SYSTEM},
        {"role": "user", "content": f"Title: {paper.title}\n\nPassages:\n{passage_text}"},
    ]
    result = await llm.chat_json(messages, MethodExtract)
    return result.method_extract
```

### 2. 聚类增强（基于真实段落）

```python
# runner.py _build_paper_clusters 中
# 为每篇论文检索代表性段落，而非只用 method_extract 字符串
for paper in high_papers:
    rep_chunks = rag_retrieve(
        query=state.normalized_topic,
        paper_ids=[paper.id],
        section_filter=["method", "experiment"],
        top_k=3,
    )
    paper_context = f"[{paper.id}] {paper.title}\n  代表段落: {rep_chunks[0].chunk.text[:300]}"
```

### 3. 报告生成（RAG 提供支撑证据）

```python
# runner.py _generate_report 中
# 生成报告前，检索相关段落作为上下文
evidence_chunks = rag_retrieve(
    query=state.normalized_topic,
    paper_ids=state.high_priority_paper_ids,
    top_k=20,
)
evidence_text = "\n\n".join(
    f"[{c.chunk.paper_id}] ({c.chunk.section}) {c.chunk.text[:200]}"
    for c in evidence_chunks
)
# 传入 REPORT_USER 作为证据上下文
```

### 4. Idea 生成（跨论文检索）

```python
# runner.py _generate_and_score_ideas 中
# 为每个聚类主题检索相关段落
for cluster in clusters:
    cluster_chunks = rag_retrieve(
        query=cluster.theme,
        paper_ids=cluster.paper_ids,
        top_k=10,
    )
    cluster_context = "\n".join(
        f"[{c.chunk.paper_id}] {c.chunk.text[:200]}"
        for c in cluster_chunks
    )
```

### 5. 引用验证（增强 doc 05）

```python
# idea 生成后，验证 method_sketch 中提到的模型/数据集是否在检索段落中存在
def verify_idea_grounding(idea, retrieved_chunks: list[RetrievedChunk]) -> bool:
    """检查idea中的技术细节是否有段落支撑"""
    all_text = " ".join(c.chunk.text.lower() for c in retrieved_chunks)
    # 提取idea中提到的模型名/数据集名
    for term in idea.key_terms:
        if term.lower() not in all_text:
            idea.warnings.append(f"未在论文中找到 '{term}' 的支撑")
    return len(idea.warnings) == 0
```

## 降级策略

```
PDF下载失败 → 用摘要作为单个chunk（section="abstract"）
PDF解析失败 → 同上
embedding模型加载失败 → 退回关键词检索（jieba分词 + TF-IDF）
RAG检索无结果 → 退回现有逻辑（用method_extract字符串）
```

## 新增依赖

```python
# requirements.txt 新增
PyMuPDF==1.24.10              # PDF文字/图片提取（主力）
pdfplumber==0.11.4            # 表格提取
sentence-transformers==3.3.0  # 本地embedding模型
chromadb==0.5.20              # 嵌入式向量数据库

# 可选依赖（扫描PDF才需要，首次import时检测）
# paddleocr==2.8.1            # OCR兜底 + 版面分析
# paddlepaddle==2.6.2         # PaddleOCR依赖（CPU版）
```

> **PaddleOCR 设为可选**：大部分 arXiv 论文是 born-digital PDF，PyMuPDF 直接提取文字即可。
> PaddleOCR（~2GB）仅在检测到扫描页时才 import，未安装则跳过该页。

### 图片存储

```
./paper_assets/
  └── {paper_id}/
      ├── fig_3_0.png        # 第3页第0个嵌入图片
      ├── fig_5_1.png        # 第5页第1个嵌入图片
      ├── page_7_vector.png  # 第7页矢量图形渲染
      └── ...
```

FastAPI 挂载静态目录：
```python
from fastapi.staticfiles import StaticFiles
app.mount("/paper_assets", StaticFiles(directory="paper_assets"))
```

## 涉及文件

| 文件 | 改动 |
|------|------|
| `backend/requirements.txt` | 新增 PyMuPDF, pdfplumber, sentence-transformers, chromadb（PaddleOCR可选） |
| `backend/app/db/models.py` | 新增 PaperChunk 模型（含 image_paths_json/page_number/extraction_method） |
| `backend/app/db/repositories.py` | 新增 chunk_repo（CRUD + load_chunks） |
| `backend/app/services/rag_service.py` | **新建**：PDF下载、多模态内联解析、ChromaDB存储、检索 |
| `backend/app/agent/runner.py` | `_score_papers` 后新增 `_fetch_and_index_pdfs`；下游函数接入 RAG |
| `backend/app/agent/prompts.py` | 新增 `METHOD_EXTRACT_SYSTEM`, `FIGURE_VLM_PROMPT` |
| `backend/app/schemas/schemas.py` | 新增 `MethodExtract`, `RetrievedChunk`（含 image_paths 列表） |
| `backend/app/main.py` | 挂载 `/paper_assets` 静态目录 |

## 实现步骤

### Step 1: 基础设施（~2h）
1. 安装依赖（PyMuPDF, pdfplumber, sentence-transformers, chromadb），验证可用
2. 新增 PaperChunk 模型（含 chunk_type/image_path/page_number）+ 数据库迁移
3. 实现 `rag_service.py`：PDF下载 + PyMuPDF文字提取 + chunk存储

### Step 2: 多模态提取（~2h）
1. 图片提取（嵌入式位图 + 矢量图形渲染）+ 保存到 paper_assets/
2. VLM 描述（gpt-4o vision，选择性：跳过小图标）
3. 表格提取（pdfplumber → markdown）
4. PaddleOCR 兜底（扫描页，可选依赖）
5. FastAPI 挂载 /paper_assets 静态目录

### Step 3: Embedding + 检索（~1h）
1. 实现 bge-base embedding + ChromaDB存储
2. 实现 `rag_retrieve` 函数（含 image_path 返回）
3. 单元测试：下载一篇 arXiv PDF → 解析 → 检索 → 验证图片路径可用

### Step 4: 下游集成（~2h）
1. `_score_papers` 后调用 `_fetch_and_index_pdfs`
2. `method_extract` 改用 RAG 段落提取
3. `_build_paper_clusters` 接入 RAG
4. `_generate_report` 接入 RAG 证据检索
5. `_generate_and_score_ideas` 接入 RAG 跨论文检索

### Step 5: 降级与容错（~1h）
1. 无 PDF 论文的降级处理
2. VLM 不可用时的降级（跳过图片描述，只存路径）
3. PaddleOCR 未安装时的降级（跳过扫描页）
4. 前端进度展示（"正在下载并解析论文全文..."）

## 性能预估

| 阶段 | 耗时（15篇高优先级论文） |
|------|----------------------|
| PDF下载（3并发） | ~30s |
| 文字解析+切分（PyMuPDF） | ~10s |
| 图片提取+VLM描述（~5图/篇, gpt-4o vision） | ~60-90s |
| 表格提取（pdfplumber） | ~5s |
| Embedding（~3000 chunks, bge-base, CPU） | ~60-90s |
| ChromaDB写入 | ~2s |
| RAG检索（单次, HNSW） | <5ms |
| **总计新增** | **~3-4min** |

## 与现有文档的关系

| 文档 | 关系 |
|------|------|
| [05-hallucination-guard.md](05-hallucination-guard.md) | 互补：05 是 prompt 层防护（已实现），10 是数据层根因修复 |
| [09-architecture-upgrade.md](09-architecture-upgrade.md) P1-A | 10 是 P1-A 的详细实现方案 |
| [02-ideas-traceability.md](02-ideas-traceability.md) | 增强：RAG 使 idea 可追溯到具体段落，不仅是论文 ID |
| [11-graphrag-planning.md](11-graphrag-planning.md) | 前置：10 的 ChromaDB + bge-base 为 GraphRAG 预留扩展基础 |
