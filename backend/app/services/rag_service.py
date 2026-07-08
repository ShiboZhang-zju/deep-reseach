"""RAG service: PDF download, multi-modal parsing, embedding, and retrieval.

Pipeline:
  1. Download PDFs for high-priority papers
  2. Parse with PyMuPDF (text + inline figures + tables)
  3. VLM description for figures (gpt-4o vision)
  4. PaddleOCR fallback for scanned pages
  5. Embedding with sentence-transformers (bge-base)
  6. ChromaDB storage and retrieval
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)

# === Constants ===

ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "paper_assets")
CHROMA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "chroma_db")
CHUNK_WORD_LIMIT = 250
PDF_DOWNLOAD_TIMEOUT = 60
MAX_CONCURRENT_DOWNLOADS = 3

SECTION_KEYWORDS = {
    "method": ["method", "methodology", "approach", "model", "architecture", "framework", "design"],
    "experiment": ["experiment", "evaluation", "results", "setup", "implementation", "ablation"],
    "introduction": ["introduction", "background", "related work", "preliminar", "overview"],
    "conclusion": ["conclusion", "discussion", "future work", "limitation", "summary"],
}


# === Data classes ===

@dataclass
class ParsedChunk:
    """A parsed chunk from PDF, before DB storage."""
    chunk_index: int
    section: str = "unknown"
    chunk_type: str = "text"
    text: str = ""
    image_paths: list[str] = field(default_factory=list)
    page_number: int = 0
    word_count: int = 0
    has_pdf: bool = True
    extraction_method: str = "pymupdf_inline"


# === Singleton: Embedding model ===

_embedding_model = None


def get_embedding_model():
    """Lazy-load sentence-transformers model (singleton)."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model BAAI/bge-base-en-v1.5 (first time ~400MB download)...")
        _embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
        logger.info("Embedding model loaded.")
    return _embedding_model


# === Singleton: ChromaDB ===

_chroma_client = None
_paper_collection = None


def get_chroma_collection():
    """Get or create the ChromaDB paper_chunks collection."""
    global _chroma_client, _paper_collection
    if _chroma_client is None:
        import chromadb
        os.makedirs(CHROMA_DIR, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    if _paper_collection is None:
        _paper_collection = _chroma_client.get_or_create_collection(
            name="paper_chunks",
            metadata={"hnsw:space": "cosine"},
        )
    return _paper_collection


# === Phase 1: PDF Download ===

async def download_pdf(pdf_url: str) -> bytes | None:
    """Download a PDF from URL. Returns None on failure."""
    try:
        async with httpx.AsyncClient(timeout=PDF_DOWNLOAD_TIMEOUT, follow_redirects=True) as client:
            resp = await client.get(pdf_url, headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code == 200:
                content_type = resp.headers.get("content-type", "")
                if "pdf" in content_type or resp.content[:4] == b"%PDF":
                    return resp.content
                logger.warning("PDF download: unexpected content-type %s for %s", content_type, pdf_url[:80])
            else:
                logger.warning("PDF download failed %d for %s", resp.status_code, pdf_url[:80])
    except Exception as e:
        logger.warning("PDF download error for %s: %s", pdf_url[:80], e)
    return None


async def download_papers_pdfs(papers: list, max_concurrent: int = MAX_CONCURRENT_DOWNLOADS) -> dict[str, bytes | None]:
    """Download PDFs for multiple papers concurrently. Returns {paper_id: pdf_bytes_or_None}."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def download_one(paper) -> tuple[str, bytes | None]:
        if not paper.pdf_url:
            return paper.id, None
        async with semaphore:
            pdf_bytes = await download_pdf(paper.pdf_url)
            return paper.id, pdf_bytes

    results = await asyncio.gather(*[download_one(p) for p in papers])
    return dict(results)


# === Phase 2: PDF Parsing (inline figures/tables) ===

async def parse_pdf_to_chunks(pdf_bytes: bytes, paper_id: str, llm) -> list[ParsedChunk]:
    """Parse PDF to chunks with inline figures and tables."""
    import fitz  # PyMuPDF

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    paper_dir = os.path.join(ASSETS_DIR, paper_id)
    os.makedirs(paper_dir, exist_ok=True)

    full_text_stream = ""

    for page_num, page in enumerate(doc):
        page_text = page.get_text("text")
        is_scanned = len(page_text.strip()) < 50

        if is_scanned:
            page_stream = await _paddleocr_page_to_stream(page, paper_id, paper_dir, page_num, llm)
        else:
            page_stream = await _pymupdf_page_to_stream(doc, page, paper_id, paper_dir, page_num, llm)

        full_text_stream += "\n" + page_stream

    # Extract tables with pdfplumber (add to text stream)
    table_text = _extract_tables_with_pdfplumber(pdf_bytes)
    if table_text:
        full_text_stream += "\n" + table_text

    chunks = _split_into_chunks(full_text_stream, paper_id)
    return chunks


async def _pymupdf_page_to_stream(doc, page, paper_id: str, paper_dir: str, page_num: int, llm) -> str:
    """Parse a born-digital page: sort elements by position, insert figures inline."""
    import fitz

    elements = []

    # 1. Collect text blocks with font info
    for block in page.get_text("dict")["blocks"]:
        if "lines" not in block:
            continue
        text = " ".join(
            span["text"] for line in block["lines"] for span in line["spans"]
        ).strip()
        if not text:
            continue
        max_size = max((span["size"] for line in block["lines"] for span in line["spans"]), default=10)
        is_bold = any(
            "bold" in span["font"].lower()
            for line in block["lines"] for span in line["spans"]
        )
        elements.append({
            "type": "text",
            "bbox": block["bbox"],
            "text": text,
            "font_size": max_size,
            "is_bold": is_bold,
        })

    # 2. Collect embedded images with page positions
    for img_info in page.get_images(full=True):
        xref = img_info[0]
        try:
            for rect in page.get_image_rects(xref):
                if rect.width < 50 or rect.height < 50:
                    continue
                elements.append({
                    "type": "image_embedded",
                    "bbox": (rect.x0, rect.y0, rect.x1, rect.y1),
                    "xref": xref,
                })
        except Exception:
            pass

    # 3. Detect vector graphics clusters (architecture diagrams)
    drawings = page.get_drawings()
    if len(drawings) > 20:
        all_x0 = [d["rect"][0] for d in drawings]
        all_y0 = [d["rect"][1] for d in drawings]
        all_x1 = [d["rect"][2] for d in drawings]
        all_y1 = [d["rect"][3] for d in drawings]
        vec_bbox = (min(all_x0), min(all_y0), max(all_x1), max(all_y1))
        # Skip if overlaps with existing embedded images
        if not _bbox_overlaps_existing(vec_bbox, elements):
            elements.append({
                "type": "vector_figure",
                "bbox": vec_bbox,
            })

    # 4. Sort by reading order (Y first, then X)
    elements.sort(key=lambda e: (int(e["bbox"][1] / 10), e["bbox"][0]))

    # 5. Build text stream with inline figure markers
    stream = ""
    for elem in elements:
        if elem["type"] == "text":
            stream += elem["text"] + " "
        elif elem["type"] in ("image_embedded", "vector_figure"):
            img_path = _save_figure(doc, page, elem, paper_dir, page_num)
            if img_path:
                description = await _vlm_describe_image(img_path, llm)
                stream += f"\n[FIGURE: {img_path}] {description}\n"

    return stream


def _bbox_overlaps_existing(bbox: tuple, elements: list[dict]) -> bool:
    """Check if a bbox overlaps significantly with existing image elements."""
    x0, y0, x1, y1 = bbox
    area = (x1 - x0) * (y1 - y0)
    if area <= 0:
        return False
    for elem in elements:
        if elem["type"] not in ("image_embedded",):
            continue
        ex0, ey0, ex1, ey1 = elem["bbox"]
        ox = max(0, min(x1, ex1) - max(x0, ex0))
        oy = max(0, min(y1, ey1) - max(y0, ey0))
        if ox * oy > 0.5 * area:
            return True
    return False


def _save_figure(doc, page, elem: dict, paper_dir: str, page_num: int) -> str | None:
    """Extract and save a figure image. Returns the file path."""
    import fitz

    try:
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
            bbox = elem["bbox"]
            clip = fitz.Rect(bbox)
            mat = fitz.Matrix(2, 2)  # 2x zoom = 144 DPI
            pix = page.get_pixmap(matrix=mat, clip=clip)
            img_path = os.path.join(paper_dir, f"vec_{page_num}.png")
            pix.save(img_path)
            return img_path
    except Exception as e:
        logger.warning("Failed to save figure on page %d: %s", page_num, e)
    return None


async def _vlm_describe_image(image_path: str, llm) -> str:
    """Use VLM (gpt-4o vision) to describe a figure."""
    try:
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()

        from app.agent.prompts import FIGURE_VLM_PROMPT
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": FIGURE_VLM_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
            ]},
        ]
        description = await llm.chat(messages, temperature=0.3)
        return description.strip()
    except Exception as e:
        logger.warning("VLM description failed for %s: %s", image_path, e)
        return "图片描述不可用"


def _extract_tables_with_pdfplumber(pdf_bytes: bytes) -> str:
    """Extract tables from PDF using pdfplumber, return as inline markdown."""
    try:
        import pdfplumber
    except ImportError:
        return ""

    tables_text = ""
    try:
        import io
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page_num, page in enumerate(pdf.pages):
                for table in page.extract_tables():
                    if not table or len(table) < 2:
                        continue
                    header = table[0]
                    rows = table[1:]
                    md_lines = ["| " + " | ".join(str(c or "") for c in header) + " |"]
                    md_lines.append("| " + " | ".join("---" for _ in header) + " |")
                    for row in rows:
                        md_lines.append("| " + " | ".join(str(c or "") for c in row) + " |")
                    tables_text += f"\n[TABLE]\n" + "\n".join(md_lines) + "\n[/TABLE]\n"
    except Exception as e:
        logger.warning("pdfplumber table extraction failed: %s", e)

    return tables_text


async def _paddleocr_page_to_stream(page, paper_id: str, paper_dir: str, page_num: int, llm) -> str:
    """Fallback: parse scanned page with PaddleOCR PP-Structure."""
    try:
        from paddleocr import PPStructure
        import cv2
        import numpy as np
        import fitz
    except ImportError:
        logger.warning("PaddleOCR not installed, skipping scanned page %d", page_num)
        return ""

    try:
        engine = PPStructure(show_log=False)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img = cv2.imdecode(np.frombuffer(pix.tobytes(), np.uint8), cv2.IMREAD_COLOR)

        result = engine(img)
        stream = ""
        for region in result:
            rtype = region.get("type", "")
            if rtype == "text":
                res = region.get("res", "")
                if isinstance(res, list):
                    res = " ".join(item.get("text", "") for item in res if isinstance(item, dict))
                stream += str(res) + " "
            elif rtype == "table":
                html = region.get("res", {}).get("html", "")
                if html:
                    stream += f"\n[TABLE]\n{html}\n[/TABLE]\n"
            elif rtype == "figure":
                bbox = region.get("bbox", [0, 0, 0, 0])
                clip = fitz.Rect(bbox)
                fig_pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=clip)
                img_path = os.path.join(paper_dir, f"ocr_fig_{page_num}.png")
                fig_pix.save(img_path)
                description = await _vlm_describe_image(img_path, llm)
                stream += f"\n[FIGURE: {img_path}] {description}\n"
        return stream
    except Exception as e:
        logger.warning("PaddleOCR failed on page %d: %s", page_num, e)
        return ""


# === Chunk splitting ===

def _classify_section(header_text: str) -> str | None:
    """Classify a header text into a section name."""
    text_lower = header_text.lower().strip()
    if len(text_lower) > 80:
        return None
    for section, keywords in SECTION_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return section
    return None


def _split_into_chunks(text_stream: str, paper_id: str) -> list[ParsedChunk]:
    """Split text stream (with inline markers) into ~250 word chunks."""
    chunks = []
    current_section = "unknown"
    chunk_buffer = ""
    chunk_index = 0

    for line in text_stream.split("\n"):
        # Section detection
        detected = _classify_section(line.strip())
        if detected:
            if chunk_buffer.strip():
                chunks.append(_make_chunk(chunk_index, current_section, chunk_buffer))
                chunk_index += 1
                chunk_buffer = ""
            current_section = detected
            continue

        chunk_buffer += " " + line
        # Count words excluding inline markers
        word_count = len(re.sub(r'\[FIGURE:.*?\]|\[TABLE\]|\[/TABLE\]', '', chunk_buffer).split())
        if word_count >= CHUNK_WORD_LIMIT:
            chunks.append(_make_chunk(chunk_index, current_section, chunk_buffer))
            chunk_index += 1
            chunk_buffer = ""

    if chunk_buffer.strip():
        chunks.append(_make_chunk(chunk_index, current_section, chunk_buffer))

    return chunks


def _make_chunk(idx: int, section: str, text: str) -> ParsedChunk:
    """Create a ParsedChunk, extracting image paths from inline markers."""
    figure_paths = re.findall(r'\[FIGURE: (.+?)\]', text)
    clean_text = re.sub(r'\[FIGURE:.*?\]|\[TABLE\]|\[/TABLE\]', '', text).split()
    word_count = len(clean_text)
    return ParsedChunk(
        chunk_index=idx,
        section=section,
        chunk_type="text",
        text=text.strip(),
        image_paths=figure_paths,
        word_count=word_count,
        has_pdf=True,
        extraction_method="pymupdf_inline",
    )


# === Fallback: abstract as single chunk ===

def chunk_from_abstract(paper) -> list[ParsedChunk]:
    """Create a single chunk from paper abstract (fallback when no PDF)."""
    if not paper.abstract:
        return []
    return [ParsedChunk(
        chunk_index=0,
        section="abstract",
        chunk_type="text",
        text=paper.abstract,
        image_paths=[],
        page_number=0,
        word_count=len(paper.abstract.split()),
        has_pdf=False,
        extraction_method="abstract_fallback",
    )]


# === Phase 3: Embedding + ChromaDB storage ===

def embed_and_store_chunks(paper_id: str, chunks: list[ParsedChunk]) -> None:
    """Generate embeddings and store in ChromaDB."""
    if not chunks:
        return

    model = get_embedding_model()
    collection = get_chroma_collection()

    # Delete existing chunks for this paper in ChromaDB
    try:
        existing = collection.get(where={"paper_id": paper_id})
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
    except Exception:
        pass

    texts = [c.text for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=False, batch_size=32)

    chunk_ids = [f"{paper_id}_chunk_{c.chunk_index}" for c in chunks]
    collection.add(
        ids=chunk_ids,
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=[
            {
                "paper_id": paper_id,
                "section": c.section,
                "chunk_index": c.chunk_index,
                "page_number": c.page_number,
            }
            for c in chunks
        ],
    )
    logger.info("Stored %d chunks for paper %s in ChromaDB", len(chunks), paper_id[:8])


# === Phase 4: RAG Retrieval ===

def rag_retrieve(
    query: str,
    top_k: int = 10,
    paper_ids: list[str] | None = None,
    section_filter: list[str] | None = None,
) -> list[dict]:
    """Retrieve relevant chunks from ChromaDB.

    Returns list of dicts: {chunk_id, text, section, paper_id, score, image_paths}
    """
    collection = get_chroma_collection()

    # Build metadata filter
    where = {}
    if paper_ids:
        where["paper_id"] = {"$in": paper_ids}
    if section_filter:
        where["section"] = {"$in": section_filter}

    # Use ChromaDB's built-in query (it handles embedding internally if we pass text)
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        where=where if where else None,
    )

    if not results["ids"] or not results["ids"][0]:
        return []

    retrieved = []
    for i in range(len(results["ids"][0])):
        distance = results["distances"][0][i]
        score = 1 - distance  # cosine distance → similarity
        if distance > 0.7:  # filter low relevance
            continue
        metadata = results["metadatas"][0][i]
        text = results["documents"][0][i]
        # Extract image paths from inline markers
        image_paths = re.findall(r'\[FIGURE: (.+?)\]', text)
        retrieved.append({
            "chunk_id": results["ids"][0][i],
            "text": text,
            "section": metadata.get("section", "unknown"),
            "paper_id": metadata.get("paper_id", ""),
            "score": score,
            "image_paths": image_paths,
        })

    return retrieved


# === Main entry point ===

async def fetch_and_index_papers(papers: list, llm, task_id: str = "") -> dict:
    """Download PDFs, parse, and index papers for RAG.

    Returns summary dict with counts.
    """
    from app.db.session import SessionLocal
    from app.db.repositories.paper_repo import save_chunks, has_chunks
    from app.services.event_service import emit_event

    if not papers:
        return {"total": 0, "pdf_success": 0, "fallback": 0, "failed": 0}

    total = len(papers)
    logger.info("RAG: Processing %d papers for PDF download and indexing", total)

    if task_id:
        emit_event(task_id, "status", {"status": "indexing_pdfs", "total": total})

    # 1. Download PDFs
    pdf_results = await download_papers_pdfs(papers)

    pdf_success = 0
    fallback = 0
    failed = 0

    # 2. Parse and index each paper
    for paper in papers:
        pdf_bytes = pdf_results.get(paper.id)

        if pdf_bytes:
            try:
                chunks = await parse_pdf_to_chunks(pdf_bytes, paper.id, llm)
                if not chunks:
                    chunks = chunk_from_abstract(paper)
                    fallback += 1
                else:
                    pdf_success += 1
            except Exception as e:
                logger.error("PDF parsing failed for paper %s: %s", paper.id[:8], e)
                chunks = chunk_from_abstract(paper)
                fallback += 1
        else:
            chunks = chunk_from_abstract(paper)
            if paper.abstract:
                fallback += 1
            else:
                failed += 1
                continue

        # 3. Save to SQLite
        db = SessionLocal()
        try:
            chunks_data = [
                {
                    "chunk_index": c.chunk_index,
                    "section": c.section,
                    "chunk_type": c.chunk_type,
                    "text": c.text,
                    "image_paths": c.image_paths,
                    "page_number": c.page_number,
                    "word_count": c.word_count,
                    "has_pdf": c.has_pdf,
                    "extraction_method": c.extraction_method,
                }
                for c in chunks
            ]
            save_chunks(db, paper.id, chunks_data)
            db.commit()
        except Exception as e:
            logger.error("Failed to save chunks for paper %s: %s", paper.id[:8], e)
            db.rollback()
        finally:
            db.close()

        # 4. Embed and store in ChromaDB
        try:
            embed_and_store_chunks(paper.id, chunks)
        except Exception as e:
            logger.error("Embedding failed for paper %s: %s", paper.id[:8], e)

    summary = {"total": total, "pdf_success": pdf_success, "fallback": fallback, "failed": failed}
    logger.info("RAG indexing complete: %s", summary)

    if task_id:
        emit_event(task_id, "status", {"status": "indexing_complete", **summary})

    return summary
