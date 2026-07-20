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
PDF_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "pdf_cache")
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
_paper_level_collection = None


def get_chroma_collection():
    """Get or create the ChromaDB paper_chunks collection (chunk-level embeddings)."""
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


def get_paper_level_collection():
    """Get or create the ChromaDB paper_embeddings collection (one embedding per paper).
    Used for literature map semantic similarity.
    """
    global _chroma_client, _paper_level_collection
    if _chroma_client is None:
        import chromadb
        os.makedirs(CHROMA_DIR, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    if _paper_level_collection is None:
        _paper_level_collection = _chroma_client.get_or_create_collection(
            name="paper_embeddings",
            metadata={"hnsw:space": "cosine"},
        )
    return _paper_level_collection


def ensure_paper_embedding(paper_id: str, title: str, abstract: str, chunks: list = None):
    """Compute and store a paper-level embedding if not already present.
    
    Uses the mean of chunk embeddings if chunks are available,
    otherwise embeds title + abstract directly.
    """
    collection = get_paper_level_collection()

    # Check if already exists
    try:
        existing = collection.get(ids=[paper_id])
        if existing["ids"]:
            return  # Already has paper-level embedding
    except Exception as e:
        logger.debug("ChromaDB check for paper %s failed: %s", paper_id[:8], e)

    model = get_embedding_model()

    if chunks:
        # Mean of chunk embeddings (better quality)
        chunk_texts = [c.text if hasattr(c, 'text') else c.get("text", "") for c in chunks]
        if chunk_texts:
            chunk_embeddings = model.encode(chunk_texts, show_progress_bar=False)
            paper_embedding = chunk_embeddings.mean(axis=0).tolist()
        else:
            paper_embedding = model.encode([f"{title} {abstract}"], show_progress_bar=False).tolist()[0]
    else:
        # Embed title + abstract
        text = f"{title} {abstract or ''}"
        paper_embedding = model.encode([text], show_progress_bar=False).tolist()[0]

    collection.upsert(
        ids=[paper_id],
        embeddings=[paper_embedding],
        documents=[f"{title}"],
        metadatas=[{"paper_id": paper_id, "title": title}],
    )


def get_paper_similarities(paper_ids: list[str], top_k_per_paper: int = 5) -> dict[str, list[dict]]:
    """For each paper, find its most similar papers from the paper_embeddings collection.
    
    Returns {paper_id: [{paper_id, title, score}, ...]}
    """
    collection = get_paper_level_collection()
    if not paper_ids:
        return {}

    # Fetch embeddings for the given paper_ids
    existing = collection.get(ids=paper_ids)
    if not existing["ids"]:
        return {}

    results = {}
    for i, pid in enumerate(existing["ids"]):
        emb = existing["embeddings"][i]
        query_result = collection.query(
            query_embeddings=[emb],
            n_results=top_k_per_paper + 1,  # +1 because the paper itself will be included
        )
        similar = []
        for j, result_id in enumerate(query_result["ids"][0]):
            if result_id == pid:
                continue  # Skip self
            distance = query_result["distances"][0][j]
            score = 1 - distance
            if score < 0.5:
                continue  # Filter low similarity
            similar.append({
                "paper_id": result_id,
                "title": query_result["metadatas"][0][j].get("title", ""),
                "score": score,
            })
        results[pid] = similar[:top_k_per_paper]

    return results


# === Phase 1: PDF Download ===

async def _try_download(url: str, timeout: int = PDF_DOWNLOAD_TIMEOUT) -> bytes | None:
    """Download a URL and verify it's a PDF. Returns None on failure."""
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code == 200:
                content_type = resp.headers.get("content-type", "")
                if "pdf" in content_type or resp.content[:4] == b"%PDF":
                    return resp.content
                logger.warning("Download: unexpected content-type %s for %s", content_type, url[:80])
            else:
                logger.warning("Download failed %d for %s", resp.status_code, url[:80])
    except Exception as e:
        logger.warning("Download error for %s: %s", url[:80], e)
    return None


async def _fetch_arxiv_pdf(arxiv_id: str) -> bytes | None:
    """Download PDF from arXiv by arxiv_id."""
    # Clean arxiv_id (remove version suffix for URL, arXiv handles both)
    clean_id = arxiv_id.split("v")[0] if "v" in arxiv_id[-2:] else arxiv_id
    url = f"https://arxiv.org/pdf/{clean_id}"
    pdf = await _try_download(url)
    if pdf:
        logger.info("arXiv PDF downloaded for %s", arxiv_id)
    return pdf


async def _fetch_unpaywall_pdf(doi: str, email: str = "research@example.com") -> bytes | None:
    """Try Unpaywall API to find an open-access PDF for a DOI."""
    url = f"https://api.unpaywall.org/v2/{doi}?email={email}"
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                return None
            data = resp.json()
            
            # Try best_oa_location first
            best_loc = data.get("best_oa_location")
            if best_loc and best_loc.get("url_for_pdf"):
                pdf = await _try_download(best_loc["url_for_pdf"])
                if pdf:
                    logger.info("Unpaywall PDF downloaded for DOI %s", doi)
                    return pdf
            
            # Try all OA locations
            for loc in data.get("oa_locations", []):
                if loc.get("url_for_pdf"):
                    pdf = await _try_download(loc["url_for_pdf"])
                    if pdf:
                        logger.info("Unpaywall (alt) PDF downloaded for DOI %s", doi)
                        return pdf
    except Exception as e:
        logger.warning("Unpaywall lookup failed for %s: %s", doi, e)
    return None


async def _fetch_scihub_pdf(doi: str) -> bytes | None:
    """Try Sci-Hub as last resort. Note: may be unreliable."""
    if not doi:
        return None
    # Try multiple Sci-Hub mirrors
    mirrors = ["sci-hub.se", "sci-hub.st", "sci-hub.ru"]
    for mirror in mirrors:
        url = f"https://{mirror}/{doi}"
        try:
            async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
                resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
                if resp.status_code != 200:
                    continue
                # Sci-Hub returns HTML with embedded PDF iframe or download link
                text = resp.text
                # Look for PDF iframe src
                import re
                pdf_url_match = re.search(r'<iframe[^>]+src="([^"]+\.pdf[^"]*)"', text)
                if pdf_url_match:
                    pdf_url = pdf_url_match.group(1)
                    if pdf_url.startswith("//"):
                        pdf_url = "https:" + pdf_url
                    elif pdf_url.startswith("/"):
                        pdf_url = f"https://{mirror}" + pdf_url
                    pdf = await _try_download(pdf_url)
                    if pdf:
                        logger.info("Sci-Hub (%s) PDF downloaded for DOI %s", mirror, doi)
                        return pdf
                
                # Try direct download button
                download_match = re.search(r'<a[^>]+href="([^"]+\.pdf[^"]*)"[^>]*>download', text, re.I)
                if download_match:
                    pdf_url = download_match.group(1)
                    if pdf_url.startswith("//"):
                        pdf_url = "https:" + pdf_url
                    pdf = await _try_download(pdf_url)
                    if pdf:
                        logger.info("Sci-Hub (%s, button) PDF downloaded for DOI %s", mirror, doi)
                        return pdf
        except Exception as e:
            logger.warning("Sci-Hub (%s) download failed for %s: %s", mirror, doi, e)
            continue
    return None


async def _fetch_s2_pdf(s2_id: str) -> bytes | None:
    """Try Semantic Scholar API to get PDF link."""
    if not s2_id:
        return None
    url = f"https://api.semanticscholar.org/graph/v1/paper/{s2_id}?fields=openAccessPdf"
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                return None
            data = resp.json()
            oa_pdf = data.get("openAccessPdf")
            if oa_pdf and oa_pdf.get("url"):
                pdf = await _try_download(oa_pdf["url"])
                if pdf:
                    logger.info("S2 PDF downloaded for paper %s", s2_id[:12])
                    return pdf
    except Exception as e:
        logger.warning("S2 PDF lookup failed for %s: %s", s2_id[:12], e)
    return None


async def _fetch_openalex_pdf(openalex_id: str) -> bytes | None:
    """Try OpenAlex API to get PDF link."""
    if not openalex_id:
        return None
    # Ensure it has the full ID format
    if not openalex_id.startswith("https"):
        openalex_id = f"https://openalex.org/{openalex_id}"
    url = f"https://api.openalex.org/works/{openalex_id.split('/')[-1]}?select=best_oa_location,open_access,primary_location"
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                return None
            data = resp.json()
            # Try best_oa_location
            best_oa = data.get("best_oa_location")
            if best_oa and best_oa.get("pdf_url"):
                pdf = await _try_download(best_oa["pdf_url"])
                if pdf:
                    logger.info("OpenAlex PDF downloaded for %s", openalex_id[-12:])
                    return pdf
            # Try primary_location
            primary = data.get("primary_location")
            if primary and primary.get("pdf_url"):
                pdf = await _try_download(primary["pdf_url"])
                if pdf:
                    logger.info("OpenAlex (primary) PDF downloaded for %s", openalex_id[-12:])
                    return pdf
            # Try open_access field
            oa = data.get("open_access")
            if oa and oa.get("oa_url"):
                pdf = await _try_download(oa["oa_url"])
                if pdf:
                    logger.info("OpenAlex (OA) PDF downloaded for %s", openalex_id[-12:])
                    return pdf
    except Exception as e:
        logger.warning("OpenAlex PDF lookup failed for %s: %s", openalex_id[-12:], e)
    return None


async def _fetch_crossref_pdf(doi: str) -> bytes | None:
    """Try Crossref API to get PDF link from links field."""
    if not doi:
        return None
    url = f"https://api.crossref.org/works/{doi}"
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url, headers={"User-Agent": "DeepResearch/1.0 (mailto:research@example.com)"})
            if resp.status_code != 200:
                return None
            data = resp.json()
            message = data.get("message", {})
            links = message.get("link", [])
            for link in links:
                url_field = link.get("URL", "")
                content_type = link.get("content-type", "")
                if "pdf" in content_type or url_field.endswith(".pdf"):
                    pdf = await _try_download(url_field)
                    if pdf:
                        logger.info("Crossref PDF downloaded for DOI %s", doi)
                        return pdf
    except Exception as e:
        logger.warning("Crossref PDF lookup failed for %s: %s", doi, e)
    return None


async def download_pdf_multi_source(paper) -> bytes | None:
    """Try multiple sources to download a paper's PDF.
    
    Order of attempts:
    0. Disk cache (if previously downloaded)
    1. Existing pdf_url (if any)
    2. arXiv (if arxiv_id available)
    3. DOI → arXiv extraction (DOI like 10.48550/arXiv.XXXX)
    4. Semantic Scholar (if s2_id available)
    5. OpenAlex (if openalex_id available)
    6. Unpaywall (if DOI available)
    7. Crossref links (if DOI available)
    
    P0-4: Sci-Hub (step 8) is DISABLED by default for legal compliance.
    Set ENABLE_SCIHUB=true in .env to opt-in.
    
    Successfully downloaded PDFs are cached to disk for reuse.
    """
    from app.config import settings
    
    # 0. Check disk cache first
    cached = _load_pdf_from_cache(paper.id)
    if cached:
        logger.info("PDF loaded from cache for paper %s", paper.id[:8])
        return cached
    
    pdf_bytes = None
    
    # 1. Try existing pdf_url
    if paper.pdf_url:
        pdf_bytes = await _try_download(paper.pdf_url)
    
    # 2. Try arXiv
    if not pdf_bytes and paper.arxiv_id:
        pdf_bytes = await _fetch_arxiv_pdf(paper.arxiv_id)
    
    # 3. Extract arXiv ID from DOI (e.g., 10.48550/arXiv.2602.21394)
    if not pdf_bytes and paper.doi:
        arxiv_id_from_doi = _extract_arxiv_from_doi(paper.doi)
        if arxiv_id_from_doi:
            pdf_bytes = await _fetch_arxiv_pdf(arxiv_id_from_doi)
            # Also save the arxiv_id to the paper for future use
            if pdf_bytes and not paper.arxiv_id:
                paper.arxiv_id = arxiv_id_from_doi
    
    # 4. Try Semantic Scholar
    if not pdf_bytes and paper.semantic_scholar_id:
        pdf_bytes = await _fetch_s2_pdf(paper.semantic_scholar_id)
    
    # 5. Try OpenAlex
    if not pdf_bytes and paper.openalex_id:
        pdf_bytes = await _fetch_openalex_pdf(paper.openalex_id)
    
    # 6. Try Unpaywall (open access)
    if not pdf_bytes and paper.doi:
        pdf_bytes = await _fetch_unpaywall_pdf(paper.doi)
    
    # 7. Try Crossref links
    if not pdf_bytes and paper.doi:
        pdf_bytes = await _fetch_crossref_pdf(paper.doi)
    
    # 8. Try Sci-Hub (last resort) — DISABLED by default (P0-4: legal compliance)
    if not pdf_bytes and paper.doi and settings.enable_scihub:
        logger.info("Attempting Sci-Hub download for DOI %s (ENABLE_SCIHUB=true)", paper.doi)
        pdf_bytes = await _fetch_scihub_pdf(paper.doi)
    
    # Cache to disk if successful
    if pdf_bytes:
        _save_pdf_to_cache(paper.id, pdf_bytes)
    
    return pdf_bytes


def _extract_arxiv_from_doi(doi: str) -> str | None:
    """Extract arXiv ID from a DOI like '10.48550/arXiv.2602.21394'."""
    if not doi:
        return None
    # Match patterns like 10.48550/arXiv.XXXX.XXXXX
    import re
    match = re.match(r'10\.48550/arXiv\.([\d.]+)', doi, re.IGNORECASE)
    if match:
        return match.group(1)
    # Also try 10.48550/arxiv.XXXX.XXXXX
    match = re.match(r'10\.48550/arxiv\.([\d.]+)', doi, re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def _get_pdf_cache_path(paper_id: str) -> str:
    """Get the disk cache path for a paper's PDF."""
    return os.path.join(PDF_CACHE_DIR, f"{paper_id}.pdf")


def _save_pdf_to_cache(paper_id: str, pdf_bytes: bytes) -> None:
    """Save PDF bytes to disk cache."""
    try:
        os.makedirs(PDF_CACHE_DIR, exist_ok=True)
        cache_path = _get_pdf_cache_path(paper_id)
        with open(cache_path, "wb") as f:
            f.write(pdf_bytes)
        logger.info("PDF cached to disk: %s (%d KB)", paper_id[:8], len(pdf_bytes) // 1024)
    except Exception as e:
        logger.warning("Failed to cache PDF for paper %s: %s", paper_id[:8], e)


def _load_pdf_from_cache(paper_id: str) -> bytes | None:
    """Load PDF from disk cache if available."""
    try:
        cache_path = _get_pdf_cache_path(paper_id)
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                return f.read()
    except Exception as e:
        logger.debug("PDF cache load failed for %s: %s", paper_id[:8], e)
    return None


async def download_pdf(pdf_url: str) -> bytes | None:
    """Download a PDF from URL. Returns None on failure."""
    return await _try_download(pdf_url)


async def download_papers_pdfs(papers: list, max_concurrent: int = MAX_CONCURRENT_DOWNLOADS) -> dict[str, bytes | None]:
    """Download PDFs for multiple papers concurrently using multi-source fallback.
    
    Tries: pdf_url → arXiv → Unpaywall → Sci-Hub
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def download_one(paper) -> tuple[str, bytes | None]:
        async with semaphore:
            pdf_bytes = await download_pdf_multi_source(paper)
            if pdf_bytes:
                logger.info("PDF downloaded for paper %s (%d KB)", paper.id[:8], len(pdf_bytes) // 1024)
            else:
                logger.warning("No PDF found for paper %s: %s", paper.id[:8], paper.title[:50])
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
        except Exception as e:
            logger.debug("Image rect extraction failed on page %d: %s", page_num, e)

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
    # Detect section headings: short text, larger font, or bold
    avg_font_size = sum(e.get("font_size", 10) for e in elements if e["type"] == "text") / max(1, sum(1 for e in elements if e["type"] == "text"))
    
    stream = ""
    for elem in elements:
        if elem["type"] == "text":
            text = elem["text"]
            font_size = elem.get("font_size", 10)
            is_bold = elem.get("is_bold", False)
            word_count = len(text.split())
            
            # Check if this looks like a section heading:
            # - Short text (≤ 10 words) AND (larger font OR bold)
            is_heading = word_count <= 10 and (font_size > avg_font_size * 1.1 or is_bold)
            
            if is_heading:
                # Try to classify as section heading
                section = _classify_section(text)
                if section:
                    stream += f"\n## {text.strip()}\n"  # Add as markdown heading
                else:
                    stream += text + " "
            else:
                stream += text + " "
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
    # Remove markdown heading markers
    text_lower = text_lower.lstrip("#").strip()
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
    except Exception as e:
        logger.warning("ChromaDB delete for paper %s failed: %s", paper_id[:8], e)

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

    # Build metadata filter — ChromaDB requires $and when combining multiple filters
    where = None
    if paper_ids and section_filter:
        where = {"$and": [
            {"paper_id": {"$in": paper_ids}},
            {"section": {"$in": section_filter}},
        ]}
    elif paper_ids:
        where = {"paper_id": {"$in": paper_ids}}
    elif section_filter:
        where = {"section": {"$in": section_filter}}

    # Embed the query with the SAME model used for storage (bge-base, 768 dim)
    # Do NOT use ChromaDB's built-in query_texts (which uses 384-dim MiniLM)
    model = get_embedding_model()
    query_embedding = model.encode([query], show_progress_bar=False).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
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

def _chroma_has_paper(paper_id: str) -> bool:
    """Check if ChromaDB has embeddings for a paper."""
    try:
        collection = get_chroma_collection()
        result = collection.get(where={"paper_id": paper_id})
        return bool(result["ids"])
    except Exception as e:
        logger.debug("ChromaDB has_paper check failed for %s: %s", paper_id[:8], e)
        return False


def reembed_from_sqlite(paper_id: str, title: str = "", abstract: str = "") -> bool:
    """Re-embed a paper's chunks from SQLite (without re-downloading PDF).
    
    Used when SQLite has chunks but ChromaDB is empty (e.g. after ChromaDB data loss).
    """
    from app.db.session import SessionLocal
    from app.db.repositories.paper_repo import get_chunks_by_paper

    db = SessionLocal()
    try:
        chunks = get_chunks_by_paper(db, paper_id)
        if not chunks:
            return False

        # Convert to ParsedChunk format
        from dataclasses import dataclass
        parsed = []
        for c in chunks:
            parsed.append(ParsedChunk(
                chunk_index=c.chunk_index,
                section=c.section or "unknown",
                chunk_type=c.chunk_type or "text",
                text=c.text or "",
                image_paths=[],
                page_number=c.page_number or 0,
                word_count=c.word_count or 0,
                has_pdf=c.has_pdf or False,
                extraction_method=c.extraction_method or "pymupdf_inline",
            ))

        embed_and_store_chunks(paper_id, parsed)
        ensure_paper_embedding(paper_id, title or paper_id, abstract or "", parsed)
        logger.info("Re-embedded %d chunks for paper %s from SQLite", len(parsed), paper_id[:8])
        return True
    except Exception as e:
        logger.error("Re-embed failed for paper %s: %s", paper_id[:8], e)
        return False
    finally:
        db.close()


def sync_chromadb_with_sqlite(paper_ids: list[str] = None) -> dict:
    """Sync ChromaDB with SQLite — re-embed any papers that have SQLite chunks but no ChromaDB embeddings.
    
    Args:
        paper_ids: If provided, only check these papers. If None, check all papers with SQLite chunks.
    
    Returns: {checked, reembedded, skipped, failed}
    """
    from app.db.session import SessionLocal
    from app.db.models import Paper, PaperChunk

    db = SessionLocal()
    stats = {"checked": 0, "reembedded": 0, "skipped": 0, "failed": 0}
    try:
        if paper_ids:
            papers = db.query(Paper).filter(Paper.id.in_(paper_ids)).all()
        else:
            # Get all papers that have chunks in SQLite
            paper_ids_with_chunks = db.query(PaperChunk.paper_id).distinct().all()
            paper_ids_set = {r[0] for r in paper_ids_with_chunks}
            papers = db.query(Paper).filter(Paper.id.in_(list(paper_ids_set))).all()

        # Also ensure paper_embeddings collection exists
        paper_level_col = get_paper_level_collection()

        for paper in papers:
            stats["checked"] += 1
            chroma_has_chunks = _chroma_has_paper(paper.id)
            
            # Check paper-level embedding
            try:
                existing = paper_level_col.get(ids=[paper.id])
                chroma_has_paper_emb = bool(existing["ids"])
            except Exception:
                chroma_has_paper_emb = False
            
            if chroma_has_chunks and chroma_has_paper_emb:
                stats["skipped"] += 1
            elif chroma_has_chunks and not chroma_has_paper_emb:
                # Has chunk embeddings but missing paper-level embedding
                ensure_paper_embedding(paper.id, paper.title, paper.abstract or "")
                stats["reembedded"] += 1
            elif not chroma_has_chunks:
                # Missing chunk embeddings — re-embed from SQLite
                success = reembed_from_sqlite(paper.id, paper.title, paper.abstract or "")
                if success:
                    stats["reembedded"] += 1
                else:
                    stats["failed"] += 1
            else:
                stats["skipped"] += 1

        logger.info("ChromaDB sync complete: %s", stats)
        return stats
    finally:
        db.close()


async def fetch_and_index_papers(papers: list, llm, task_id: str = "") -> dict:
    """Download PDFs, parse, and index papers for RAG.

    Three-way check per paper:
    1. SQLite has chunks + ChromaDB has embeddings → skip (fully indexed)
    2. SQLite has chunks + ChromaDB empty → re-embed from SQLite (no PDF download)
    3. Neither has data → full pipeline (download PDF, parse, embed)

    Returns summary dict with counts.
    """
    from app.db.session import SessionLocal
    from app.db.repositories.paper_repo import save_chunks, has_chunks
    from app.services.event_service import emit_event

    if not papers:
        return {"total": 0, "pdf_success": 0, "fallback": 0, "failed": 0, "skipped": 0, "reembedded": 0}

    papers_to_index = []    # Need full pipeline (download PDF + parse + embed)
    papers_to_reembed = []  # Have SQLite chunks but missing ChromaDB embeddings
    skipped = 0

    for paper in papers:
        db_check = SessionLocal()
        try:
            sqlite_has = has_chunks(db_check, paper.id)
        finally:
            db_check.close()

        chroma_has = _chroma_has_paper(paper.id)

        if sqlite_has and chroma_has:
            skipped += 1  # Fully indexed, skip
        elif sqlite_has and not chroma_has:
            papers_to_reembed.append(paper)  # Need re-embed only
        else:
            papers_to_index.append(paper)  # Need full pipeline

    if skipped > 0:
        logger.info("RAG: %d fully indexed (skip), %d need re-embed, %d need full pipeline",
                    skipped, len(papers_to_reembed), len(papers_to_index))

    # Re-embed papers that have SQLite chunks but no ChromaDB embeddings
    reembedded = 0
    for paper in papers_to_reembed:
        if reembed_from_sqlite(paper.id, paper.title, paper.abstract or ""):
            reembedded += 1

    if not papers_to_index:
        return {"total": len(papers), "pdf_success": 0, "fallback": 0, "failed": 0,
                "skipped": skipped, "reembedded": reembedded}

    total = len(papers_to_index)
    logger.info("RAG: Processing %d new papers for PDF download and indexing", total)

    if task_id:
        emit_event(task_id, "status", {"status": "indexing_pdfs", "total": total,
                                       "skipped": skipped, "reembedded": reembedded})

    # 1. Download PDFs (only for new papers)
    pdf_results = await download_papers_pdfs(papers_to_index)

    pdf_success = 0
    fallback = 0
    failed = 0

    # 2. Parse and index each paper
    for paper in papers_to_index:
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
            # Also compute and store paper-level embedding (for literature map)
            ensure_paper_embedding(paper.id, paper.title, paper.abstract or "", chunks)
        except Exception as e:
            logger.error("Embedding failed for paper %s: %s", paper.id[:8], e)

    summary = {"total": len(papers), "pdf_success": pdf_success, "fallback": fallback,
               "failed": failed, "skipped": skipped, "reembedded": reembedded}
    logger.info("RAG indexing complete: %s", summary)

    if task_id:
        emit_event(task_id, "status", {"status": "indexing_complete", **summary})

    return summary
