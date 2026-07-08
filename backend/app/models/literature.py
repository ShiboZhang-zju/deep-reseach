"""Literature graph data models.

Defines in-memory representations of papers, chunks, and their relationships,
with clear traceability and typed relations for the literature map.

Relationship types:
  Paper → Paper:  cites / co_cited / biblio_coupled / semantic_similar
  Chunk → Paper:  belongs_to (which paper this chunk is from)
  Chunk → Chunk:  sequential (same paper, next chunk) / cross_reference (cites another paper's chunk)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# === Relation type enums ===

class PaperRelationType(str, Enum):
    """Paper-to-paper relationship types."""
    CITES = "cites"                    # A directly cites B (from S2/OpenAlex API)
    CO_CITED = "co_cited"              # A and B are co-cited by some paper C
    BIBLIO_COUPLED = "biblio_coupled"  # A and B share common references
    SEMANTIC_SIMILAR = "semantic_similar"  # A and B have high embedding similarity


class ChunkRelationType(str, Enum):
    """Chunk-to-chunk relationship types."""
    SEQUENTIAL = "sequential"          # Next chunk in same paper
    SAME_SECTION = "same_section"      # Chunks from same section of same paper
    CROSS_REFERENCE = "cross_reference"  # Chunk in Paper A references content in Paper B
    SEMANTIC_SIMILAR = "semantic_similar"  # Chunks with high embedding similarity


class ChunkType(str, Enum):
    """Types of paper chunks."""
    TEXT = "text"
    FIGURE = "figure"
    TABLE = "table"
    FORMULA = "formula"


class PaperPriority(str, Enum):
    """Paper priority levels within a task."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    CITATION_ONLY = "citation_only"    # Paper found via citation expansion, not directly searched


# === Core data classes ===

@dataclass
class PaperNode:
    """In-memory representation of a paper node in the literature graph.
    
    Wraps the SQLAlchemy Paper model with computed relationships.
    """
    id: str
    title: str
    abstract: str = ""
    year: Optional[int] = None
    venue: Optional[str] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    semantic_scholar_id: Optional[str] = None
    openalex_id: Optional[str] = None
    url: Optional[str] = None
    pdf_url: Optional[str] = None
    citation_count: int = 0
    authors: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)  # which APIs found this paper
    
    # Task-specific (filled when queried within a task context)
    priority: Optional[str] = None
    final_score: Optional[float] = None
    relevance_score: Optional[float] = None
    method_extract: Optional[str] = None  # from TaskPaper.summary
    is_seed: bool = True  # True if directly searched, False if citation expansion
    
    # Computed relationships (lazy-loaded)
    chunk_ids: list[str] = field(default_factory=list)
    has_pdf: bool = False
    has_chunks: bool = False
    
    # Citation relationships
    cites: list[str] = field(default_factory=list)              # paper IDs this paper cites
    cited_by: list[str] = field(default_factory=list)           # paper IDs that cite this paper
    co_cited_with: list[tuple[str, float]] = field(default_factory=list)  # (paper_id, score)
    biblio_coupled_with: list[tuple[str, float]] = field(default_factory=list)  # (paper_id, score)
    semantic_similar: list[tuple[str, float]] = field(default_factory=list)  # (paper_id, similarity)
    
    @property
    def display_name(self) -> str:
        """Short display name for UI."""
        year_str = f" ({self.year})" if self.year else ""
        return f"{self.title[:60]}{year_str}"
    
    @property
    def has_citation_data(self) -> bool:
        """Whether this paper has citation relationships loaded."""
        return bool(self.cites or self.cited_by or self.co_cited_with or self.biblio_coupled_with)
    
    @classmethod
    def from_orm(cls, paper_orm, task_paper_orm=None) -> "PaperNode":
        """Create from SQLAlchemy ORM objects."""
        import json
        authors = []
        if paper_orm.authors_json:
            try:
                authors = json.loads(paper_orm.authors_json)
            except Exception:
                pass
        sources = []
        if paper_orm.sources_json:
            try:
                sources = json.loads(paper_orm.sources_json)
            except Exception:
                pass
        
        node = cls(
            id=paper_orm.id,
            title=paper_orm.title,
            abstract=paper_orm.abstract or "",
            year=paper_orm.year,
            venue=paper_orm.venue,
            doi=paper_orm.doi,
            arxiv_id=paper_orm.arxiv_id,
            semantic_scholar_id=paper_orm.semantic_scholar_id,
            openalex_id=paper_orm.openalex_id,
            url=paper_orm.url,
            pdf_url=paper_orm.pdf_url,
            citation_count=paper_orm.citation_count or 0,
            authors=authors,
            sources=sources,
        )
        
        if task_paper_orm:
            node.priority = task_paper_orm.priority
            node.final_score = task_paper_orm.final_score
            node.relevance_score = task_paper_orm.relevance_score
            node.method_extract = task_paper_orm.summary
        
        return node


@dataclass
class ChunkNode:
    """In-memory representation of a paper chunk with full traceability.
    
    Every chunk knows exactly which paper it belongs to, what section,
    and can be linked to other chunks.
    """
    id: str
    paper_id: str               # which paper this chunk is from
    chunk_index: int            # position within the paper
    section: str = "unknown"    # method / experiment / introduction / conclusion / abstract
    chunk_type: str = "text"    # text / figure / table / formula
    text: str = ""
    image_paths: list[str] = field(default_factory=list)
    page_number: int = 0
    word_count: int = 0
    has_pdf: bool = False
    extraction_method: str = "pymupdf_inline"
    
    # ChromaDB metadata
    chroma_id: str = ""         # ID in ChromaDB (usually "{paper_id}_chunk_{index}")
    embedding: Optional[list[float]] = None  # cached embedding vector
    
    # Computed relationships
    cross_references: list[str] = field(default_factory=list)  # chunk IDs in other papers
    semantic_similar_chunks: list[tuple[str, float]] = field(default_factory=list)  # (chunk_id, score)
    
    @property
    def display_text(self) -> str:
        """Clean text for display (strip inline markers)."""
        import re
        return re.sub(r'\[FIGURE:.*?\]|\[TABLE\]|\[/TABLE\]', '', self.text).strip()
    
    @property
    def is_method_section(self) -> bool:
        return self.section in ("method", "methodology", "approach", "model", "architecture")
    
    @property
    def is_experiment_section(self) -> bool:
        return self.section in ("experiment", "evaluation", "results", "setup")
    
    @classmethod
    def from_orm(cls, chunk_orm) -> "ChunkNode":
        """Create from SQLAlchemy PaperChunk ORM."""
        import json
        image_paths = []
        if chunk_orm.image_paths_json:
            try:
                image_paths = json.loads(chunk_orm.image_paths_json)
            except Exception:
                pass
        
        return cls(
            id=chunk_orm.id,
            paper_id=chunk_orm.paper_id,
            chunk_index=chunk_orm.chunk_index,
            section=chunk_orm.section or "unknown",
            chunk_type=chunk_orm.chunk_type or "text",
            text=chunk_orm.text or "",
            image_paths=image_paths,
            page_number=chunk_orm.page_number or 0,
            word_count=chunk_orm.word_count or 0,
            has_pdf=chunk_orm.has_pdf or False,
            extraction_method=chunk_orm.extraction_method or "pymupdf_inline",
            chroma_id=f"{chunk_orm.paper_id}_chunk_{chunk_orm.chunk_index}",
        )
    
    @classmethod
    def from_chroma_result(cls, chroma_id: str, text: str, metadata: dict, score: float = 0.0) -> "ChunkNode":
        """Create from a ChromaDB query result.
        
        This ensures every retrieved chunk has full paper traceability.
        """
        paper_id = metadata.get("paper_id", "")
        chunk_index = metadata.get("chunk_index", 0)
        return cls(
            id=f"{paper_id}_chunk_{chunk_index}",
            paper_id=paper_id,
            chunk_index=chunk_index,
            section=metadata.get("section", "unknown"),
            text=text,
            page_number=metadata.get("page_number", 0),
            chroma_id=chroma_id,
        )


@dataclass
class CitationEdge:
    """A directed relationship between two papers."""
    source_paper_id: str
    target_paper_id: str
    relation_type: str   # PaperRelationType
    weight: float = 1.0
    source_task_id: Optional[str] = None
    
    @property
    def is_directed(self) -> bool:
        """cites is directed; others are undirected."""
        return self.relation_type == PaperRelationType.CITES.value
    
    def to_dict(self) -> dict:
        return {
            "source": self.source_paper_id,
            "target": self.target_paper_id,
            "type": self.relation_type,
            "weight": self.weight,
        }


@dataclass
class LiteratureGraph:
    """Complete literature graph for a task: nodes (papers) + edges (citations) + clusters.
    
    This is the in-memory representation of the literature map.
    """
    task_id: str
    nodes: dict[str, PaperNode] = field(default_factory=dict)  # paper_id → PaperNode
    edges: list[CitationEdge] = field(default_factory=list)
    chunks: dict[str, list[ChunkNode]] = field(default_factory=dict)  # paper_id → chunks
    
    # Clustering results
    clusters: dict[int, list[str]] = field(default_factory=dict)  # cluster_id → [paper_ids]
    cluster_names: dict[int, str] = field(default_factory=dict)   # cluster_id → name
    
    @property
    def node_count(self) -> int:
        return len(self.nodes)
    
    @property
    def edge_count(self) -> int:
        return len(self.edges)
    
    @property
    def seed_papers(self) -> list[PaperNode]:
        """Papers directly searched (not citation expansion)."""
        return [n for n in self.nodes.values() if n.is_seed]
    
    @property
    def citation_papers(self) -> list[PaperNode]:
        """Papers found via citation expansion."""
        return [n for n in self.nodes.values() if not n.is_seed]
    
    def get_neighbors(self, paper_id: str) -> list[str]:
        """Get all paper IDs connected to this paper (any relation type)."""
        neighbors = set()
        for edge in self.edges:
            if edge.source_paper_id == paper_id:
                neighbors.add(edge.target_paper_id)
            elif edge.target_paper_id == paper_id:
                neighbors.add(edge.source_paper_id)
        return list(neighbors)
    
    def get_papers_by_cluster(self, cluster_id: int) -> list[PaperNode]:
        """Get all papers in a cluster."""
        paper_ids = self.clusters.get(cluster_id, [])
        return [self.nodes[pid] for pid in paper_ids if pid in self.nodes]
    
    def to_dict(self) -> dict:
        """Serialize for API response / frontend visualization."""
        return {
            "task_id": self.task_id,
            "nodes": [
                {
                    "id": n.id,
                    "title": n.title,
                    "year": n.year,
                    "venue": n.venue,
                    "priority": n.priority,
                    "final_score": n.final_score,
                    "citation_count": n.citation_count,
                    "is_seed": n.is_seed,
                    "has_chunks": n.has_chunks,
                    "cluster_id": next(
                        (cid for cid, pids in self.clusters.items() if n.id in pids), None
                    ),
                }
                for n in self.nodes.values()
            ],
            "edges": [e.to_dict() for e in self.edges],
            "clusters": [
                {
                    "id": cid,
                    "name": self.cluster_names.get(cid, f"Cluster {cid}"),
                    "paper_count": len(pids),
                    "paper_ids": pids,
                }
                for cid, pids in self.clusters.items()
            ],
            "stats": {
                "total_papers": self.node_count,
                "seed_papers": len(self.seed_papers),
                "citation_papers": len(self.citation_papers),
                "total_edges": self.edge_count,
                "cluster_count": len(self.clusters),
            },
        }
