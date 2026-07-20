"""Step: Build paper clusters (from wiki or LLM fallback)."""

import logging

from app.agent.state import ResearchState
from app.agent.prompts import CLUSTER_SYSTEM, CLUSTER_USER
from app.db.models import Paper, TaskPaper
from app.db.repositories import paper_repo
from app.schemas.schemas import ClusterList

logger = logging.getLogger(__name__)


async def build_paper_clusters(db, state: ResearchState, llm, task_id: str):
    """Cluster ALL papers into thematic groups.

    Primary: Use pre-compiled LLM Wiki concept pages (replaces GraphRAG community detection).
    Fallback: LLM-based clustering if wiki is not yet built.
    """
    # === Primary: Get clusters from wiki ===
    try:
        from app.services.wiki_service import get_wiki_clusters
        wiki_clusters = get_wiki_clusters(db, task_id)
        if wiki_clusters and wiki_clusters.clusters:
            logger.info("Task %s: using wiki clusters (%d concept pages, %d cross-gaps)",
                       task_id[:8], len(wiki_clusters.clusters), len(wiki_clusters.cross_cluster_gaps))
            paper_repo.save_trace(db, task_id, "build_clusters", "action",
                                  output_data={"source": "wiki",
                                               "cluster_count": len(wiki_clusters.clusters),
                                               "cross_gaps": len(wiki_clusters.cross_cluster_gaps)})
            db.commit()
            return wiki_clusters
    except Exception as e:
        logger.warning("Task %s: wiki cluster retrieval failed, falling back to LLM: %s", task_id[:8], e)

    # === Fallback: LLM-based clustering ===
    from sqlalchemy.orm import joinedload
    all_tps = db.query(TaskPaper).options(
        joinedload(TaskPaper.paper)
    ).filter(
        TaskPaper.task_id == task_id,
        TaskPaper.priority.in_(["high", "medium"]),
    ).order_by(TaskPaper.final_score.desc().nullslast()).limit(80).all()

    all_papers = [(tp.paper, tp) for tp in all_tps if tp.paper]

    if len(all_papers) < 5:
        logger.info("Task %s: too few papers (%d) for clustering, skipping", task_id[:8], len(all_papers))
        return None

    papers_text = "\n".join(
        f"- [{p.id}] {p.title} ({p.year}) [{p.venue or 'N/A'}]: {(p.abstract or '')[:300]}\n  方法: {tp.summary or 'N/A'}"
        for p, tp in all_papers
    )

    messages = [
        {"role": "system", "content": CLUSTER_SYSTEM},
        {"role": "user", "content": CLUSTER_USER.format(
            topic=state.normalized_topic,
            papers=papers_text,
        )},
    ]

    try:
        cluster_list = await llm.chat_json(messages, ClusterList)
        paper_repo.save_trace(db, task_id, "build_clusters", "action",
                              output_data={"source": "llm_fallback",
                                           "cluster_count": len(cluster_list.clusters),
                                           "cross_gaps": len(cluster_list.cross_cluster_gaps)})
        db.commit()
        logger.info("Task %s: built %d clusters (LLM fallback), %d cross-cluster gaps",
                    task_id[:8], len(cluster_list.clusters), len(cluster_list.cross_cluster_gaps))
        return cluster_list
    except Exception as e:
        logger.error("Task %s: clustering failed: %s", task_id[:8], e)
        return None
