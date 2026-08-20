"""Step: Generate a Research Landscape Brief (O9).

This brief is produced regardless of whether the pipeline ultimately yields
credible research ideas. Its purpose is to guarantee the user always receives
a valuable, evidence-grounded deliverable — a map of the research landscape —
plus an honest explanation of why (if applicable) no credible idea was produced
and what targeted follow-up would help.

Design principles:
- Built ONLY from already-collected structured data (questions, coverage,
  paper roles, limitation evidence, gaps and their audit status). It never
  invents facts.
- Deterministic core: the brief is assembled from DB rows without requiring an
  LLM call, so it works even when the LLM is unavailable or the run terminated
  early. An optional LLM pass may polish the prose but its failure is non-fatal.
- Saved as a standard Report row so the existing report API/UI surfaces it.
"""

import json
import logging
from collections import Counter

from app.agent.state import ResearchState
from app.db.models import (
    ResearchContract,
    ResearchQuestion,
    CoverageRecord,
    EvidenceUnit,
    PaperRole,
    Paper,
    TaskPaper,
    GapCandidate,
    GapPhenomenonPlan,
    InterventionCandidate,
    ResearchIdea,
)
from app.db.repositories import gap_repo, paper_repo
from sqlalchemy import func

logger = logging.getLogger(__name__)

_LIMITATION_TYPES = {"limitation", "negative_result", "future_work"}


def _latest_coverage_per_question(db, task_id, question_ids):
    if not question_ids:
        return {}
    max_round_subq = db.query(
        CoverageRecord.question_id.label("q_id"),
        func.max(CoverageRecord.round_number).label("max_round"),
    ).filter(
        CoverageRecord.task_id == task_id,
        CoverageRecord.question_id.in_(question_ids),
    ).group_by(CoverageRecord.question_id).subquery()
    records = db.query(CoverageRecord).join(
        max_round_subq,
        (CoverageRecord.question_id == max_round_subq.c.q_id)
        & (CoverageRecord.round_number == max_round_subq.c.max_round)
        & (CoverageRecord.task_id == task_id),
    ).all()
    return {r.question_id: r for r in records}


def build_landscape_brief_markdown(db, task_id: str, contract_id: str | None,
                                   terminal_status: str = "",
                                   terminal_reason: str = "",
                                   state: ResearchState | None = None) -> str:
    """Assemble the landscape brief deterministically from DB rows."""
    lines: list[str] = ["# 研究态势简报 (Research Landscape Brief)\n"]

    contract = db.get(ResearchContract, contract_id) if contract_id else None
    topic = contract.topic if contract else ""
    lines.append(f"> 研究方向：{topic or '(未指定)'}\n")

    # --- 1. Terminal status & honest explanation ---
    if terminal_status:
        lines.append("## 本次运行结果\n")
        explanation = _explain_terminal(terminal_status, terminal_reason)
        lines.append(explanation + "\n")

    # --- 2. Research question tree + coverage ---
    questions = db.query(ResearchQuestion).filter(
        ResearchQuestion.task_id == task_id,
        ResearchQuestion.contract_id == contract_id,
    ).order_by(ResearchQuestion.importance.desc().nullslast()).all() if contract_id else []
    latest_cov = _latest_coverage_per_question(db, task_id, [q.id for q in questions])

    lines.append("## 研究问题与覆盖度\n")
    if not questions:
        lines.append("(尚未分解出研究问题)\n")
    else:
        lines.append("| 研究问题 | 重要性 | 状态 | 覆盖度 | 证据质量 | 检索饱和 |")
        lines.append("|---|---|---|---|---|---|")
        sat_map = {"INSUFFICIENT_OBSERVATION": "观察不足",
                   "STILL_GAINING": "仍在增长", "SATURATED": "已饱和"}
        for q in questions:
            cov = latest_cov.get(q.id)
            cov_txt = f"{cov.coverage_score:.2f}" if cov and cov.coverage_score is not None else "—"
            if cov and not cov.coverage_score and cov.unavailable_reason:
                cov_txt = f"不可得({cov.unavailable_reason[:20]})"
            eq_txt = f"{cov.evidence_quality:.2f}" if cov and cov.evidence_quality is not None else "—"
            sat_txt = sat_map.get(cov.search_saturation, "—") if cov else "—"
            imp = f"{q.importance:.2f}" if q.importance is not None else "—"
            qtext = (q.question or "")[:60].replace("|", "/")
            lines.append(f"| {qtext} | {imp} | {q.status or '—'} | {cov_txt} | {eq_txt} | {sat_txt} |")
        lines.append("")

    # --- 3. Paper role distribution ---
    roles = db.query(PaperRole.role, func.count(PaperRole.id)).filter(
        PaperRole.task_id == task_id,
    ).group_by(PaperRole.role).all()
    total_papers = db.query(func.count(TaskPaper.id)).filter(
        TaskPaper.task_id == task_id,
    ).scalar() or 0
    scored_papers = db.query(func.count(TaskPaper.id)).filter(
        TaskPaper.task_id == task_id, TaskPaper.priority.isnot(None),
    ).scalar() or 0
    high_papers = db.query(func.count(TaskPaper.id)).filter(
        TaskPaper.task_id == task_id, TaskPaper.priority == "high",
    ).scalar() or 0
    evidence_papers = db.query(func.count(func.distinct(EvidenceUnit.paper_id))).filter(
        EvidenceUnit.task_id == task_id,
    ).scalar() or 0

    lines.append("## 论文概览\n")
    lines.append(f"入库论文 {total_papers} 篇，其中完成评分 {scored_papers} 篇"
                 f"（高优先级 {high_papers} 篇），已抽取证据覆盖 {evidence_papers} 篇。\n")
    if total_papers > scored_papers:
        lines.append(f"另有 {total_papers - scored_papers} 篇为定向补检索召回、尚未评分，"
                     "未进入证据抽取池（不计入有效分析范围）。\n")
    if roles:
        lines.append("论文角色分布：" + "，".join(f"{role}: {cnt}" for role, cnt in roles) + "。\n")

    # --- 4. Known limitations (the gap fuel) ---
    limitation_evidence = db.query(EvidenceUnit).filter(
        EvidenceUnit.task_id == task_id,
        EvidenceUnit.evidence_type.in_(list(_LIMITATION_TYPES)),
        ~EvidenceUnit.verification_status.in_(["rejected", "conflicted"]),
    ).limit(30).all()

    lines.append("## 已知局限与研究空白线索\n")
    if not limitation_evidence:
        lines.append("(尚未从文献中抽取到明确的局限/负面结果信号 — 这通常意味着需要下载更多论文全文，"
                     "或针对性检索 \"limitations of ...\" / \"failure cases of ...\"。)\n")
    else:
        for ev in limitation_evidence[:15]:
            claim = (ev.normalized_claim or "")[:150].replace("\n", " ")
            lines.append(f"- [{ev.evidence_type}] {claim}")
        lines.append("")

    # --- 5. Candidate gaps and audit status ---
    # One row per canonical gap (its latest non-superseded version), so a gap
    # that was narrowed (v1 superseded -> v2) never surfaces as two unrelated
    # gaps with potentially contradictory audit verdicts.
    gaps = [g for g in gap_repo.list_canonical_gap_heads(db, task_id, contract_id)] if contract_id else []
    lines.append("## 候选研究空白 (Gap) 与审计状态\n")
    if not gaps:
        lines.append("(本次运行未挖掘出可入库的候选 Gap。)\n")
    else:
        surviving = [g for g in gaps if g.status == "surviving"]
        rejected = [g for g in gaps if g.status == "rejected"]
        inconclusive = [g for g in gaps if g.status == "inconclusive"]
        other = [g for g in gaps if g.status not in ("surviving", "rejected", "inconclusive")]
        lines.append(f"候选 Gap 共 {len(gaps)} 个：存活 {len(surviving)}，被驳回 {len(rejected)}，"
                     f"未确认 {len(inconclusive)}（检索预算耗尽后仍缺少足够证据），"
                     f"待定/审计中 {len(other)}。\n")
        # Preload phenomenon plans for surviving gaps in one query.
        plans_by_gap = {}
        if surviving:
            for plan in db.query(GapPhenomenonPlan).filter(
                GapPhenomenonPlan.gap_id.in_([g.id for g in surviving]),
            ).all():
                plans_by_gap[plan.gap_id] = plan
        for g in gaps[:12]:
            tier = "A(全文支撑)" if g.provenance_status == "complete" else "B(摘要级)"
            desc = (g.description or g.missing_capability or "")[:120].replace("\n", " ")
            vtag = f" v{g.version}" if (g.version or 1) > 1 else ""
            status_label = {"surviving": "存活", "rejected": "驳回",
                            "inconclusive": "未确认", "auditing": "审计中",
                            "audited": "已审计", "candidate": "候选",
                            "superseded": "已取代"}.get(g.status, g.status)
            lines.append(f"- [{status_label}][{tier}]{vtag} {desc}")
            # Honest inconclusive attribution: distinguish "this gap's own
            # novelty budget ran out" from "the task-global budget ran out".
            # A new canonical family must not inherit an old family's spend, so
            # per-family usage is the primary signal; the task total is context.
            if g.status == "inconclusive" and state is not None:
                fam = g.canonical_gap_id or g.id
                fam_used = int((state.gap_remediation_used or {}).get(fam, 0))
                fam_cap = settings.max_remediation_attempts_per_gap
                task_used = int((state.remediation_attempts or {}).get("__total__", 0))
                task_cap = settings.max_remediation_rounds_total
                if fam_used >= fam_cap:
                    cause = f"gap novelty 检索预算已用尽 ({fam_used}/{fam_cap})"
                else:
                    cause = f"gap novelty 预算未用尽 ({fam_used}/{fam_cap})，但任务总预算用尽"
                lines.append(f"  - 未确认归因: {cause}；task remediation {task_used}/{task_cap} 已用")
            if g.status == "surviving" and g.nearest_prior_art_title:
                conf = {"INSUFFICIENT_OBSERVATION": "观察不足",
                        "high": "高", "medium": "中", "low": "低"}.get(
                    g.search_confidence or "", "未知")
                lines.append(f"  - 最近已知 prior art: {g.nearest_prior_art_title}")
                if g.residual_gap:
                    residual = (g.residual_gap or "")[:200].replace("\n", " ")
                    lines.append(f"  - 剩余缺口: {residual}")
                stability = ""
                if g.npa_stability is not None:
                    stability += f"，近邻稳定性 {g.npa_stability:.2f}"
                if g.family_coverage is not None:
                    stability += f"，query family 覆盖 {g.family_coverage:.0%}"
                lines.append(f"  - 检索置信度: {conf}{stability}")
            plan = plans_by_gap.get(g.id)
            if plan:
                lines.append(f"  - 待验证现象: {(plan.phenomenon or '')[:140].replace(chr(10), ' ')}")
                if plan.mechanism_under_test:
                    lines.append(f"  - 被测机制: {(plan.mechanism_under_test or '')[:140].replace(chr(10), ' ')}")
                if plan.supports_gap_claim:
                    lines.append(f"  - 支持的 Gap claim: {(plan.supports_gap_claim or '')[:140].replace(chr(10), ' ')}")
                lines.append(f"  - 证伪实验: {(plan.oracle_experiment or '')[:140].replace(chr(10), ' ')}")
                if plan.kill_criterion:
                    lines.append(f"  - 放弃判据: {(plan.kill_criterion or '')[:140].replace(chr(10), ' ')}")
                if plan.kill_criterion_basis:
                    lines.append(f"  - 阈值依据: {(plan.kill_criterion_basis or '')[:140].replace(chr(10), ' ')}")
        lines.append("")

    # --- 5b. Graded candidate directions (O1) ---
    _append_graded_directions(db, lines, task_id, contract_id)

    # --- 5c. Constrained-retrieval notice (high-priority #2) ---
    from app.config import settings
    if settings.constrained_retrieval_mode:
        lines.append("## 检索供给提示\n")
        lines.append(
            "当前未配置 Semantic Scholar API Key，也未提供 OpenAlex/Crossref 邮箱，"
            "高质量检索源（含引用关系与近邻对比）受限，可能降低研究缺口通过审计的比例。"
            "缺口审计的准入门已自动放宽以适配受限模式。若产出偏少，可在 .env 填入 "
            "OPENALEX_EMAIL / CROSSREF_EMAIL（无需审批、即时生效）或申请 Semantic Scholar Key 以改善。\n"
        )

    # --- 6. Recommended next steps ---
    lines.append("## 建议的下一步\n")
    for step in _recommend_next_steps(db, task_id, questions, latest_cov,
                                      limitation_evidence, gaps, terminal_status):
        lines.append(f"- {step}")
    lines.append("")

    return "\n".join(lines)


def _append_graded_directions(db, lines, task_id, contract_id):
    """O1: surface graded (A/B/C) intervention/idea directions in the brief.

    Even when the pipeline did not reach the experiment stage, any interventions
    that were generated are shown with their confidence tier so the user gets a
    ranked list of directions instead of nothing.
    """
    if not contract_id:
        return
    interventions = db.query(InterventionCandidate).filter(
        InterventionCandidate.task_id == task_id,
        InterventionCandidate.contract_id == contract_id,
    ).all()
    ideas = db.query(ResearchIdea).filter(
        ResearchIdea.task_id == task_id,
        ResearchIdea.contract_id == contract_id,
        ResearchIdea.idea_status == "active",
    ).all()
    if not interventions and not ideas:
        return

    lines.append("## 分级候选方向 (Confidence Tier)\n")
    lines.append("A = 证据充分、通过全部闸门；B = 证据部分支撑或部分闸门待确认，可行但需人工/实验确认；"
                 "C = 推测性方向，某闸门未通过，仅供参考。\n")

    tier_order = {"A": 0, "B": 1, "C": 2}

    def _tier_label(tier: str) -> str:
        return {"A": "A(可信)", "B": "B(待确认)", "C": "C(推测)"}.get(tier or "C", "C(推测)")

    if ideas:
        lines.append("研究想法：")
        for idea in sorted(ideas, key=lambda x: tier_order.get(x.confidence_tier or "C", 2)):
            title = (idea.title or "")[:80].replace("\n", " ")
            lines.append(f"- [{_tier_label(idea.confidence_tier)}] {title}")
        lines.append("")

    if interventions:
        lines.append("干预方案：")
        for itv in sorted(interventions, key=lambda x: tier_order.get(x.confidence_tier or "C", 2))[:12]:
            desc = (itv.proposed_intervention or "")[:100].replace("\n", " ")
            gate_summary = f"E:{itv.evidence_gate}/N:{itv.novelty_gate}/F:{itv.feasibility_gate}"
            lines.append(f"- [{_tier_label(itv.confidence_tier)}][{itv.status}] {desc} （闸门 {gate_summary}）")
        lines.append("")


def _explain_terminal(status: str, reason: str) -> str:
    mapping = {
        "more_research_required": (
            "本次运行判定证据尚不足以形成可信的研究空白，需要补充检索。"
            "这是一个诚实的中间结果，而非失败 — 下方给出了具体的补充方向。"
        ),
        "insufficient_evidence": (
            "在现有证据下未能产出达到可信标准的研究想法。系统坚持不编造 Idea，"
            "因此以 insufficient_evidence 结束，并保留了完整的领域地图供人工判断。"
        ),
        "abstained": (
            "通过了 Gap 审计但未能生成合格的最小实验方案，系统选择弃权（abstain）而非硬凑。"
        ),
        "failed": (
            "控制面数据不完整导致本次运行失败。通常是检索限流或 PDF 全文抽取失败所致，"
            "建议配置 Semantic Scholar API key 后重试。"
        ),
        "waiting_for_user_review": (
            "已产出证据支撑的研究方向，等待用户查看与选择。"
        ),
    }
    base = mapping.get(status, f"运行状态：{status}。")
    if reason:
        base += f"（原因：{reason}）"
    return base


def _recommend_next_steps(db, task_id, questions, latest_cov,
                          limitation_evidence, gaps, terminal_status) -> list[str]:
    steps: list[str] = []

    # Low-coverage high-importance questions
    low_cov_high_imp = [
        q for q in questions
        if (q.importance or 0) >= 0.7
        and (q.id not in latest_cov or (latest_cov[q.id].coverage_score or 0) < 0.3)
    ]
    if low_cov_high_imp:
        names = "; ".join((q.question or "")[:40] for q in low_cov_high_imp[:3])
        steps.append(f"针对覆盖不足的高重要性问题做定向检索：{names}")

    if not limitation_evidence:
        steps.append("检索并下载核心论文全文，或用 \"limitations of <方法>\"、\"failure cases\"、"
                     "\"threats to validity\" 等 query 补充局限性证据（Gap 挖掘依赖这些信号）。")

    b_tier_gaps = [g for g in gaps if g.provenance_status != "complete"]
    if b_tier_gaps:
        steps.append(f"有 {len(b_tier_gaps)} 个 B 档（仅摘要级证据）候选 Gap，"
                     "建议补充这些方向论文的全文以升级为 A 档并通过审计。")

    if terminal_status in ("more_research_required", "insufficient_evidence"):
        steps.append("配置 Semantic Scholar API key（免费）以消除 429 限流，提升经典论文与近邻审计的覆盖。")

    if not steps:
        steps.append("当前领域覆盖较为充分，可进入用户选择与实验方案环节。")
    return steps


async def generate_landscape_brief(db, state: ResearchState, task_id: str,
                                   terminal_status: str = "",
                                   terminal_reason: str = "") -> str:
    """Generate and persist the Research Landscape Brief.

    Idempotent-friendly: always writes a fresh Report row (history preserved).
    Deterministic; never raises on missing data — returns whatever can be built.
    """
    try:
        markdown = build_landscape_brief_markdown(
            db, task_id, state.contract_id, terminal_status, terminal_reason, state=state
        )
        content_json = json.dumps({
            "type": "landscape_brief",
            "terminal_status": terminal_status,
            "terminal_reason": terminal_reason,
        }, ensure_ascii=False)
        paper_repo.save_report(db, task_id, markdown, content_json)
        paper_repo.save_trace(db, task_id, "generate_landscape_brief", "action",
                              output_data={"terminal_status": terminal_status,
                                           "chars": len(markdown)})
        db.commit()
        logger.info("Task %s: landscape brief generated (%d chars, status=%s)",
                    task_id[:8], len(markdown), terminal_status)
        return markdown
    except Exception as e:
        logger.warning("Task %s: landscape brief generation failed (non-fatal): %s",
                       task_id[:8], e)
        db.rollback()
        return ""
