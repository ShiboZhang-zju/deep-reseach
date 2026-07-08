"""LLM prompt templates."""

CLARIFY_SYSTEM = """You are a research advisor. Analyze the user's research direction input and determine if it's specific enough to start a literature search.

If the direction is clear, provide:
- normalized_topic: a concise academic topic description (in English, for search purposes)
- keywords: 5-10 relevant search keywords (in English, for search purposes)

If unclear, generate 2-3 clarifying questions IN CHINESE."""

CLARIFY_USER = "Research direction: {user_input}"


QUERIES_SYSTEM = """You are a research search strategist. Based on the research topic, keywords, and previous queries used, generate {num_queries} new search queries for academic paper search.

Guidelines:
- Each query should be a concise phrase (3-8 words) suitable for academic search engines
- Vary the queries: some specific, some broader
- Include alternative terminology and synonyms
- Avoid repeating previously used queries
- Focus on finding high-quality, relevant papers"""

QUERIES_USER = """Research topic: {topic}
Keywords: {keywords}
Previous queries used:
{used_queries}

Knowledge gaps to address:
{gaps}

User feedback: {feedback}

Generate {num_queries} search queries."""


SCORE_SYSTEM = """You are an expert paper reviewer. Score the following paper based on its relevance and quality for the given research topic.

Score each dimension from 0.0 to 1.0:
- relevance: How directly relevant is this paper to the research topic?
- authority: Quality of venue, citation count, author reputation
- recency: How recent is the publication? (1.0 for current year, lower for older)
- novelty: Does it introduce new methods, tasks, or benchmarks?
- idea_potential: Could this paper inspire new research ideas?

Also provide:
- summary: A one-sentence summary of the paper (IN CHINESE)
- method_extract: Extract the SPECIFIC technical details — what model/architecture/algorithm/dataset does this paper use? Be concrete (e.g., "使用Transformer-XL作为基础架构，在attention层增加memory gate，在MultiWOZ数据集上训练" not "使用深度学习方法"). (IN CHINESE)
- reason: A brief reason for the scores (IN CHINESE)"""

SCORE_USER = """Research topic: {topic}

Paper title: {title}
Abstract: {abstract}
Authors: {authors}
Year: {year}
Venue: {venue}
Citations: {citations}"""


ROUND_SUMMARY_SYSTEM = """You are a research summarizer. Based on the papers found in this round, provide:
1. A concise summary of key findings and trends
2. A list of knowledge gaps that still need to be addressed (areas where more papers are needed)

Write IN CHINESE."""

ROUND_SUMMARY_USER = """Research topic: {topic}
Round number: {round_num}

Papers found this round:
{papers_summary}

Previous knowledge gaps:
{previous_gaps}"""


REPORT_SYSTEM = """You are a senior research analyst. Write a comprehensive research report in Markdown format based on all collected papers and round summaries.

The report should include:
1. **Overview** - Research topic and scope
2. **Key Trends** - Major themes and developments
3. **Important Papers** - Highlights of high-priority papers with brief analysis
4. **Methods & Approaches** - Common methodologies in the field
5. **Open Challenges** - Unresolved problems and gaps
6. **Future Directions** - Emerging areas worth exploring
7. **References** - Full reference list with DOIs

Write in clear, academic Chinese. Use proper Markdown formatting.

IMPORTANT: 
- When referencing a paper, use the [P1], [P2] etc. numbering from the provided paper list.
- Only reference papers from the provided list. Do not invent or reference any paper not in the list.
- If you mention a method or finding not covered by the listed papers, clearly state it as "general knowledge" without attributing it to a specific paper.
- Include a References section at the end listing all cited papers with their [Px] number, title, year, and DOI."""

REPORT_USER = """Research topic: {topic}
Keywords: {keywords}

Round summaries:
{round_summaries}

High-priority papers:
{high_papers}

Knowledge gaps:
{gaps}"""


IDEAS_SYSTEM = """You are a creative research idea generator. Based on the research report, paper clusters, and collected papers, generate {num_ideas} novel research ideas.

For each idea, provide:
- title: A concise title (IN CHINESE)
- description: Brief description of the idea (IN CHINESE). Must start with "基于[聚类X]的[具体方法]..." to show which cluster's method this idea builds upon.
- motivation: Why is this idea important? What SPECIFIC gap does it fill? Reference specific cluster limitations. (IN CHINESE)
- method_sketch: A DETAILED technical method description (IN CHINESE). MUST include ALL of:
  * 具体模型架构: e.g., "基于Llama-3-8B，在attention层增加dual memory gate" (NOT "使用大语言模型")
  * 具体算法: e.g., "top-k sparse retrieval + relevance scoring + decay-based forgetting" (NOT "优化检索策略")
  * 具体数据集: e.g., "MultiWOZ + bAbI + 自定义多轮对话数据" (NOT "对话数据集")
  * 具体评估指标: e.g., "BLEU-4 + task completion rate + memory retrieval latency" (NOT "准确性")
  * 具体基线: e.g., "MemGPT, A-Mem, 标准LLM无记忆" (NOT "现有方法")
- expected_contribution: What would this contribute to the field? (IN CHINESE)
- related_paper_ids: List of paper IDs from the provided paper list that are most relevant
- related_paper_titles: List of corresponding paper titles (keep original paper titles)

CRITICAL RULES:
1. Do NOT use vague concepts like "动态优化" "智能管理" "生物启发" without explaining the EXACT mechanism
2. Do NOT invent datasets, models, or concepts that don't exist. Only use well-known, real datasets and models.
3. Each idea MUST be grounded in specific cluster methods — do not generate generic ideas
4. method_sketch must be specific enough that a grad student could implement it
5. Baselines must be REAL methods from the papers or well-known methods (e.g., "MemGPT", "RAG", "fine-tuned LLM")

Write all text fields (title, description, motivation, method_sketch, expected_contribution) IN CHINESE."""

IDEAS_USER = """Research topic: {topic}

Report:
{report}

High-priority papers (use paper_id for related_paper_ids):
{papers}

Knowledge gaps:
{gaps}"""


IDEA_SCORE_SYSTEM = """You are a research idea evaluator. Score the following research idea on multiple dimensions (0.0 to 1.0):

- novelty: How novel is this idea compared to existing work?
- feasibility: How feasible is it to implement?
- significance: What is the academic research significance?
- evidence_support: How well is it supported by existing literature?
- differentiation: How different is it from existing approaches?
- experimentability: How easy is it to design experiments to validate?
- potential_impact: What is the potential impact (engineering, industry, open source)?
- risk: What is the risk level (higher = more risky)?

Scoring calibration:
- 0.9-1.0: Exceptional — groundbreaking idea with clear innovation and specific technical plan
- 0.7-0.8: Strong — promising direction with solid foundation AND concrete method (specific model, algorithm, dataset, metrics)
- 0.5-0.6: Moderate — interesting direction but method is vague or lacks technical specifics
- 0.3-0.4: Weak — incremental, poorly grounded, or method_sketch uses vague terms like "动态优化" without specifics
- 0.0-0.2: Poor — already exists, infeasible, or invented concepts/datasets

PENALIZE: Ideas with vague method_sketch (e.g., "优化检索策略" without specific algorithm) should score ≤0.5.
PENALIZE: Ideas that invent non-existent datasets or concepts should score ≤0.3.
REWARD: Ideas with specific, implementable technical plans should score ≥0.7.

Note: Research ideas inherently carry risk. Do not over-penalize risk — a novel idea with moderate risk can still score 0.7+.

Also provide a brief reason for the scores. Write the reason IN CHINESE."""

IDEA_SCORE_USER = """Research topic: {topic}

Idea title: {title}
Idea description: {description}
Idea motivation: {motivation}
Idea method: {method}
Expected contribution: {contribution}

Related papers:
{related_papers}"""


EXPERIMENT_SYSTEM = """You are a research experiment designer. Based on the research idea, design a detailed experiment plan.

Include:
- hypothesis: The hypothesis to test
- dataset: Recommended datasets
- baselines: Baseline methods to compare against
- metrics: Evaluation metrics
- steps: Step-by-step experiment procedure
- risks: Potential risks and mitigations

Write in clear, academic Chinese."""

EXPERIMENT_USER = """Research topic: {topic}

Idea title: {title}
Idea description: {description}
Idea method: {method}
Expected contribution: {contribution}

Related papers:
{related_papers}"""


# === P0-A: Report self-feedback ===

REPORT_FEEDBACK_SYSTEM = """You are a research report reviewer. Evaluate the report and identify areas for improvement.

Check:
1. Completeness — Are there important topics or papers missing?
2. Organization — Is the structure logical and clear?
3. Evidence — Are claims supported by cited papers?
4. Depth — Does the report go beyond surface-level summaries?

Provide specific, actionable suggestions. Write IN CHINESE."""

REPORT_FEEDBACK_USER = """Research topic: {topic}

Report:
{report}

High-priority papers available:
{papers}

Evaluate the report and provide improvement suggestions."""

REPORT_REFINE_SYSTEM = """You are a research report editor. Improve the report based on the feedback.

Requirements:
- Keep the existing structure and citations [P1], [P2] etc.
- Address each feedback point
- Add missing content using the provided papers
- Do not remove existing content unless it is factually wrong
- Write in clear, academic Chinese."""

REPORT_REFINE_USER = """Original report:
{report}

Feedback:
{feedback}

High-priority papers:
{papers}

Improve the report based on the feedback."""


# === P0-B: Idea novelty check ===

NOVELTY_CHECK_SYSTEM = """You are a research novelty evaluator. Determine whether the proposed research idea is novel or if similar work already exists.

Consider:
1. Has this exact idea been proposed in the provided existing papers?
2. Is the core method or approach already published?
3. What makes this idea different from existing work?

Be strict — if there is substantial overlap with existing work, mark as not novel.
Write the novelty_reason IN CHINESE."""

NOVELTY_CHECK_USER = """Research idea:
标题: {title}
描述: {description}
方法: {method}

Existing papers found by searching:
{existing_papers}

Is this idea novel? What are the key differences from existing work?"""


# === P0-C: Idea self-feedback ===

IDEA_FEEDBACK_SYSTEM = """You are a research idea quality evaluator. Review the generated ideas and identify which ones need improvement.

For each idea, check:
1. Is it truly novel (not just a minor variation of existing work)?
2. Is the method sketch specific enough to be actionable?
3. Is it well-grounded in the provided papers?
4. Does it address a real gap?

Identify the weakest ideas and suggest how to improve them. Write IN CHINESE."""

IDEA_FEEDBACK_USER = """Research topic: {topic}

Generated ideas:
{ideas}

High-priority papers:
{papers}

Knowledge gaps:
{gaps}

Which ideas are weak and how should they be improved?"""


# === Paper clustering (reference Idea2Paper) ===

CLUSTER_SYSTEM = """You are a research literature analyst. Group the provided papers into 3-6 thematic clusters based on their methods, problems, and approaches.

For each cluster, extract:
- cluster_name: A concise name for this research direction (IN CHINESE)
- core_method: The main method/technique category (IN CHINESE, e.g., "基于Transformer的记忆增强" not "深度学习")
- technique_details: SPECIFIC technical details — what exact models, algorithms, datasets, architectures are used? Be concrete (e.g., "MemGPT使用LLM管理分层记忆池，通过summarization压缩旧记忆；A-Mem用agentic memory实现自主记忆管理，基于Llama-3训练"). (IN CHINESE)
- problem_addressed: What problem does this cluster address? (IN CHINESE)
- key_findings: Key findings or conclusions (IN CHINESE)
- limitations: What limitations or unsolved problems remain? (IN CHINESE)
- representative_papers: List of paper titles that best represent this cluster

Also identify cross_cluster_gaps: research opportunities that COMBINE specific methods from different clusters. Be concrete (e.g., "聚类1的MemGPT分层记忆 + 聚类2的类比检索 → 可探索分层类比记忆架构" not "可以结合不同方法").

Write all text fields IN CHINESE."""

CLUSTER_USER = """Research topic: {topic}

Papers (all priorities, not just high):
{papers}

Group these papers into thematic clusters and extract structured information."""


# === RAG: Method extraction from full-text passages ===

METHOD_EXTRACT_SYSTEM = """You are a technical paper analyst. Extract the SPECIFIC methods, models, algorithms, and datasets from the given paper passages.

Focus on:
- Model architecture (e.g., "Transformer-XL", "BERT-base", "Llama-3-8B")
- Algorithms/techniques (e.g., "contrastive learning", "RLHF", "top-k retrieval")
- Datasets (e.g., "MultiWOZ", "MMLU", "GLUE")
- Evaluation metrics (e.g., "F1", "BLEU-4", "accuracy")
- Baselines compared against

Be CONCRETE. Do not use vague terms like "deep learning" or "neural network".
Output in Chinese. Format: a concise paragraph listing the key technical details."""

METHOD_EXTRACT_USER = """Paper title: {title}

Relevant passages from full text:
{passages}

Extract the specific technical details (models, algorithms, datasets, metrics, baselines)."""


# === RAG: VLM figure description ===

FIGURE_VLM_PROMPT = """Describe this figure from an academic paper concisely.
Focus on:
1. Figure type (architecture diagram / results plot / flowchart / example / comparison)
2. Key components, methods, or data shown
3. Main takeaways
Keep under 100 words. Output in Chinese."""
