# v7 Replay Pilot — Blind Prior-Art Review

30 v7-only opportunities (claimed discoveries with no v6 counterpart).

For EACH block below, judge whether existing prior art already covers the
claimed delta. You are given the claim, its supporting evidence, and a
recall-oriented candidate prior-art list retrieved during the audit.

Verdict semantics (fill human_verdicts.jsonl, keyed by review_id):
- FULL    — a candidate (or other work you know) already covers the claimed
            delta completely  -> false-open gap
- PARTIAL — prior art covers part of the delta; a narrower residual remains
- NONE    — no coverage you can identify  -> true gap

Candidate-list disclosure: recall is prioritized over precision. Candidate
papers historically share zero literal tokens with the claim ~39% of the
time and are NOT pre-filtered — a weak-looking candidate list is noise,
not evidence of novelty. You may (and should) bring in any prior art you
know yourself; the candidates are a starting point, not a boundary.

The blocks are anonymized and shuffled. Do not open review_mapping.json
until scoring.

---

## v7r-001  [mechanism_gap]

Topic: General-purpose uncertainty quantification for deep learning

Claimed delta: Provides a mechanistic explanation for TTEM failure, enabling the design of modified objectives that prevent overconfidence on OOD data without sacrificing in-distribution performance.

Supporting evidence (as extracted from the literature):
- [negative_result/abstract] 在分布外数据上，覆盖率无法得到维持；随着数据集偏移增加，当前不确定性量化技术的缺陷显现。
- [negative_result/introduction] 深度学习模型在图像分类实验中经常对特定实例失败，尽管对其预测非常自信。 (2021)

Candidate prior art (audit-recalled, unfiltered):
- Beyond Accuracy: Assessing Calibration of Geospatial Foundation Models and Their Sensitivity to Distribution Shifts (2026) — arXiv (Cornell University)
    overlaps: problem=partial mechanism=no claim=partial
- Backbone agnostic Pareto evidential networks for trustworthy fault diagnosis and out of distribution detection (2026) — Scientific Reports
    overlaps: problem=partial mechanism=no claim=no
- Tuning Out-of-Distribution (OOD) Detectors Without Given OOD Data (2026) — arXiv
    overlaps: problem=partial mechanism=no claim=no
- The Peril of Popular Deep Learning Uncertainty Estimation Methods (2021) — arXiv
    overlaps: problem=no mechanism=no claim=no
- Uncertainty Quantification in Machine Learning for Polymer Properties: A Benchmark Study (2025) — ChemRxiv
    overlaps: problem=no mechanism=no claim=no

## v7r-002  [contradiction]

Topic: Test-time training for adaptive inference in language models

Claimed delta: Identifies the specific boundary condition (e.g., number of reasoning steps > 3 vs <= 1) where self-correction efficacy inverts, resolving the conflict between positive scaling results and negative truncation results.

Supporting evidence (as extracted from the literature):
- [negative_result/method] 截断所有步骤（即元梯度的二阶近似）导致性能大幅下降，即使在简单的QA1-1K设置中也是如此。 (2025)
- [result/abstract] 通过扩展睡眠时计算，在 Stateful GSM-Symbolic 上准确率最高提升13%，在 Stateful AIME 上最高提升18%。 (2025)

Candidate prior art (audit-recalled, unfiltered):
- Sleep-time Compute: Beyond Inference Scaling at Test-time (2025) — arXiv
    overlaps: problem=partial mechanism=partial claim=partial
- PERK: Long-Context Reasoning as Test-Time Learning (2025) — arXiv
    overlaps: problem=partial mechanism=partial claim=partial
- Large Language Models are Zero-Shot Reasoners (2022) — arXiv (Cornell University)
    overlaps: problem=partial mechanism=no claim=no
- LiteSearch: Efficient Tree Search with Dynamic Exploration Budget for Math Reasoning (2025) — Proceedings of the AAAI Conference on Artificial Intelligence
    overlaps: problem=no mechanism=no claim=no
- Learning to Discover at Test Time (2026) — arXiv
    overlaps: problem=no mechanism=no claim=no

## v7r-003  [mechanism_gap]

Topic: Mechanistic interpretability of mixture-of-experts routing decisions

Claimed delta: Moves beyond correlational clustering metrics (NMI > 0.8) to establish causal necessity/sufficiency of specific model components for expert selection, resolving the ambiguity between stable specialization and collapse.

Supporting evidence (as extracted from the literature):
- [result/conclusion] 路由签名按任务类别强烈聚类。 (2026)
- [comparison/experiment] 分类器仅看到路由模式，未看到token身份或输出文本，表明任务信息可从路由签名中线性获取。 (2026)
- [limitation/introduction] 相互激活MoE架构特别容易受到崩溃影响，因为相互强化会产生正反馈回路，迅速使专家行为同质化。 (2026)
- [limitation/conclusion] 分析是相关性的而非因果性的；测量了路由结构但未对专家进行干预。 (2026)

Candidate prior art (audit-recalled, unfiltered):
- Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability (2023) — arXiv (Cornell University)
    overlaps: problem=partial mechanism=yes claim=partial
- Separating Tongue from Thought: Activation Patching Reveals Language-Agnostic Concept Representations in Transformers (2024) — arXiv
    overlaps: problem=partial mechanism=yes claim=partial
- OmniGene-4: A Unified Bio-Language MoE Model with Router-Level Interpretability (2026) — bioRxiv (Cold Spring Harbor Laboratory)
    overlaps: problem=partial mechanism=no claim=no
- PASs-MoE: Mitigating Misaligned Co-drift among Router and Experts via Pathway Activation Subspaces for Continual Learning (2026) — Annual Meeting of the Association for Computational Linguistics
    overlaps: problem=partial mechanism=no claim=no
- Who Speaks for the Trigger? Dynamic Expert Routing in Backdoored Mixture-of-Experts Transformers (2025) — Advances in Neural Information Processing Systems 38
    overlaps: problem=partial mechanism=partial claim=no

## v7r-004  [boundary_gap]

Topic: How to improve the scientific discovery capability of large language models

Claimed delta: Current domain-specific models achieve high precision (>90%) within their target domain but drop to near-random or significant error rates (<50% valid hypothesis rate) when applied to cross-domain tasks due to ontology mismatch and lack of unified reasoning schemas.

Supporting evidence (as extracted from the literature):
- [limitation/abstract] 该框架目前专注于生物启发的材料领域，未明确涵盖其他科学领域。 (2024)
- [problem/abstract] 当前面临的关键挑战包括数据质量与整合、模型可解释性以及缺乏标准基准测试 (2024)
- [limitation/abstract] 该综述对代理的讨论范围扩展到了化学领域之外的任何科学领域 (2024)
- [limitation/abstract] 当前LLMs在满足包含完整方法论和实验细节的细粒度假设发现标准方面存在能力上限限制。 (2026)

Candidate prior art (audit-recalled, unfiltered):
- SciAgents: Automating Scientific Discovery Through Bioinspired Multi‐Agent Intelligent Graph Reasoning (2024) — Advanced Materials
    overlaps: problem=yes mechanism=yes claim=partial
- PubTator 3.0: an AI-powered literature resource for unlocking biomedical knowledge (2024) — Nucleic Acids Research
    overlaps: problem=partial mechanism=partial claim=partial
- Causal Reasoning and Large Language Models: Opening a New Frontier for Causality (2023) — arXiv (Cornell University)
    overlaps: problem=partial mechanism=partial claim=partial
- Multi-modal Cross-domain Self-supervised Pre-training for fMRI and EEG Fusion (2024) — arXiv
    overlaps: problem=partial mechanism=partial claim=no
- Sparks of Artificial General Intelligence: Early experiments with GPT-4 (2023) — arXiv (Cornell University)
    overlaps: problem=partial mechanism=no claim=no

## v7r-005  [mechanism_gap]

Topic: Mechanistic interpretability of mixture-of-experts routing decisions

Claimed delta: Identifies the specific causal component of the (n+1)-th path (e.g., distinct feature capture vs. load balancing) that drives the performance improvement, moving beyond simple performance benchmarking.

Supporting evidence (as extracted from the literature):
- [comparison/conclusion] 对全部2,041个专家的完整普查LLM-as-a-judge评估发现，两种路由类型的专家纯度之间没有统计学显著差异。 (2025)
- [negative_result/method] 在Token-Choice路由中，某些专家的超容量比例可达20%-40%，导致大量被路由到这些专家的令牌被丢弃。 (2022)
- [negative_result/conclusion] Fedus等人提出的top-n路由优势在8倍更大规模的实验中被反转，发现top-(n+1)路由略优于top-n路由。 (2022)

Candidate prior art (audit-recalled, unfiltered):
- OmniGene-4: A Unified Bio-Language MoE Model with Router-Level Interpretability (2026) — bioRxiv (Cold Spring Harbor Laboratory)
    overlaps: problem=partial mechanism=partial claim=partial
- Task-Conditioned Routing Signatures in Sparse Mixture-of-Experts Transformers (2026) — arXiv
    overlaps: problem=partial mechanism=partial claim=partial
- DeepSeek-V3 Technical Report (2024) — arXiv (Cornell University)
    overlaps: problem=partial mechanism=no claim=no
- HyperMoE: Towards Better Mixture of Experts via Transferring Among Experts (2024) — Annual Meeting of the Association for Computational Linguistics
    overlaps: problem=partial mechanism=no claim=no
- Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer (2017) — arXiv (Cornell University)
    overlaps: problem=partial mechanism=no claim=no

## v7r-006  [untested_assumption]

Topic: How to improve the scientific discovery capability of large language models

Claimed delta: Introducing a causal-relevance filter for retrieval reduces hallucination rates in scientific hypothesis generation by >15% compared to standard semantic similarity retrieval, while maintaining retrieval speed.

Supporting evidence (as extracted from the literature):
- [method/abstract] 核心创新是相似性驱动检索(SDR)机制，该机制利用相似的嵌入来动态选择特定领域的演示(demos)以优化每个推理步骤。 (2025)
- [negative_result/introduction] 存在大型语言模型犯基本因果错误的实例。 (2023)
- [method/introduction] 提出了知识增强生成（KAG），充分利用知识图谱（KG）和检索增强生成（RAG）技术的互补特性。 (2025)
- [problem/conclusion] 形式化的Token因果性方法难以表示因果判断中的背景常识元素，如必要性、充分性、正常性和责任。 (2023)

Candidate prior art (audit-recalled, unfiltered):
- SciAgents: Automating Scientific Discovery Through Bioinspired Multi‐Agent Intelligent Graph Reasoning (2024) — Advanced Materials
    overlaps: problem=yes mechanism=partial claim=partial
- OpenScholar: Synthesizing Scientific Literature with Retrieval-augmented LMs (2024) — arXiv (Cornell University)
    overlaps: problem=partial mechanism=no claim=no
- Trustworthiness in Retrieval-Augmented Generation Systems: A Survey (2024) — arXiv (Cornell University)
    overlaps: problem=partial mechanism=no claim=no
- Retrieval-Augmented Generation with Graphs (GraphRAG) (2024) — arXiv (Cornell University)
    overlaps: problem=partial mechanism=partial claim=no
- Biomedical knowledge graph-optimized prompt generation for large language models (2024) — Bioinformatics
    overlaps: problem=partial mechanism=no claim=no

## v7r-007  [boundary_gap]

Topic: Mechanistic interpretability of mixture-of-experts routing decisions

Claimed delta: Extends causal tracing applicability from <7B parameter MoEs to >7B parameter MoEs while maintaining <10% localization error rate.

Supporting evidence (as extracted from the literature):
- [limitation/conclusion] 其追踪阶段需要对每个候选专家进行三次前向传播，这增加了计算成本。 (2026)
- [problem/introduction] 传统混合专家（MoE）方法（如Switch Transformer）在各层内独立路由专家，导致不同层的路由器做出的专家选择之间相关性不强。 (2025)
- [limitation/abstract] 研究范围限定于OLMoE-1B-7B-0125-Instruct这一特定模型，暗示其他模型或设置未被直接评估。 (2026)
- [limitation/conclusion] 分析是相关性的而非因果性的；测量了路由结构但未对专家进行干预。 (2026)

Candidate prior art (audit-recalled, unfiltered):
- OmniGene-4: A Unified Bio-Language MoE Model with Router-Level Interpretability (2026) — bioRxiv (Cold Spring Harbor Laboratory)
    overlaps: problem=partial mechanism=partial claim=partial
- How Does Chain of Thought Think? Mechanistic Interpretability of Chain-of-Thought Reasoning with Sparse Autoencoding (2026) — Proceedings of the AAAI Conference on Artificial Intelligence
    overlaps: problem=partial mechanism=partial claim=no
- Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability (2023) — arXiv (Cornell University)
    overlaps: problem=partial mechanism=partial claim=no
- Task-Conditioned Routing Signatures in Sparse Mixture-of-Experts Transformers (2026) — arXiv
    overlaps: problem=partial mechanism=no claim=no
- From Observation to Intervention: A Causal Audit of Expert Importance in Mixture-of-Experts Models (2026) — arXiv
    overlaps: problem=partial mechanism=partial claim=no

## v7r-008  [contradiction]

Topic: Context compression for retrieval-augmented generation with long documents

Claimed delta: Clarifies the mechanism behind the performance gap between attention-based and structural compression methods, specifically addressing their handling of logical dependencies.

Supporting evidence (as extracted from the literature):
- [comparison/method] 传统文档压缩方法通常关注文档的表面形式，而xRAG从模态融合的角度解决问题。 (2024)
- [comparison/introduction] 离散方法保留了与问题相关的单词但经常省略主语，而分块方法保留了单词的主语，从而保持更准确的语义信息。 (2025)

Candidate prior art (audit-recalled, unfiltered):
- InfoFlow KV: Information-Flow-Aware KV Recomputation for Long Context (2026) — arXiv (Cornell University)
    overlaps: problem=partial mechanism=no claim=no
- Hyper-RAG: combating LLM hallucinations using hypergraph-driven retrieval-augmented generation (2026) — Nature Communications
    overlaps: problem=partial mechanism=no claim=no
- DepCache: A KV Cache Management Framework for GraphRAG with Dependency Attention (2025) — Proceedings of the ACM on Management of Data
    overlaps: problem=partial mechanism=partial claim=no
- HieraSparse: Hierarchical Semi-Structured Sparse KV Attention (2026) — arXiv (Cornell University)
    overlaps: problem=partial mechanism=partial claim=no
- C 2 KV: Compressed and Composable KV Cache Reuse for Efficient LLM Inference (2026) — Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2
    overlaps: problem=partial mechanism=partial claim=no

## v7r-009  [scaling_regime]

Topic: Position bias (lost-in-the-middle) in long-context language models

Claimed delta: Establishes the first quantitative boundary for text-LMM position bias in the >128k token regime, contrasting with the established <128k baseline.

Supporting evidence (as extracted from the literature):
- [result/experiment] 结合YaRN插值，PoSE可将上下文窗口扩展至128k（表2数据）。 (2023)
- [result/experiment] PoSE在固定2k训练窗口下，将上下文扩展至16k/32k推理时，相比Full-length仅有轻微性能下降（基于GovReport和Proof-pile上的Perplexity评估）。 (2023)
- [result/introduction] 在视频模态下，Gemini 1.5 Pro在9.9M token（长达10.5小时）时保持该召回性能。 (2024)

Candidate prior art (audit-recalled, unfiltered):
- Sequential-NIAH: A Needle-In-A-Haystack Benchmark for Extracting Sequential Needles from Long Contexts (2025)
    overlaps: problem=yes mechanism=yes claim=yes
- Retrieval and Multi-Hop Reasoning in 1M-Token Context Windows: Evaluating LLMs on Classical Chinese Text (2026) — arXiv
    overlaps: problem=yes mechanism=yes claim=yes
- Long Context Scaling: Divide and Conquer via Multi-Agent Question-driven Collaboration (2025) — arXiv (Cornell University)
    overlaps: problem=partial mechanism=partial claim=partial
- Qwen2.5-1M Technical Report (2025) — arXiv (Cornell University)
    overlaps: problem=partial mechanism=no claim=partial
- LongAlign: A Recipe for Long Context Alignment of Large Language Models (2024)
    overlaps: problem=partial mechanism=no claim=no

## v7r-010  [untested_assumption]

Topic: Test-time training for adaptive inference in language models

Claimed delta: Improves accuracy on OOD tasks by >10% compared to static allocation baselines while maintaining <2x latency overhead.

Supporting evidence (as extracted from the literature):
- [limitation/introduction] 重复采样仅关注探索（即纯探索），缺乏显式的利用机制。 (2025)
- [problem/method] 现有的自适应采样方法主要关注计算资源的分配位置（如集中在更难的问题上），但样本仍从固定的条件分布中抽取，导致在基础分布下正确解概率较低时，增加采样往往收益递减。 (2026)
- [limitation/conclusion] 自适应策略仅在决策信号显著低于其门控动作的成本且在校准接近操作阈值时才能改善质量-成本前沿；否则，门控自身的成本及其校准误差会抵消节省。 (2026)
- [limitation/introduction] 现有的多轮设置探索与利用方法大多设计于推理时扩展的优势被充分认可之前，且使用固定的“宽度”（即从单个提示生成的答案数量作为固定超参数）。 (2025)

Candidate prior art (audit-recalled, unfiltered):
- Thermodynamic Theory of Token Compute Allocation -&amp;nbsp;Optimal Inference-Time Compute Distribution via Statistical Mechanics, with Empirical Validation (2026) — SSRN Electronic Journal
    overlaps: problem=partial mechanism=no claim=no
- Adaptive Test-Time Compute Allocation for Block Diffusion Language Models in Complex Reasoning (2026)
    overlaps: problem=partial mechanism=partial claim=no
- PERK: Long-Context Reasoning as Test-Time Learning (2025) — arXiv
    overlaps: problem=no mechanism=no claim=no
- Sleep-time Compute: Beyond Inference Scaling at Test-time (2025) — arXiv
    overlaps: problem=no mechanism=no claim=no
- Artificial Neural Variability for Deep Learning: On Overfitting, Noise Memorization, and Catastrophic Forgetting (2020) — arXiv
    overlaps: problem=no mechanism=no claim=no

## v7r-011  [cross_domain_transfer]

Topic: Mechanistic interpretability of mixture-of-experts routing decisions

Claimed delta: Demonstrates that dense attention heads can be decomposed into functional clusters with >80% purity using MoE-derived routing signature metrics, a capability not present in standard causal tracing which focuses on individual head attribution.

Supporting evidence (as extracted from the literature):
- [method/introduction] 激活修补可以在不同粒度级别进行，包括残差流、MLP输出、注意力头、单个神经元或稀疏自编码器特征。 (2024)
- [result/conclusion] 路由签名按任务类别强烈聚类。 (2026)
- [method/abstract] 实现了从干净到损坏残差流的修补技术。 (2026)
- [method/introduction] 引入了“路由签名”（routing signature）的概念，即给定提示下各层专家使用频率的紧凑表示，用于衡量路由模式是否按任务聚类。 (2026)

Candidate prior art (audit-recalled, unfiltered):
- Task-Conditioned Routing Signatures in Sparse Mixture-of-Experts Transformers (2026) — arXiv
    overlaps: problem=partial mechanism=partial claim=partial
- Identifying and Evaluating Inactive Heads in Pretrained LLMs (2025) — arXiv
    overlaps: problem=partial mechanism=no claim=no
- Modular Deep Learning (2023) — arXiv (Cornell University)
    overlaps: problem=partial mechanism=no claim=no
- Mixture-Of-Experts with Expert Choice Routing (2022) — Advances in Neural Information Processing Systems 35
    overlaps: problem=no mechanism=no claim=no
- OmniGene-4: A Unified Bio-Language MoE Model with Router-Level Interpretability (2026) — bioRxiv (Cold Spring Harbor Laboratory)
    overlaps: problem=no mechanism=no claim=no

## v7r-012  [mechanism_gap]

Topic: How to improve the scientific discovery capability of large language models

Claimed delta: Demonstrates that the collaborative mechanism contributes <10% to the total accuracy gain in novel hypothesis generation, whereas the verification mechanism contributes >40%, challenging the assumption that multi-agent diversity is the primary bottleneck solver.

Supporting evidence (as extracted from the literature):
- [result/abstract] 在所有下游任务中，BioScientist Agent在各种指标上均超越了现有的最先进基线模型。 (2025)
- [limitation/abstract] 使用此类技术存在产生错误结果的可能性。 (2023)
- [method/abstract] BioScientist Agent是一个端到端框架，整合了十亿级事实的生物医学知识图谱、用于表示学习和链接预测的变分图自编码器、用于遍历图谱恢复生物学合理机制路径的强化学习模块，以及编排这些组件的大型语言模型多智能体层。 (2025)
- [metric/abstract] 评估生成洞察的三个关键标准是：清晰度、有用性以及对理由的置信度。 (2025)

Candidate prior art (audit-recalled, unfiltered):
- Accelerating scientific discovery with Co-Scientist (2026) — Nature
    overlaps: problem=yes mechanism=no claim=no
- SciAgents: Automating Scientific Discovery Through Bioinspired Multi‐Agent Intelligent Graph Reasoning (2024) — Advanced Materials
    overlaps: problem=yes mechanism=no claim=no
- Many Heads Are Better Than One: Improved Scientific Idea Generation by A LLM-Based Multi-Agent System (2025) — Annual Meeting of the Association for Computational Linguistics
    overlaps: problem=yes mechanism=no claim=no
- Position: Multimodal Large Language Models Can Significantly Advance Scientific Reasoning (2026)
    overlaps: problem=partial mechanism=no claim=no
- Closing the Empirical Loop: Autonomous AI Agents Conduct End‐to‐end Research With Human Participants (2026) — Advanced Science
    overlaps: problem=partial mechanism=partial claim=no

## v7r-013  [contradiction]

Topic: Mechanistic interpretability of mixture-of-experts routing decisions

Claimed delta: Resolves the discrepancy between 'no purity difference' and 'task-clustered routing' by demonstrating that standard purity metrics fail to capture routing-level semantic distinctions that are visible only through routing signature analysis.

Supporting evidence (as extracted from the literature):
- [comparison/conclusion] 对全部2,041个专家的完整普查LLM-as-a-judge评估发现，两种路由类型的专家纯度之间没有统计学显著差异。 (2025)
- [result/conclusion] 路由签名按任务类别强烈聚类。 (2026)

Candidate prior art (audit-recalled, unfiltered):
- Task-Conditioned Routing Signatures in Sparse Mixture-of-Experts Transformers (2026) — arXiv
    overlaps: problem=yes mechanism=yes claim=yes
- OmniGene-4: A Unified Bio-Language MoE Model with Router-Level Interpretability (2026) — bioRxiv (Cold Spring Harbor Laboratory)
    overlaps: problem=partial mechanism=yes claim=partial
- The Myth of Expert Specialization in MoEs: Why Routing Reflects Geometry, Not Necessarily Domain Expertise (2026) — arXiv
    overlaps: problem=partial mechanism=no claim=partial
- Mixture-Of-Experts with Expert Choice Routing (2022) — Advances in Neural Information Processing Systems 35
    overlaps: problem=no mechanism=no claim=no
- Select-then-Route : Taxonomy guided Routing for LLMs (2025) — Conference on Empirical Methods in Natural Language Processing
    overlaps: problem=no mechanism=no claim=no

## v7r-014  [scaling_regime]

Topic: Position bias (lost-in-the-middle) in long-context language models

Claimed delta: Identifies the specific context length threshold where zero-shot inference fails compared to fine-tuned approaches, isolating the value of post-pretraining adaptation.

Supporting evidence (as extracted from the literature):
- [result/conclusion] YaRN在对少于约0.1%的原始预训练数据进行微调后，在上下文窗口扩展方面达到了最先进性能。 (2023)
- [limitation/introduction] 直接对目标长度输入进行微调以扩展窗口的方法因新位置索引引入的巨大干扰而效果有限。 (2023)
- [limitation/conclusion] 位置编码的一个反复出现的局限性是无法超越训练期间看到的上下文窗口进行泛化。 (2023)

Candidate prior art (audit-recalled, unfiltered):
- Effective Long-Context Scaling of Foundation Models (2024)
    overlaps: problem=yes mechanism=partial claim=yes
- Context Length Alone Hurts LLM Performance Despite Perfect Retrieval (2025)
    overlaps: problem=yes mechanism=yes claim=yes
- L-Eval: Instituting Standardized Evaluation for Long Context Language Models (2024)
    overlaps: problem=yes mechanism=no claim=partial
- PERK: Long-Context Reasoning as Test-Time Learning (2025) — arXiv
    overlaps: problem=partial mechanism=partial claim=partial
- Lost in the Middle: How Language Models Use Long Contexts (2024) — Transactions of the Association for Computational Linguistics
    overlaps: problem=partial mechanism=yes claim=partial

## v7r-015  [boundary_gap]

Topic: Mechanistic interpretability of mixture-of-experts routing decisions

Claimed delta: Improves the ability to distinguish expert-specific semantic contributions by >25% in imbalanced settings compared to standard activation patching.

Supporting evidence (as extracted from the literature):
- [method/introduction] 激活修补可以在不同粒度级别进行，包括残差流、MLP输出、注意力头、单个神经元或稀疏自编码器特征。 (2024)
- [problem/introduction] 传统MoE中独立的token选择通常导致专家负载不均衡，从而引起训练效率低下和模型次优训练。 (2022)
- [negative_result/method] 在Token-Choice路由中，某些专家的超容量比例可达20%-40%，导致大量被路由到这些专家的令牌被丢弃。 (2022)
- [limitation/conclusion] 模型Top-k准确率是离散的，会在阈值处过度代表变化，且对未跨越阈值的较大效应显示无变化。 (2024)

Candidate prior art (audit-recalled, unfiltered):
- OmniGene-4: A Unified Bio-Language MoE Model with Router-Level Interpretability (2026) — bioRxiv (Cold Spring Harbor Laboratory)
    overlaps: problem=partial mechanism=no claim=no
- Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability (2023) — arXiv (Cornell University)
    overlaps: problem=partial mechanism=partial claim=no
- CuMA: Aligning LLMs with Sparse Cultural Values via Demographic-Aware Mixture of Adapters (2026)
    overlaps: problem=partial mechanism=no claim=no
- Task-Conditioned Routing Signatures in Sparse Mixture-of-Experts Transformers (2026) — arXiv
    overlaps: problem=partial mechanism=no claim=no
- Least-Loaded Expert Parallelism: Load Balancing An Imbalanced Mixture-of-Experts (2026) — arXiv
    overlaps: problem=partial mechanism=no claim=no

## v7r-016  [cross_domain_transfer]

Topic: Mechanistic interpretability of mixture-of-experts routing decisions

Claimed delta: Provides a quantitative 'MoE Stability Index' that predicts load imbalance events 5 steps ahead with >90% accuracy, whereas current entropy-based metrics fail to predict such failures.

Supporting evidence (as extracted from the literature):
- [problem/introduction] 传统MoE中独立的token选择通常导致专家负载不均衡，从而引起训练效率低下和模型次优训练。 (2022)
- [method/abstract] 定义了电路稳定性指数（Scircuit）来正式化网络在主要注意力头被抑制时重新路由信息的有效性。 (2026)
- [metric/abstract] 计算所有层和头的直接逻辑归因矩阵。 (2026)
- [limitation/conclusion] 分析是相关性的而非因果性的；测量了路由结构但未对专家进行干预。 (2026)

Candidate prior art (audit-recalled, unfiltered):
- MoE-GPS: Guidlines for Prediction Strategy for Dynamic Expert Duplication in MoE Load Balancing (2025) — arXiv
    overlaps: problem=partial mechanism=partial claim=partial
- OD-MoE: On-Demand Expert Loading for Cacheless Edge-Distributed MoE Inference (2025) — arXiv (Cornell University)
    overlaps: problem=partial mechanism=partial claim=partial
- DeepSeek-V3 Technical Report (2024) — arXiv (Cornell University)
    overlaps: problem=partial mechanism=no claim=no
- OmniGene-4: A Unified Bio-Language MoE Model with Router-Level Interpretability (2026) — bioRxiv (Cold Spring Harbor Laboratory)
    overlaps: problem=partial mechanism=no claim=no
- Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer (2017) — arXiv (Cornell University)
    overlaps: problem=partial mechanism=no claim=no

## v7r-017  [scaling_regime]

Topic: How to improve the scientific discovery capability of large language models

Claimed delta: Identifying the 'noise threshold' in cross-domain KG integration where reasoning accuracy drops below random baseline, a regime not covered by single-domain studies.

Supporting evidence (as extracted from the literature):
- [limitation/abstract] 该框架目前专注于生物启发的材料领域，未明确涵盖其他科学领域。 (2024)
- [result/abstract] 在所有下游任务中，BioScientist Agent在各种指标上均超越了现有的最先进基线模型。 (2025)
- [result/abstract] SciAgents在生物启发材料领域揭示了以前被认为无关的隐藏跨学科关系，其规模、精度和探索能力超越了人类研究方法。 (2024)
- [limitation/abstract] 该综述对代理的讨论范围扩展到了化学领域之外的任何科学领域 (2024)

Candidate prior art (audit-recalled, unfiltered):
- The Impact of Large Language Models on Scientific Discovery: a Preliminary Study using GPT-4 (2023) — arXiv (Cornell University)
    overlaps: problem=partial mechanism=no claim=no
- DALK: Dynamic Co-Augmentation of LLMs and KG to answer Alzheimer’s Disease Questions with Scientific Literature (2024)
    overlaps: problem=partial mechanism=yes claim=no
- MedRAG: Enhancing Retrieval-augmented Generation with Knowledge Graph-Elicited Reasoning for Healthcare Copilot (2025) — The Web Conference
    overlaps: problem=no mechanism=partial claim=no
- Can Knowledge Graphs Reduce Hallucinations in LLMs? : A Survey (2024) — North American Chapter of the Association for Computational Linguistics
    overlaps: problem=no mechanism=no claim=no
- Retrieval-Augmented Generation for Large Language Models: A Survey (2023) — arXiv (Cornell University)
    overlaps: problem=no mechanism=no claim=no

## v7r-018  [scaling_regime]

Topic: Mechanistic interpretability of mixture-of-experts routing decisions

Claimed delta: Tests the upper bound of interpretability resolution by applying fine-grained causal tracing to models with 10x the expert density of previously studied baselines.

Supporting evidence (as extracted from the literature):
- [comparison/conclusion] 对全部2,041个专家的完整普查LLM-as-a-judge评估发现，两种路由类型的专家纯度之间没有统计学显著差异。 (2025)
- [metric/introduction] 使用内部凝聚力（Internal Cohesion）和外部纯度（External Purity）指标对专家行为进行定量可解释性分析。 (2025)
- [limitation/abstract] 研究范围限定于OLMoE-1B-7B-0125-Instruct这一特定模型，暗示其他模型或设置未被直接评估。 (2026)

Candidate prior art (audit-recalled, unfiltered):
- EXACT: Expert Attribution via Causal Tracing for Targeted Knowledge Unlearning in Mixture-of-Experts Language Models (2026) — Lecture notes in computer science
    overlaps: problem=partial mechanism=yes claim=partial
- OmniGene-4: A Unified Bio-Language MoE Model with Router-Level Interpretability (2026) — bioRxiv (Cold Spring Harbor Laboratory)
    overlaps: problem=partial mechanism=no claim=no
- Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability (2023) — arXiv (Cornell University)
    overlaps: problem=partial mechanism=partial claim=no
- Task-Conditioned Routing Signatures in Sparse Mixture-of-Experts Transformers (2026) — arXiv
    overlaps: problem=partial mechanism=no claim=no
- Efficient MoE Inference with Fine-Grained Scheduling of Disaggregated Expert Parallelism (2025) — arXiv
    overlaps: problem=no mechanism=no claim=no

## v7r-019  [mechanism_gap]

Topic: Test-time training for adaptive inference in language models

Claimed delta: Identifies that the failure is caused by optimization instability (noise accumulation) rather than representation mismatch, allowing for targeted regularization strategies.

Supporting evidence (as extracted from the literature):
- [negative_result/method] 截断所有步骤（即元梯度的二阶近似）导致性能大幅下降，即使在简单的QA1-1K设置中也是如此。 (2025)
- [comparison/experiment] 该方法与基于键值绑定(KVB)的先前工作TTT-KVB进行了对比，通过将层间重建损失替换为标准的下一个令牌预测损失，使TTT在测试时变为端到端(E2E)。 (2025)

Candidate prior art (audit-recalled, unfiltered):
- End-to-End Test-Time Training for Long Context (2025) — arXiv
    overlaps: problem=yes mechanism=partial claim=partial
- PERK: Long-Context Reasoning as Test-Time Learning (2025) — arXiv
    overlaps: problem=partial mechanism=no claim=no
- Learning to Discover at Test Time (2026) — arXiv
    overlaps: problem=partial mechanism=no claim=no
- Large Language Models as Optimizers (2023) — arXiv (Cornell University)
    overlaps: problem=partial mechanism=no claim=no
- Wider or Deeper?  Scaling LLM Inference-Time Compute with Adaptive Branching Tree Search (2025) — Advances in Neural Information Processing Systems 38
    overlaps: problem=partial mechanism=no claim=no

## v7r-020  [cross_domain_transfer]

Topic: How to improve the scientific discovery capability of large language models

Claimed delta: Shifts the paradigm from 'knowledge retrieval and synthesis' to 'causal structure learning', reducing hallucination rates in hypothesis generation by enforcing causal validity checks derived from systems biology methodologies.

Supporting evidence (as extracted from the literature):
- [limitation/method] 大型语言模型存在一个关键局限性，即倾向于在不发出警告的情况下生成错误，包括数学、编程、归因和更高层次的概念错误。 (2023)
- [negative_result/introduction] 存在大型语言模型犯基本因果错误的实例。 (2023)
- [method/introduction] 提出了知识增强生成（KAG），充分利用知识图谱（KG）和检索增强生成（RAG）技术的互补特性。 (2025)
- [method/abstract] BioScientist Agent是一个端到端框架，整合了十亿级事实的生物医学知识图谱、用于表示学习和链接预测的变分图自编码器、用于遍历图谱恢复生物学合理机制路径的强化学习模块，以及编排这些组件的大型语言模型多智能体层。 (2025)

Candidate prior art (audit-recalled, unfiltered):
- Unitless Unrestricted Markov-Consistent SCM Generation: Better Benchmark Datasets for Causal Discovery (2025) — arXiv
    overlaps: problem=partial mechanism=partial claim=partial
- SciAgents: Automating Scientific Discovery Through Bioinspired Multi‐Agent Intelligent Graph Reasoning (2024) — Advanced Materials
    overlaps: problem=partial mechanism=no claim=no
- Autonomous chemical research with large language models (2023) — Nature
    overlaps: problem=partial mechanism=no claim=no
- Knowledge Engineering Using Large Language Models (2023) — DROPS (Schloss Dagstuhl – Leibniz Center for Informatics)
    overlaps: problem=partial mechanism=no claim=no
- Dissociating language and thought in large language models (2024) — Trends in Cognitive Sciences
    overlaps: problem=partial mechanism=no claim=no

## v7r-021  [scaling_regime]

Topic: Test-time training for adaptive inference in language models

Claimed delta: Identifies the specific compute threshold (e.g., >500 token generations per query) where adaptive strategies cease to improve accuracy and begin to degrade due to compounding errors.

Supporting evidence (as extracted from the literature):
- [method/conclusion] 提出了自适应分支蒙特卡洛树搜索（AB-MCTS），这是一种在推理时增强大型语言模型在复杂任务上性能的新框架，通过有效整合多轮探索与利用，并基于外部反馈动态决定“拓宽”或“加深”搜索路径。 (2025)
- [result/abstract] 通过扩展睡眠时计算，在 Stateful GSM-Symbolic 上准确率最高提升13%，在 Stateful AIME 上最高提升18%。 (2025)
- [comparison/experiment] 与Best-of-25600基线进行比较，该基线匹配TTT-Discover的采样预算（50步 x 512次rollout = 25,600次采样）。 (2026)

Candidate prior art (audit-recalled, unfiltered):
- Adaptive Test-Time Compute Allocation for Block Diffusion Language Models in Complex Reasoning (2026)
    overlaps: problem=yes mechanism=yes claim=partial
- Adaptive Test-Time Compute Allocation with Evolving In-Context Demonstrations (2026) — arXiv
    overlaps: problem=yes mechanism=partial claim=partial
- Thermodynamic Theory of Token Compute Allocation -&amp;nbsp;Optimal Inference-Time Compute Distribution via Statistical Mechanics, with Empirical Validation (2026) — SSRN Electronic Journal
    overlaps: problem=yes mechanism=partial claim=partial
- Adaptive Computation in the LLM Era : A Unified Survey of Routing, Cascades, and Test-Time Scaling (2026)
    overlaps: problem=yes mechanism=no claim=no
- Sleep-time Compute: Beyond Inference Scaling at Test-time (2025) — arXiv
    overlaps: problem=partial mechanism=no claim=no

## v7r-022  [scaling_regime]

Topic: Test-time training for adaptive inference in language models

Claimed delta: Determines if the 'scaling law' of test-time compute holds for long contexts, or if a regime boundary exists around 10K tokens where gradient-based adaptation becomes ineffective.

Supporting evidence (as extracted from the literature):
- [dataset/experiment] 使用BabiLong框架进行Needles-in-a-Haystack (NIAH) 推理评估，包含单跳(QA1)、两跳(QA2)和三跳(QA3)任务，文档采样自FineWeb-Edu。 (2025)
- [result/conclusion] PERK展现出强大的测试时长外推能力，能够泛化到比训练数据长32倍的未见上下文。 (2025)
- [comparison/introduction] 本文方法与之前的长上下文TTT工作不同，其内循环直接在网络末端优化下一个token预测损失。 (2025)

Candidate prior art (audit-recalled, unfiltered):
- End-to-End Test-Time Training for Long Context (2025) — arXiv
    overlaps: problem=yes mechanism=yes claim=yes
- PERK: Long-Context Reasoning as Test-Time Learning (2025) — arXiv
    overlaps: problem=partial mechanism=yes claim=partial
- Sleep-time Compute: Beyond Inference Scaling at Test-time (2025) — arXiv
    overlaps: problem=partial mechanism=partial claim=partial
- Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context (2024) — arXiv (Cornell University)
    overlaps: problem=partial mechanism=no claim=no
- Learning to Discover at Test Time (2026) — arXiv
    overlaps: problem=partial mechanism=no claim=no

## v7r-023  [boundary_gap]

Topic: Test-time training for adaptive inference in language models

Claimed delta: Achieves >10% higher accuracy than truncated gradient methods (<=4 steps) while maintaining <5% latency overhead compared to zero-shot baselines.

Supporting evidence (as extracted from the literature):
- [negative_result/method] 截断所有步骤（即元梯度的二阶近似）导致性能大幅下降，即使在简单的QA1-1K设置中也是如此。 (2025)
- [limitation/method] 提示（Prompting）并不总是有益的：添加上下文可能因上下文长度效应、演示偏差或匹配不佳的示例而降低性能。 (2026)
- [limitation/method] 由于硬件资源限制，内层循环步骤数的分析仅扩展到4步，需要可扩展硬件设置下的更多系统实验。 (2025)

Candidate prior art (audit-recalled, unfiltered):
- End-to-End Test-Time Training for Long Context (2025) — arXiv
    overlaps: problem=partial mechanism=yes claim=partial
- Adaptive Computation in the LLM Era : A Unified Survey of Routing, Cascades, and Test-Time Scaling (2026)
    overlaps: problem=partial mechanism=no claim=no
- PERK: Long-Context Reasoning as Test-Time Learning (2025) — arXiv
    overlaps: problem=partial mechanism=partial claim=no
- Adapting in the Dark: Efficient and Stable Test-Time Adaptation for Black-Box Models (2026) — arXiv
    overlaps: problem=partial mechanism=partial claim=no
- Learning to Discover at Test Time (2026) — arXiv
    overlaps: problem=no mechanism=no claim=no

## v7r-024  [contradiction]

Topic: How to improve the scientific discovery capability of large language models

Claimed delta: Shows that while multi-agent systems increase the volume of generated hypotheses by >50%, the verifiability rate of these hypotheses drops below 5%, whereas human-generated hypotheses maintain a verifiability rate above 30%.

Supporting evidence (as extracted from the literature):
- [method/abstract] 引入了细粒度假设发现，要求假设包含完整的方法论和实验细节，并辅以理论框架来实证评估当前LLMs满足此标准的上限能力。 (2026)
- [result/abstract] SciAgents在生物启发材料领域揭示了以前被认为无关的隐藏跨学科关系，其规模、精度和探索能力超越了人类研究方法。 (2024)
- [limitation/abstract] 当前LLMs在满足包含完整方法论和实验细节的细粒度假设发现标准方面存在能力上限限制。 (2026)

Candidate prior art (audit-recalled, unfiltered):
- The future of fundamental science led by generative closed-loop artificial intelligence (2026) — Frontiers in Artificial Intelligence
    overlaps: problem=partial mechanism=no claim=no
- Artificial Intelligence agents for biological research: a survey (2026) — Briefings in Bioinformatics
    overlaps: problem=partial mechanism=no claim=no
- A Survey of Adaptation of Large Language Models to Idea and Hypothesis Generation: Downstream Task Adaptation, Knowledge Distillation Approaches and Challenges (2025) — ACM Computing Surveys
    overlaps: problem=partial mechanism=no claim=no
- A review of large language models and autonomous agents in chemistry (2024) — Chemical Science
    overlaps: problem=partial mechanism=no claim=no
- Exploring the role of large language models in the scientific method: from hypothesis to discovery (2025) — npj Artificial Intelligence
    overlaps: problem=partial mechanism=no claim=no

## v7r-025  [scaling_regime]

Topic: General-purpose uncertainty quantification for deep learning

Claimed delta: Establishes the performance boundary of lightweight UQ methods (MC Dropout) vs. heavy methods (Ensembles) specifically in the >1M sample regime, where linear scaling of ensemble cost becomes prohibitive and approximation errors may amplify.

Supporting evidence (as extracted from the literature):
- [limitation/introduction] 高斯过程（GP）和贝叶斯神经网络（BNN）分别在输入数据维度和参数空间方面扩展性不佳，对于现实世界的问题通常不可行。 (2021)
- [method/introduction] 蒙特卡洛丢弃（MCDropout）是深度学习中一种流行的不确定性估计方法，它在推理时应用Dropout来计算不确定性估计。 (2021)
- [limitation/experiment] MC Dropout 近似的是风险而非不确定性 (2021)
- [result/abstract] 集成模型在高Tg聚合物上表现出优越的性能，但计算成本显著更高 (2025)

Candidate prior art (audit-recalled, unfiltered):
- A Simple Approach to Improve Single-Model Deep Uncertainty via Distance-Awareness (2022) — arXiv (Cornell University)
    overlaps: problem=yes mechanism=no claim=partial
- Evaluating Scalable Bayesian Deep Learning Methods for Robust Computer Vision (2020) — 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)
    overlaps: problem=yes mechanism=yes claim=partial
- On Mixup Training: Improved Calibration and Predictive Uncertainty for Deep Neural Networks (2019) — Neural Information Processing Systems
    overlaps: problem=partial mechanism=no claim=partial
- A review of uncertainty quantification in deep learning: Techniques, applications and challenges (2021) — Information Fusion
    overlaps: problem=partial mechanism=no claim=no
- A survey of uncertainty in deep neural networks (2023) — Artificial Intelligence Review
    overlaps: problem=partial mechanism=no claim=no

## v7r-026  [untested_assumption]

Topic: Position bias (lost-in-the-middle) in long-context language models

Claimed delta: Demonstrates that while perplexity improves by <5% over baseline extrapolation methods, the retrieval accuracy for middle-positioned facts drops by >15% in high-noise settings compared to short-context baselines, revealing a gap between extrapolation success and semantic retention.

Supporting evidence (as extracted from the literature):
- [result/conclusion] YaRN在对少于约0.1%的原始预训练数据进行微调后，在上下文窗口扩展方面达到了最先进性能。 (2023)
- [result/experiment] 结合YaRN插值，PoSE可将上下文窗口扩展至128k（表2数据）。 (2023)
- [result/experiment] PoSE在固定2k训练窗口下，将上下文扩展至16k/32k推理时，相比Full-length仅有轻微性能下降（基于GovReport和Proof-pile上的Perplexity评估）。 (2023)
- [result/introduction] 随着提示中噪声增加（文档数量k增大），下游任务性能下降。 (2024)

Candidate prior art (audit-recalled, unfiltered):
- Found in the Middle: Calibrating Positional Attention Bias Improves Long Context Utilization (2024) — arXiv
    overlaps: problem=yes mechanism=yes claim=yes
- Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context (2024) — arXiv (Cornell University)
    overlaps: problem=partial mechanism=no claim=partial
- PERK: Long-Context Reasoning as Test-Time Learning (2025) — arXiv
    overlaps: problem=partial mechanism=partial claim=partial
- Untie the Knots: An Efficient Data Augmentation Strategy for Long-Context Pre-Training in Language Models (2025) — Annual Meeting of the Association for Computational Linguistics
    overlaps: problem=partial mechanism=no claim=partial
- Summary of a Haystack: A Challenge to Long-Context LLMs and RAG Systems (2024)
    overlaps: problem=partial mechanism=no claim=no

## v7r-027  [missing_evaluation]

Topic: Test-time training for adaptive inference in language models

Claimed delta: We provide the first empirical analysis of catastrophic forgetting in test-time training, showing that while OOD accuracy improves by >5%, base distribution accuracy drops by >10% for models <7B parameters, whereas larger models (>70B) maintain base accuracy within 2%.

Supporting evidence (as extracted from the literature):
- [negative_result/method] 截断所有步骤（即元梯度的二阶近似）导致性能大幅下降，即使在简单的QA1-1K设置中也是如此。 (2025)
- [limitation/introduction] 修改采样行为的方法通常需要为每个查询运行专门的搜索过程或辅助探索机制，增加了推理时的开销并使部署复杂化。 (2026)
- [method/introduction] 后训练方法旨在解决灾难性遗忘、奖励黑客行为和推理时间权衡等关键挑战。 (2025)

Candidate prior art (audit-recalled, unfiltered):
- Learning to Discover at Test Time (2026) — arXiv
    overlaps: problem=partial mechanism=yes claim=no
- End-to-End Test-Time Training for Long Context (2025) — arXiv
    overlaps: problem=partial mechanism=yes claim=no
- Overcoming Catastrophic Forgetting in Graph Neural Networks (2021) — Proceedings of the AAAI Conference on Artificial Intelligence
    overlaps: problem=partial mechanism=no claim=no
- On the Usage of Continual Learning for Out-of-Distribution Generalization in Pre-trained Language Models of Code (2023) — ESEC/SIGSOFT FSE
    overlaps: problem=partial mechanism=no claim=no
- Parameter-efficient fine-tuning of large-scale pre-trained language models (2023) — Nature Machine Intelligence
    overlaps: problem=no mechanism=no claim=no

## v7r-028  [missing_evaluation]

Topic: Position bias (lost-in-the-middle) in long-context language models

Claimed delta: Provides a concrete trade-off curve between inference cost and position-bias mitigation efficacy, filling the gap left by purely accuracy-focused evaluations.

Supporting evidence (as extracted from the literature):
- [negative_result/introduction] Chen等人使用32个A100 GPU将LLaMA模型的上下文从2k扩展到8k，使用128个A100 GPU扩展到更大上下文，这种开销使得无法将上下文窗口扩展到极端长度。 (2023)
- [result/abstract] 针对RALM设置专门优化的文档检索和排序机制可以进一步提升性能。 (2023)
- [method/experiment] YaRN方法结合了注意力缩放（公式14）和“NTK-by-parts”插值，通过缩放复数旋转位置嵌入来改变注意力机制，无需修改代码且无额外开销。 (2023)

Candidate prior art (audit-recalled, unfiltered):
- Lost in the Middle: How Language Models Use Long Contexts (2024) — Transactions of the Association for Computational Linguistics
    overlaps: problem=yes mechanism=no claim=partial
- LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression (2024)
    overlaps: problem=partial mechanism=partial claim=no
- Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context (2024) — arXiv (Cornell University)
    overlaps: problem=partial mechanism=no claim=no
- Understand What LLM Needs: Dual Preference Alignment for Retrieval-Augmented Generation (2025) — The Web Conference
    overlaps: problem=partial mechanism=yes claim=no
- Leveraging long context in retrieval augmented language models for medical question answering (2025) — npj Digital Medicine
    overlaps: problem=partial mechanism=partial claim=no

## v7r-029  [missing_evaluation]

Topic: Test-time training for adaptive inference in language models

Claimed delta: We demonstrate that self-correction fails to correct hallucinations in >50% of cases in code generation tasks, often entering error loops, whereas it succeeds in >80% of cases in factual QA tasks.

Supporting evidence (as extracted from the literature):
- [limitation/introduction] 重复采样仅关注探索（即纯探索），缺乏显式的利用机制。 (2025)
- [limitation/conclusion] 当前方法仅适用于具有连续奖励的问题。 (2026)

Candidate prior art (audit-recalled, unfiltered):
- AnalogCoder: Analog Circuit Design via Training-Free Code Generation (2025) — Proceedings of the AAAI Conference on Artificial Intelligence
    overlaps: problem=partial mechanism=partial claim=partial
- LLM Hallucinations in Practical Code Generation: Phenomena, Mechanism, and Mitigation (2025) — Proceedings of the ACM on software engineering.
    overlaps: problem=partial mechanism=no claim=partial
- Adaptive Test-Time Compute Allocation for Block Diffusion Language Models in Complex Reasoning (2026)
    overlaps: problem=partial mechanism=partial claim=no
- Hallucination is Inevitable: An Innate Limitation of Large Language Models (2024) — arXiv (Cornell University)
    overlaps: problem=partial mechanism=no claim=no
- Large Language Models are Better Reasoners with Self-Verification (2023) — Conference on Empirical Methods in Natural Language Processing
    overlaps: problem=partial mechanism=yes claim=no

## v7r-030  [cross_domain_transfer]

Topic: How to improve the scientific discovery capability of large language models

Claimed delta: Transforms hypothesis generation from a passive retrieval/synthesis task to an active experimental design task, reducing the number of required validation steps by >20% while achieving equivalent or higher confidence in discovered laws.

Supporting evidence (as extracted from the literature):
- [metric/abstract] 评估指标包括与人类的一致性水平以及解释质量 (2023)
- [method/method] 将大语言模型与知识图谱集成可以通过从非结构化文本中自动提取结构化信息来增强其能力，从而改进知识图谱的构建和增强。 (2024)
- [method/introduction] 提出了知识增强生成（KAG），充分利用知识图谱（KG）和检索增强生成（RAG）技术的互补特性。 (2025)
- [method/abstract] BioScientist Agent是一个端到端框架，整合了十亿级事实的生物医学知识图谱、用于表示学习和链接预测的变分图自编码器、用于遍历图谱恢复生物学合理机制路径的强化学习模块，以及编排这些组件的大型语言模型多智能体层。 (2025)

Candidate prior art (audit-recalled, unfiltered):
- Towards a fully automated algorithm driven platform for biosystems design (2019) — Nature Communications
    overlaps: problem=yes mechanism=yes claim=partial
- Automated Hypothesis Validation with Agentic Sequential Falsifications (2025) — arXiv (Cornell University)
    overlaps: problem=partial mechanism=no claim=partial
- From inductive reasoning to automated scientific discovery: foundations and methods with large language models (2026)
    overlaps: problem=partial mechanism=no claim=partial
- SciAgents: Automating Scientific Discovery Through Bioinspired Multi‐Agent Intelligent Graph Reasoning (2024) — Advanced Materials
    overlaps: problem=partial mechanism=no claim=no
- Causal Reasoning and Large Language Models: Opening a New Frontier for Causality (2023) — arXiv (Cornell University)
    overlaps: problem=partial mechanism=no claim=no
