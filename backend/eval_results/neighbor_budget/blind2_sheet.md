# 独立人工抽查（盲，第二版）

**在做完全部判断之前，不要打开 `blind2_key.json`。**

本次与上次不同：上次我是在 v1 rubric 隐含标准下判断的，等于被同一套规则锚定。
这次先读 v2 rubric 的判定口径，然后**只依据单元内容与证据**独立判断，不看任何机器判定。

## 判定口径（v2）

判断单位是 **delta（CLAIMED DELTA 那句具体断言）**，不是话题。三种情形必须区分：

- **(A) 相邻**：同主题/同任务/同模型族，但没有检验这个具体断言。**相邻不等于覆盖。**
- **(B) 部分覆盖**：检验了同一断言或其必要子命题，且结果在可比条件下可迁移（同机制、同量级/同域）。
- **(C) 已确立**：已在可比设定下证明了该 delta。

判定：

- `NONE` — 无工作确立该 delta。情形 (A) 单独**不构成** PARTIAL。
- `PARTIAL` — 已确立该 delta 的实质部分，即 (B)。残留空白真实但狭窄。
- `FULL` — 已确立该 delta，即 (C)。这是假空白。

**两个方向都不许默认。** 对所有条目答 NONE 与对所有条目答 PARTIAL 一样无用。

## 判断步骤（请照做）

1. 先用自己的话一句话复述 delta（写进理由里）
2. 问：有谁**检验了那句断言**（不是只做了这个话题）？若没有 → NONE
3. 若有，是只确立了一部分/可迁移的邻近结果 → PARTIAL；已确立 → FULL

共 14 个单元。

---

## 1. `nb-002` — gap_type: `boundary_gap`

**CLAIMED DELTA**
    Prior work validates specialization only via selection-frequency statistics under a single
    balancing regime and separately shows expert ablation gives no gain; it never tests the
    boundary at which the frequency-based specialization proxy decouples from causal importance
    once balancing pressure is increased, so the validity range of the dominant metric is
    undocumented.

**描述**
    Specialization analyses in MoE interpretability rely mainly on expert selection frequency
    across domains and are acknowledged as lacking stricter validation; causal
    ablation/intervention results show that disabling experts yields no gain and that
    interventions outside middle layers or on multilingual-specialist experts degrade performance.
    What is untested is whether selection-frequency-based specialization judgments remain valid
    when the balancing regime is changed (auxiliary-loss weight / capacity factor), i.e., whether
    the frequency metric itself breaks down as a proxy under strong balancing pressure.

**目标设定**
    Sparse MoE language models trained with auxiliary load-balancing losses and capacity factors,
    evaluated for whether selection-frequency specialization metrics track causal expert
    importance.

**观察到的问题**
    Specialization conclusions rest on domain-wise expert selection frequencies without stricter
    verification ([paper]); any expert-deactivation scheme failed to improve performance ([paper])
    and off-middle-layer or multilingual-specialist interventions only degraded performance
    ([paper], [paper]); micro-batch balancing constraints both limit specialization and hurt
    performance ([paper]).

**已有覆盖**
    Selection-frequency specialization analysis ([paper]), routing purity/utilization metrics
    ([paper]), bias-only balancing variants ([paper]), and causal expert-ablation studies
    ([paper], [paper], [paper]).

**缺失能力**
    A test of whether selection-frequency specialization metrics stay consistent with causal
    expert importance specifically under varying balancing strength (auxiliary-loss weight and
    capacity factor), where the frequency distribution is deliberately flattened.

**可检验假设**
    As the auxiliary balancing-loss weight increases across at least 4 values spanning a 10x
    range, the rank correlation between selection-frequency specialization scores and causal
    expert-ablation importance drops below 0.3 at the highest weight, versus above 0.6 at the
    lowest weight.

**证伪条件**
    If the rank correlation between selection-frequency specialization and causal ablation
    importance stays above 0.6 at every tested balancing weight across the 10x range, then the
    frequency metric does not break down under balancing pressure and the claimed boundary does
    not exist.

**证据（5 条）**
- [negative_result] 在中间层之外进行干预或针对多语言专用专家的干预只会导致性能下降。
- [limitation] 对专业化的分析主要集中于不同领域的专家选择频率，未进行更严格的验证。
- [negative_result] 任何专家停用方案都未能带来性能提升
- [problem] 微批次平衡约束不仅限制了专家专业化，还对模型性能产生负面影响。
- [metric] 通过路由纯度和专家利用率指标直接量化专家专业化程度，提供标准困惑度指标无法捕捉的路由动态可见性

**我的判断**：`[ ] NONE   [ ] PARTIAL   [ ] FULL`

**复述 delta + 理由**：

    

**有相邻工作吗**（有/无）：

---

## 2. `nb-012` — gap_type: `untested_assumption`

**CLAIMED DELTA**
    Existing work reports post-hoc quality of specific compressors; the missing delta is an
    isolated, ratio-matched test of whether increasing information density by >20% at constant
    token budget changes faithfulness scores by more than measurement noise, which no supplied
    method isolates.

**描述**
    Multiple context-compression methods for long-document RAG implicitly assume that compressing
    retrieved context toward higher information density (via lexical self-information pruning, TF-
    IDF discrete compression, semantic AMR-level redundancy removal, or diversity-based selection)
    preserves or improves downstream grounding and faithfulness. This density-preserves-grounding
    assumption is inherited across independent methods but is never directly tested as a
    controlled hypothesis; the evidence shows density gains and quality trade-offs measured
    separately, not whether increased density itself causes grounding loss.

**目标设定**
    Long-document RAG systems where retrieved context approaches or exceeds the LLM context window
    and must be compressed via query-focused summarization, token pruning, or redundancy removal
    before generation.

**观察到的问题**
    Compression methods optimize for removing redundant/low-information tokens and report accuracy
    or FLOPs, but none isolate the causal effect of raising information density on
    faithfulness/grounding. SelCon-style lexical pruning is sensitive to granularity ([paper]),
    performs poorly on small backbones ([paper]), token-level compression breaks syntax and
    traceability ([paper]), and aggressive compression degrades long-tail reasoning ([paper]) —
    yet density is treated as beneficial by default rather than as a variable that could itself
    harm grounding.

**已有覆盖**
    Prior work compares compression families (keyword extraction, summarization, SelCon)
    ([paper]), measures compression-rate vs quality trade-offs ([paper], [paper]), reports FLOPs
    reduction ([paper]), uses factuality/claim-based metrics ([paper]), and offers faithfulness-
    oriented evaluation tools such as RAGAS faithfulness and RGB counterfactual robustness
    ([paper], [paper]). These cover measuring quality after compression but not disentangling
    density from grounding.

**缺失能力**
    A controlled experiment that holds compression ratio fixed while varying only the achieved
    information density (redundancy removed per retained token), and measures
    grounding/faithfulness change attributable to density alone rather than to content loss.

**可检验假设**
    At a fixed memory-token budget, increasing compressed-context information density by more than
    20% (measured as removed-redundancy fraction per retained token) reduces RAGAS-style
    faithfulness or attribution overlap by more than 5 percentage points relative to a lower-
    density compression of the same length, across at least 3 long-document QA datasets.

**证伪条件**
    The opportunity is closed if a ratio-matched experiment shows that a >20% increase in
    information density at constant token budget leaves faithfulness/attribution scores unchanged
    within 5 percentage points (or improves them), demonstrating that density and grounding are
    decoupled and the assumption is harmless.

**证据（6 条）**
- [limitation] SelCon的有效性依赖于确定合适的冗余去除粒度，因此对词汇单元的选择较为敏感。
- [method] 为验证实验1的全面性与多样性结果，采用基于事实性主张(claim)的度量方法，并使用Claimify提取事实性主张
- [comparison] 对比三种上下文压缩方法：上下文关键词抽取、上下文摘要以及Selective Context (SelCon)。
- [negative_result] 以bloom-560m作为骨干LLM时，在两个数据集的In区间∆呈负趋势，SelCon表现也不理想
- [metric] RAGAS工具从上下文相关性、忠实度和答案相关性三个维度评估检索质量与生成质量。
- [limitation] 现有隐式记忆方法存在局限：激进压缩常导致长尾推理性能下降，且纯潜在方法缺乏大规模多文档流所需的结构化检索能力。

**我的判断**：`[ ] NONE   [ ] PARTIAL   [ ] FULL`

**复述 delta + 理由**：

    

**有相邻工作吗**（有/无）：

---

## 3. `nb-017` — gap_type: `untested_assumption`

**CLAIMED DELTA**
    Provide the first iteration-sweep characterization showing whether LLM TTT quality is monotone
    in update count; claim that beyond a measurable threshold (e.g., >10 update iterations)
    quality gains saturate or reverse for at least one parameter-efficient TTT method, which
    current work assumes away.

**描述**
    Multiple test-time-training (TTT) and test-time-adaptation (TTA) works for LLMs implicitly
    assume that performing online parameter updates at inference reliably improves adaptation
    quality relative to its cost, i.e., that more/longer update iterations monotonically buy
    better task performance. This assumption is inherited without direct verification: TTT-NN
    fine-tunes on retrieved neighbors per test instance ([paper]), QueST adapts via LoRA at test
    time ([paper]), uTTT shares fast-weight states across grouped layers ([paper]), and EATA
    performs source-free online updates ([paper]). Yet the same body of evidence reports that full
    50-iteration TTT can be computationally infeasible ([paper]), that prompt-tuning under low-
    resource settings causes catastrophic WER degradation ([paper]), and that online generation-
    time updates can be costly and unstable ([paper]). No supplied study measures the update-
    iteration-vs-quality curve to test whether the assumed monotone benefit holds.

**目标设定**
    Online test-time adaptation of LLMs where parameter-efficient updates (LoRA/adapter/fast
    weights) are applied per test instance or per stream under distribution shift, with a bounded
    compute budget.

**观察到的问题**
    The field assumes test-time parameter updates improve adaptation, but evidence shows high
    iteration counts are infeasible ([paper]), prompt tuning can catastrophically degrade output
    ([paper]), and online updates are unstable ([paper]). Whether additional update iterations
    actually yield monotone gains, or saturate/reverse, is never directly measured.

**已有覆盖**
    Papers establish that TTT/TTA can help: TTT-NN improves effective capacity/perplexity
    ([paper]), QueST builds auxiliary supervision and adapts with LoRA ([paper]), uTTT trains
    faster via routing ([paper]), EATA needs no source data and preserves source accuracy
    ([paper]). Efficiency metrics (GFLOPs, wall-clock, GPU memory) are introduced ([paper]) and
    parameter overhead is reported ([paper]).

**缺失能力**
    A controlled measurement of the update-iteration / update-magnitude versus adaptation-quality
    curve (including saturation and reversal points) for LLM test-time training, rather than
    assuming monotone improvement.

**可检验假设**
    For at least one parameter-efficient LLM TTT method, increasing test-time update iterations
    from 1 to 50 yields non-monotone adaptation quality: quality peaks at fewer than 25 iterations
    and drops by more than 5% relative to the peak when running all 50 iterations.

**证伪条件**
    If a sweep over 1 to 50 update iterations shows adaptation quality is monotone non-decreasing
    (within 1% noise) up to 50 iterations for all tested parameter-efficient TTT methods, the
    untested monotone-benefit assumption is validated and this gap is closed.

**证据（6 条）**
- [method] QueST通过生成辅助问题–解答对直接从输入查询构建监督，并在测试时通过LoRA进行高效参数自适应。
- [method] 提出测试时最近邻训练（TTT-NN）：对每个测试实例，从大型数据库中检索其最近邻，并在应用模型前在这些邻居上微调模型。
- [method] 将层分组，每组共享一份快速权重状态：p=L时各组为层私有，p=1时所有深度共享同一状态
- [comparison] 先前TTA在生成过程中执行在线更新，可能代价高且不稳定。
- [comparison] EATA在在线设置下不需要源数据，且能保持源精度，而其他方法如微调、持续学习等不满足此特性。
- [limitation] 测试时训练显著增加推理时的计算开销，某些应用运行全部50次迭代可能不可行

**我的判断**：`[ ] NONE   [ ] PARTIAL   [ ] FULL`

**复述 delta + 理由**：

    

**有相邻工作吗**（有/无）：

---

## 4. `nb-028` — gap_type: `missing_evaluation`

**CLAIMED DELTA**
    Introduce and validate an evaluation protocol that measures the
    novelty–feasibility–truthfulness trade-off via downstream validation on at least 20 expert-
    curated discovery tasks, with contamination controls and non-self-referential judging,
    yielding rankings that differ from existing retrieval/citation-alignment benchmarks by more
    than 15% in rank agreement.

**描述**
    Hypothesis-generation and discovery benchmarks are acknowledged to rarely connect generated
    hypotheses to downstream experimental validation, and novelty judgments by LLMs are disputed
    by domain experts; yet no benchmark measures the novelty–feasibility–truthfulness trade-off
    through actual downstream verification, nor controls for data contamination and self-
    referential LLM-as-judge bias in this scientific-discovery setting.

**目标设定**
    Evaluation of LLM-generated scientific hypotheses and research questions, where quality must
    be judged on novelty, feasibility, truthfulness and downstream experimental validation rather
    than literature-retrieval or citation-alignment proxies.

**观察到的问题**
    Existing benchmarks emphasize literature retrieval, citation alignment or specific intensive
    analysis tasks and implicitly encode prior assumptions via which data was collected; benchmark
    design seldom links generated hypotheses to downstream validation, leaving the
    novelty–feasibility–truthfulness tension unresolved; experts strongly disagree with LLM
    novelty judgments; knowledge-oriented benchmarks suffer contamination and LLM-as-Judge risks
    circular self-validation.

**已有覆盖**
    DiscoveryBench tests high-specificity hypothesis generation given datasets and discovery
    goals; InnovatorBench benchmarks end-to-end innovation on 20 curated tasks; CausalGame covers
    six causal dimensions; ChemBench and ScienceQA-style knowledge benchmarks exist; factuality
    can be scored as supported-claims ratio; author-anchored research questions serve as
    references.

**缺失能力**
    A contamination-controlled, expert-anchored evaluation protocol that scores generated
    hypotheses on novelty, feasibility and truthfulness jointly and traces them to downstream
    experimental validation outcomes.

**可检验假设**
    An expert-anchored, downstream-validation-linked evaluation protocol applied to at least 20
    discovery tasks will produce system rankings whose agreement with existing retrieval/citation-
    alignment benchmarks is below 85%, and LLM-as-Judge novelty scores will diverge from expert
    novelty ratings on more than 20% of evaluated hypotheses.

**证伪条件**
    The gap is closed if an existing benchmark already reports joint
    novelty–feasibility–truthfulness scores tied to downstream experimental validation on at least
    20 tasks while controlling for training-data contamination and judge self-reference, and its
    rankings agree with retrieval/citation-alignment benchmarks above 85%.

**证据（6 条）**
- [limitation] LLM-as-Judge方法存在自我指涉陷阱，当GPT-4等前沿模型在MT-Bench或BiGGenBench中评估对话深度或指令保真度时，可能循环验证自身的生成模式，偏向熟悉性而非真实能力。
- [limitation] 作者锚定RQ只是参考RQ，并非同一背景唯一有效的RQ；一个背景可以支持许多好问题。
- [limitation] 现有基准存在局限：数据集的存在本身已隐含对潜在结果的先验假设，体现在收集了什么数据以及如何收集上；在实验构思阶段研究者可能尚无数据集，而现有基准主要强调文献检索、引用对齐或特定密集型分析任务。
- [limitation] 基准设计很少将生成的假设与下游验证连接起来，导致新颖性-可行性-真实性张力未解决。
- [comparison] DiscoveryBench通过测试模型在给定数据集和发现目标下生成高特异性假设的能力来评估自动数据驱动发现。
- [limitation] 知识导向基准面临数据污染问题：随着模型在日益扩展的网络规模数据集上训练，基准问题出现在训练数据中的概率增加，可能抬高分数并损害评估有效性。

**我的判断**：`[ ] NONE   [ ] PARTIAL   [ ] FULL`

**复述 delta + 理由**：

    

**有相邻工作吗**（有/无）：

---

## 5. `nb-032` — gap_type: `missing_evaluation`

**CLAIMED DELTA**
    Existing studies report ID calibration and at most single-dataset OOD detection; none
    quantifies the fraction of OOD/adversarial cases where Bayesian approximation or deep-ensemble
    confidence drops by less than 15% while accuracy falls by more than 20%, which we claim occurs
    for the majority of tested shift settings.

**描述**
    Bayesian and ensemble uncertainty methods are widely compared on calibration (ECE, NLL) and
    sometimes runtime, but the supplied evidence repeatedly documents that OOD robustness of these
    methods is asserted rather than measured: future work explicitly calls for deeper OOD
    evaluation, standard ensembles can be overconfident in unseen scenarios, and non-invasive
    methods degrade under OOD/adversarial conditions. The missing capability is a direct,
    quantitative measurement of the frequency and degree of overconfidence (calibration failure)
    of Bayesian approximations and deep ensembles under OOD and adversarial inputs, using a shared
    protocol.

**目标设定**
    Deep neural networks using Bayesian approximations (VI, MC dropout) or deep ensembles for
    uncertainty quantification, deployed on inputs subject to distribution shift (OOD data) and
    adversarial perturbation.

**观察到的问题**
    Methods report ID calibration but remain overconfident on OOD inputs (confidence drop <=15%),
    some are underconfident in-distribution yet still overconfident OOD, and BNN complexity does
    not clearly beat simpler alternatives; authors themselves flag OOD robustness evaluation as
    future work, indicating it was not measured.

**已有覆盖**
    Controlled comparisons of deterministic NN, BNN, MC dropout and GP on calibration metrics
    (accuracy, NLL, Brier, ECE, reliability diagrams); runtime comparisons among BNN techniques;
    OOD detection tested with SVHN; calibration measured with 20 bins.

**缺失能力**
    A quantified, shared-protocol measurement of overconfidence frequency and magnitude (e.g., ECE
    increase and confidence-drop) for Bayesian approximations and deep ensembles under OOD and
    adversarial inputs across multiple domains.

**可检验假设**
    For both MC dropout and deep ensembles, on at least 3 OOD or adversarial shift settings, mean
    predictive confidence will decrease by less than 15% while accuracy decreases by more than
    20%, yielding an ECE increase of more than 0.10 relative to in-distribution calibration.

**证伪条件**
    The gap is closed if a shared-protocol evaluation finds that for MC dropout and deep ensembles
    the confidence drop exceeds 15% or the accuracy drop stays below 20% (i.e., ECE increase
    <=0.10) in at least 3 OOD/adversarial settings.

**证据（6 条）**
- [comparison] EMM位于对角线上方，在ID数据上信心不足（smECE=0.193），但在OOD输入上仍然过度自信。
- [problem] 标准集成方法（即用不同随机初始化独立训练的模型）在未见过的场景中可能产生过度自信的预测，因为集成中的模型往往表现出相似的行为。
- [comparison] 预训练模型、ABNN和Whitebox在ID与偏移输入之间的预测置信度几乎没有下降（≤15%），在OOD上仍然过度自信。
- [metric] 使用期望校准误差(ECE)衡量MC-Dropout不确定性估计与模型误差的校准程度，并比较不同Dropout率与层放置配置（编码器稀释、解码器稀释、全局稀释）
- [negative_result] 尽管BNN模型复杂，但并未明显优于更简单的替代方法；HNN模型可以成为有竞争力且平衡的替代方案，且更轻量、训练更快更直接。
- [future_work] 未来工作应更深入评估方法对OOD情况的鲁棒性，例如从训练集中排除一个飞行类别或使用不同分布偏移程度的训练和测试单元组合。

**我的判断**：`[ ] NONE   [ ] PARTIAL   [ ] FULL`

**复述 delta + 理由**：

    

**有相邻工作吗**（有/无）：

---

## 6. `nb-040` — gap_type: `missing_evaluation`

**CLAIMED DELTA**
    Provide the first quantitative end-to-end evaluation showing whether a multi-agent scientific-
    discovery pipeline improves hypothesis quality and factuality by more than 10 percentage
    points over both a single LLM and an LLM-RAG baseline on at least 2 standardized benchmarks.

**描述**
    Multi-agent LLM architectures for end-to-end scientific discovery (literature understanding →
    hypothesis generation → experimental design → validation) are described and asserted to
    outperform single LLMs, but no empirical head-to-head evaluation against independent LLM or
    LLM-RAG baselines on standardized benchmarks has been reported; the superiority claim remains
    an expectation rather than a measurement.

**目标设定**
    Autonomous multi-agent LLM systems that perform the full scientific discovery pipeline
    (knowledge retrieval, hypothesis generation, experimental design, computational experiments,
    result verification) compared against single-LLM and LLM-RAG pipelines.

**观察到的问题**
    Proposed multi-agent frameworks explicitly defer empirical validation to future work and only
    expect broader research capability than a single language model; meanwhile other work reports
    that strong subtask performance does not translate into reliable end-to-end experimental
    autonomy, and that assistant systems remain unreliable in open-ended applications.

**已有覆盖**
    Architectures with evidence-grounded retrieval, specialized scientific agents, independent
    hypothesis critique, automated experiment design, provenance logs and human-in-the-loop
    approval are specified; structured evidence retrieval and critic/verifier agents are described
    as methods; some component-level comparisons exist (ExpeL vs ReAct/Act).

**缺失能力**
    A controlled, standardized-benchmark comparison of the full multi-agent pipeline versus
    independent LLM and LLM-RAG systems measuring hypothesis quality, factuality, and
    experimental-design outcomes.

**可检验假设**
    On at least 2 standardized scientific-discovery benchmarks, a multi-agent pipeline will exceed
    both a single-LLM baseline and an LLM-RAG baseline by more than 10 percentage points on
    factuality score F=(C_s/C_t)x100 and on expert-rated hypothesis quality, whereas the current
    literature reports no such measured comparison.

**证伪条件**
    The gap is closed if a published study reports a controlled comparison of a full multi-agent
    discovery pipeline against single-LLM and LLM-RAG baselines on at least 2 standardized
    benchmarks with hypothesis-quality and factuality metrics, showing either a gain above 10
    percentage points or demonstrating that no measurable difference exists.

**证据（6 条）**
- [future_work] 未来的实证研究应实现所提出的架构，并使用标准化基准将其与独立LLM和LLM-RAG系统进行比较，评估假设质量、事实性和实验等方面。
- [method] 提出包含证据支撑的知识检索、专用科学智能体、独立假设批评、自动化实验设计、计算实验、结果验证、溯源审计日志与人类在环审批的多智能体架构。
- [result] 助手系统在检索和综合方面相对成熟，但在开放式应用中仍不可靠。
- [future_work] 该框架预期能证明多智能体系统比单一语言模型具备更广泛的研究能力（属预期结果而非实测）。
- [comparison] ExpeL在所有领域上持续优于ReAct和Act基线。
- [negative_result] 强大的子任务表现尚不能转化为可靠的端到端实验自主性。

**我的判断**：`[ ] NONE   [ ] PARTIAL   [ ] FULL`

**复述 delta + 理由**：

    

**有相邻工作吗**（有/无）：

---

## 7. `nb-045` — gap_type: `missing_evaluation`

**CLAIMED DELTA**
    Report the first quantified false-premise-rejection rate and control-variable-preservation
    rate for end-to-end research agents on at least 40 tasks, metrics absent from current agent
    evaluations.

**描述**
    End-to-end autonomous research agents are claimed to integrate literature mining, hypothesis
    generation, and experiment design, but there is no evaluation of their ability to reject false
    premises embedded in the input literature/data, nor of whether they correctly ignore control
    variables or handle noisy data during experiment design. Failure modes such as ignoring
    controls and mishandling uncertainty are asserted but never quantified on a shared benchmark.

**目标设定**
    Closed-loop LLM-driven autonomous research systems that take a research question plus
    retrieved literature and raw data, then produce experiment code, intermediate results, and a
    final report.

**观察到的问题**
    Current models struggle with sustained multi-step scientific reasoning, logical organization,
    citation fidelity, and grounding, and agent systems are argued to be unfit for autonomous
    natural science due to training/deployment strategy rather than scale; yet no benchmark
    measures false-premise rejection or control-variable handling rates in these end-to-end
    pipelines.

**已有覆盖**
    Benchmarks exist that hide the target paper and ask systems to generate artifacts (code,
    results, report) judged against the original publication, and datasets cover retrieval
    robustness, negative-sample rejection, and counterfactual robustness in general QA settings;
    iterative Plan-Act-Observe-Refine loops are proposed.

**缺失能力**
    A controlled evaluation injecting false premises and noisy/confounded data into the
    literature-mining and experiment-design stages, measuring the rate at which agents detect and
    reject invalid premises and preserve control variables.

**可检验假设**
    On at least 40 research tasks with deliberately injected false premises, state-of-the-art
    autonomous research agents will explicitly flag or reject fewer than 50% of false premises,
    and will omit at least one required control variable in more than 30% of generated experiment
    protocols.

**证伪条件**
    The gap is closed if a benchmark shows autonomous research agents rejecting at least 90% of
    injected false premises and preserving all required control variables in at least 90% of
    generated protocols across 40 or more tasks.

**证据（6 条）**
- [dataset] CRUD是一个基准，覆盖创意生成、知识密集型问答、错误纠正和摘要等任务
- [method] 基准判定系统能否基于 (q, L, D) 生成高质量研究产物，以及这些产物是否达到或超越目标论文 p*
- [limitation] 作者主张当前智能体系统并非为自主自然科学而构建，其缺陷源于训练与部署策略本身，而非规模或工具不足。
- [negative_result] 在层次化目录构建、文献综述生成、表格综合和综述评估等应用任务中，当前模型虽能产出流畅输出，但在逻辑组织、引用保真度和长文连贯综合方面仍有困难
- [dataset] RGB是一个基准，用于评估检索质量、生成质量、噪声鲁棒性、负样本拒绝、信息整合和反事实鲁棒性等维度
- [negative_result] 当前模型在持续多步科学推理上仍弱于表层理解，且多模态科学产物仍会引发幻觉与接地失败

**我的判断**：`[ ] NONE   [ ] PARTIAL   [ ] FULL`

**复述 delta + 理由**：

    

**有相邻工作吗**（有/无）：

---

## 8. `nb-062` — gap_type: `cross_domain_transfer`

**CLAIMED DELTA**
    Prospective pre-registered scoring reduces the measured model-vs-human novelty advantage
    inflated by LLM judges by at least 50% relative to retrospective LLM-judge scoring on the same
    hypothesis set

**描述**
    The target setting is LLM-driven automated scientific discovery, where generated hypotheses
    are judged by LLM judges or overlap metrics. Evidence [paper] and [paper] document that LLM
    judges systematically inflate novelty ('novelty illusion'), strongly favoring model outputs
    over author-written baselines, and [paper] shows that anchor-overlap scores conflate
    memorization with genuine novelty recovery. The source-domain mechanism is the mature
    statistical methodology of pre-registration combined with null-hypothesis significance testing
    as used in clinical trials and open-science registries: hypotheses and analysis plans are
    time-stamped and registered before data collection, which makes post-hoc novelty/validity
    claims auditable and separates confirmatory from exploratory results. Evidence [paper]
    explicitly imports the null-hypothesis-testing idea into a benchmark for predicting scientific
    outcomes, but only as a scoring device for outcome prediction, not as a registry that fixes
    hypotheses before evaluation. Evidence [paper] proposes a centralized pre-registration
    repository for all AI-generated hypotheses but does not connect it to the judge-bias problem.
    The mapping: pre-registration timestamps + confirmatory/exploratory separation -> a defense
    against LLM-judge novelty inflation, because a hypothesis registered before seeing the target
    paper cannot be retroactively credited with rediscovering it. Smallest missing capability: a
    pre-registered hypothesis ledger whose entries are scored against later-arriving ground truth
    rather than by an LLM judge.

**目标设定**
    LLM-driven closed-loop scientific discovery pipelines evaluated on benchmarks such as
    RCBench/DiscoveryBench, where hypotheses are scored for novelty and validity

**观察到的问题**
    Novelty and validity of LLM-generated hypotheses are assessed by LLM judges that amplify
    novelty hallucination and favor model outputs, while anchor-overlap metrics cannot distinguish
    memorization from genuine discovery

**已有覆盖**
    Null-hypothesis-testing-inspired outcome-prediction benchmarking ([paper]) and a proposal for
    a centralized pre-registration repository ([paper]) exist, plus anchor-overlap memory
    diagnostics ([paper])

**缺失能力**
    A pre-registration ledger that fixes AI-generated hypotheses before ground-truth exposure and
    scores them prospectively, replacing retrospective LLM-judge novelty ratings

**可检验假设**
    On at least 100 AI-generated hypotheses across the 40 RCBench tasks, prospective pre-
    registered scoring will shrink the LLM-judge-reported model-over-baseline novelty advantage by
    more than 50% compared with retrospective judging

**证伪条件**
    If pre-registered prospective scoring reproduces the same model-over-baseline novelty
    advantage (within 10 percentage points) as retrospective LLM-judge scoring, the transfer claim
    is falsified

**证据（6 条）**
- [result] 在比较设置下，LLM评委放大了新颖性幻觉，严重偏向模型输出而非作者撰写的基线
- [future_work] 建议建立集中式预注册仓库，收录所有AI生成的假设。
- [result] LLM评判者一致将模型生成的RQ评为高度新颖，产生“新颖性幻象”
- [dataset] RCBench 当前包含 10 个科学领域的 40 个任务
- [metric] 作者锚点重叠分数作为记忆诊断：若模型记住论文或其RQ则分数应高；分数低可能表示生成了另一个有效问题或未恢复作者方向
- [method] 提出了一个初步框架，用于基准测试模型预测科学成果的能力，灵感来自零假设检验，评估模型推理出合理科学结果而非简单检索的能力。

**我的判断**：`[ ] NONE   [ ] PARTIAL   [ ] FULL`

**复述 delta + 理由**：

    

**有相邻工作吗**（有/无）：

---

## 9. `nb-065` — gap_type: `untested_assumption`

**CLAIMED DELTA**
    First quantification of the self-supervised-loss vs task-accuracy relationship in LLM TTT;
    claim that for at least one input-derived objective, a >10% relative reduction in adaptation
    loss corresponds to less than 2% accuracy gain (i.e., weak or absent alignment), contradicting
    the inherited assumption.

**描述**
    A cluster of LLM test-time-training works implicitly assumes that self-supervised signals
    derived from the input itself (input perplexity minimization, generated auxiliary Q&A pairs,
    nearest-neighbor fine-tuning targets) are aligned with the specific reasoning structure a
    single query requires, so that minimizing them improves downstream answer quality. This
    assumption is shared across methods without being tested: TTT-NN fine-tunes on retrieved
    nearest neighbors ([paper]), QueST generates auxiliary question-answer pairs as supervision
    ([paper]), and generic TTT minimizes the input's own perplexity ([paper] explicitly notes this
    objective is not directly aligned with the specific reasoning structure a query needs).
    Meanwhile evidence shows self-derived supervision can fail badly: prompt tuning causes
    catastrophic WER degradation ([paper]), and online updates during generation are costly and
    unstable ([paper]). No supplied study directly tests whether lower self-supervised adaptation
    loss implies higher task accuracy.

**目标设定**
    Self-supervised test-time training for LLMs where the adaptation objective is constructed from
    the test input (perplexity, auxiliary Q&A, or neighbor-based targets) and used to update
    parameters before answering.

**观察到的问题**
    Methods optimize input-derived self-supervised objectives and assume this transfers to the
    query's required reasoning, but [paper] admits the perplexity objective is misaligned with
    per-query reasoning structure, and [paper]/[paper] show such self-supervised updates can
    degrade outputs or become unstable. The loss-to-accuracy link is assumed, not measured.

**已有覆盖**
    QueST constructs auxiliary question-answer supervision and adapts with LoRA ([paper]); TTT-NN
    retrieves neighbors and fine-tunes on them ([paper]); TTT-NN improves language-modeling
    perplexity/capacity ([paper]); consistent-label TTT-Probe improves OOD accuracy using the
    model's own full-budget answers as supervision ([paper]).

**缺失能力**
    A direct correlation analysis between reduction in the self-supervised adaptation objective
    and change in downstream task accuracy across queries, to verify or refute the assumed
    alignment.

**可检验假设**
    Across a held-out set of at least 200 shifted-distribution queries, the per-query relative
    reduction in the self-supervised TTT objective correlates with downstream accuracy change at
    Pearson r < 0.3, meaning a >10% loss reduction yields under 2% average accuracy improvement.

**证伪条件**
    If per-query self-supervised loss reduction correlates strongly (Pearson r >= 0.6) with
    downstream accuracy gain, and a >10% loss reduction yields more than 5% average accuracy
    improvement, then the alignment assumption holds and this gap is closed.

**证据（6 条）**
- [method] QueST通过生成辅助问题–解答对直接从输入查询构建监督，并在测试时通过LoRA进行高效参数自适应。
- [method] 提出测试时最近邻训练（TTT-NN）：对每个测试实例，从大型数据库中检索其最近邻，并在应用模型前在这些邻居上微调模型。
- [result] 一致标签的TTT-Probe适用于无真实标签的部署场景：在OOD数据上相对一致基线提升34.0%，且仅需模型自身的全预算答案作为监督
- [comparison] 先前TTA在生成过程中执行在线更新，可能代价高且不稳定。
- [problem] 现有测试时训练方法通过最小化输入自身的困惑度来更新参数，但该目标与单个查询所需的特定推理结构并不直接对齐。
- [result] 测试时训练结合最近邻可以提升语言建模任务中模型的有效容量（以困惑度衡量）

**我的判断**：`[ ] NONE   [ ] PARTIAL   [ ] FULL`

**复述 delta + 理由**：

    

**有相邻工作吗**（有/无）：

---

## 10. `nb-081` — gap_type: `mechanism_gap`

**CLAIMED DELTA**
    Existing comparisons report 9.5x speed and 12.61x memory savings alongside marginal accuracy
    gains, but do not vary budget while fixing objective; the proposed gap targets demonstrating
    whether a >2x relaxation of the memory budget changes accuracy by more than 5 percentage
    points, isolating the causal role of the efficiency mechanism.

**描述**
    Several TTA/TTT methods report large efficiency wins (e.g., ZERO being 9.5x faster and using
    12.61x less memory than TPT) and some report accuracy near zero-shot levels (StatA down only
    1%), but the causal link between the specific design choice enabling efficiency and the
    resulting accuracy behavior is unexplained. In particular, it is unknown whether the
    efficiency-preserving mechanism (e.g., avoiding per-view forward passes, or entropy-
    minimization-free objectives) is what causes the limited accuracy gains, or whether accuracy
    and efficiency are independent axes. The evidence documents both the efficiency advantage and
    the marginal-accuracy outcome without establishing a causal relationship.

**目标设定**
    Parameter-efficient online TTT for LLMs/VLMs operating under fixed memory and latency budgets,
    where the method must trade adaptation cost against adaptation benefit.

**观察到的问题**
    Efficient TTA designs achieve order-of-magnitude speed/memory savings while accuracy remains
    marginal or slightly below zero-shot (StatA -1%), but no work establishes whether the
    efficiency mechanism causally limits achievable accuracy or whether the two are decoupled.

**已有覆盖**
    Work reports efficiency metrics (GFLOPs, wall-clock time, GPU memory), compares efficient
    methods against costly ones (ZERO vs TPT), and notes that entropy minimization is suboptimal
    and that single-sample TTA should not be expected to beat online TTA. These are correlational
    observations across differing settings, not causal isolation.

**缺失能力**
    A controlled ablation that holds the adaptation objective fixed while varying only the
    efficiency mechanism (memory/latency budget), to determine whether relaxing the budget
    causally increases accuracy or leaves it unchanged.

**可检验假设**
    When the adaptation objective is held constant and only the memory/latency budget is relaxed
    by at least 2x, accuracy changes by less than 5 percentage points, indicating that the
    efficiency mechanism is not the causal bottleneck for the marginal accuracy gains reported.

**证伪条件**
    The gap is closed if a controlled budget-sweep experiment shows that relaxing the memory
    budget by 2x while fixing the objective improves accuracy by more than 5 percentage points,
    proving the efficiency mechanism directly causes the accuracy ceiling and rendering the
    decoupling hypothesis false.

**证据（6 条）**
- [comparison] ZERO比TPT快9.5倍，内存占用少12.61倍，在时间和空间上均实现了一个数量级的计算节省。
- [metric] 效率指标定义为每种腐蚀类型下推理与自适应的平均总运行时间
- [limitation] 熵最小化在测试时是次优的
- [result] StatA在所有类别场景下相比零样本下降1%。
- [negative_result] 大多数现有TTA方法相比早期基线TPT的性能提升是边际性的
- [metric] 引入新的评估指标，包括GFLOPs、挂钟时间和GPU内存使用量，以衡量在线场景中的效率。

**我的判断**：`[ ] NONE   [ ] PARTIAL   [ ] FULL`

**复述 delta + 理由**：

    

**有相邻工作吗**（有/无）：

---

## 11. `nb-096` — gap_type: `untested_assumption`

**CLAIMED DELTA**
    Existing work either reports observational specialization metrics OR runs localized
    interventions, but does not quantify the agreement between the two rankings across >100
    experts and >=8 layers in a single MoE, nor report the correlation coefficient between
    selection-frequency rank and ablation causal-effect rank.

**描述**
    Multiple MoE interpretability and routing-analysis works implicitly assume that observable
    router statistics (routing purity, expert utilization, expert selection frequency, group
    routing summaries) are valid proxies for causal expert importance and functional
    specialization. This proxy assumption is inherited without being tested as a causal claim:
    specialization is inferred from selection-frequency differences across domains/languages
    rather than from interventional validation.

**目标设定**
    Sparse-gated MoE language models (Switch Transformer / GShard-style FFN routing), analyzing
    whether routing statistics predict causal expert function.

**观察到的问题**
    Specialization analyses rest mainly on expert selection frequencies per domain and are
    explicitly noted as lacking stricter verification; observational routing metrics were found
    insufficient to predict ablation-based causal importance; and local inference-observable
    router statistics failed to beat chance at predicting sign flips. Yet downstream claims about
    specialization and selective repair continue to lean on these observational proxies.

**已有覆盖**
    Prior work reports routing purity/utilization metrics, compares observational group-routing
    summaries against expert-ablation causal importance, and performs masking interventions on
    language-specific experts at early/mid/late layers. Some works intervene on routing mediators
    under paired clean/quantized computation.

**缺失能力**
    A direct, systematic test of whether observational routing statistics causally track expert
    function — i.e., correlating selection-frequency/purity rankings with intervention-derived
    causal-effect rankings across many experts and layers within the same model.

**可检验假设**
    Across at least 128 experts and 8 layers in a sparse MoE, the Spearman correlation between
    selection-frequency-based specialization rank and ablation-derived causal-importance rank will
    be below 0.3, showing observational routing statistics are weak proxies for causal expert
    function.

**证伪条件**
    If selection-frequency/purity rankings correlate with ablation causal-effect rankings at
    Spearman rho >= 0.7 across >=128 experts and >=8 layers, the proxy assumption holds and the
    opportunity is closed.

**证据（5 条）**
- [comparison] Engmann等人在OLMoE、Qwen1.5-MoE和DeepSeek上审计群体路由摘要是否能预测专家消融的因果重要性，发现观察指标不足；本文则在配对干净/量化计算下干预路由中介。
- [limitation] 对专业化的分析主要集中于不同领域的专家选择频率，未进行更严格的验证。
- [limitation] 检测极限结果仅限于所测试的局部、推理可观测的路由器统计量特征族，未涵盖其他潜在预测器。
- [metric] 通过路由纯度和专家利用率指标直接量化专家专业化程度，提供标准困惑度指标无法捕捉的路由动态可见性
- [negative_result] 在测试的局部、推理可观测的路由器统计量中，没有发现能高于随机水平预测翻转损失符号的预测器——这是一个经验性的收益检测障碍，限制了仅基于该特征族的选择性修复。

**我的判断**：`[ ] NONE   [ ] PARTIAL   [ ] FULL`

**复述 delta + 理由**：

    

**有相邻工作吗**（有/无）：

---

## 12. `val-002` — gap_type: `cross_domain_transfer`

**CLAIMED DELTA**
    Apparent self-consistency is mediated by the transitivity of the pairwise-agreement graph
    rather than by mean pairwise similarity: a spectral centrality score over that graph separates
    transitive consensus from hub artifacts, so mean similarity misranks hallucinated answers
    precisely when one sampled answer dominates the comparison graph.

**描述**
    Self-consistency hallucination scores are built on pairwise lexical/NLI agreement between
    sampled answers, which treats every pair as equally informative and ignores that agreement
    structure is transitive and graph-valued. In adjacent fields, rank aggregation and spectral
    ranking (Condorcet/Bradley-Terry aggregation, PageRank-style centrality over a comparison
    graph) provide mature machinery for turning noisy pairwise comparisons into a single
    calibrated score with a significance estimate. The transfer is to view the sampled answers as
    candidates in a noisy pairwise-comparison tournament and compute a spectral/rank-aggregation
    consensus score instead of averaging pairwise similarities; the target literature currently
    reduces the comparison graph to a mean similarity or an entailment count and discards the
    graph topology that determines whether apparent agreement is transitive consensus or a hub
    artifact.

**目标设定**
    Self-consistency sampling for hallucination detection: multiple sampled answers are compared
    pairwise (lexical similarity or mutual entailment) and the aggregate disagreement is used as a
    hallucination signal.

**观察到的问题**
    Pairwise agreement is collapsed into a scalar (mean similarity, entailment count, or cluster
    entropy), so the detector cannot distinguish transitive consensus among many answers from a
    single answer that is spuriously similar to all others, and it requires a held-out validation
    set to calibrate the resulting threshold.

**已有覆盖**
    Lexical-similarity consistency baselines, entailment-graph consistency (mutual entailment
    among samples), semantic entropy/energy over clusters, and perplexity/energy OOD scores. These
    use pairwise comparisons but reduce them to means or counts; none applies rank-aggregation or
    spectral centrality to the comparison graph.

**缺失能力**
    A graph-spectral / rank-aggregation consensus score over the pairwise comparison graph of
    sampled answers, with a significance estimate, that replaces mean pairwise similarity and
    removes the need for a separate validation-set threshold.

**可检验假设**
    Across at least 3 open-domain QA datasets and 2 model families, a spectral-centrality
    consensus score over the pairwise-entailment graph achieves higher AUROC than mean pairwise
    similarity for detecting hallucinated answers by more than 5 points, and the gain concentrates
    on queries whose comparison graph has a single high-degree hub node.

**证伪条件**
    The opportunity is closed if spectral centrality over the pairwise-agreement graph is within 2
    AUROC points of mean pairwise similarity on every query stratum, including the single-hub
    stratum, showing that graph topology carries no information beyond the mean.

**证据（6 条）**
- [method] 基于自一致性直觉：对同一问题多次高温随机采样生成之间的相互蕴含程度在幻觉时低于正确回答时。
- [comparison] 对比基线包括基于不确定性的Perplexity和Length-normalized Entropy、基于一致性的Lexical Similarity，以及OOD检测方法Energy score。
- [limitation] NLI指标在低资源语言上失效。
- [problem] 现有的基于不确定性或一致性的幻觉检测指标在logit空间或语言空间中操作，忽略了LLM内部状态中保留的稠密语义信息。
- [comparison] 相比单次生成方法，利用多次生成的方法更有效，因为它们能捕获更广泛的幻觉信号。
- [negative_result] 预测准确率并非不确定性质量的可靠代理：高准确率模型仍可能校准差或无法区分正确与错误预测

**我的判断**：`[ ] NONE   [ ] PARTIAL   [ ] FULL`

**复述 delta + 理由**：

    

**有相邻工作吗**（有/无）：

---

## 13. `val-005` — gap_type: `scaling_regime`

**CLAIMED DELTA**
    Consistency-based hallucination detection fails through a knowledge-coverage boundary rather
    than a uniform weakness: its discriminative power is preserved for facts the model has seen
    and collapses abruptly once fact rarity crosses a threshold, because below that coverage level
    all samples converge on one wrong cluster and the score becomes constant regardless of
    correctness.

**描述**
    Multi-sample consistency detectors are evaluated on benchmarks whose items are drawn from
    well-populated knowledge (TruthfulQA-derived, HaluEval, standard QA), and the documented
    failure mode is that when the model lacks knowledge of a query it emits several semantically
    identical wrong answers and the consistency score stays at zero. What is never reported is the
    regime boundary in terms of how rare or obscure the queried fact is: the point along a
    knowledge-frequency axis where consistency-based detection stops working. Existing controlled
    studies use small item sets and do not stratify items by how often the underlying fact appears
    in pretraining data, so the operating range of the detector in fact-rarity space is
    unmeasured.

**目标设定**
    Open-domain QA where the queried facts vary in how frequently they occur in pretraining data,
    from common entities to long-tail/rare facts, scored by multi-sample consistency.

**观察到的问题**
    When the model has no knowledge of a query, five semantically identical wrong answers yield a
    consistency score of zero and the hallucination goes undetected; structured gates also miss
    confident fabrication on conflicting-evidence items. These failures are reported qualitatively
    but not located on any measurable rarity axis, and the evaluations use only tens to ~100 items
    per model.

**已有覆盖**
    Work documents the single-cluster zero-score failure, shows semantic energy consistently
    outperforms semantic entropy across models/datasets, compares multi-sample methods favorably
    to single-pass methods, and uses TruthfulQA/HallucinationEval plus a 100-item abstention
    stress test; controlled regimes use ~10 items each and larger-scale replication on mature
    benchmarks is listed as future work.

**缺失能力**
    An empirical curve of consistency-based detection performance as a function of fact rarity
    (pretraining frequency of the queried fact), identifying the rarity threshold beyond which the
    consistency signal carries no information; no current work stratifies benchmark items by fact
    frequency or reports such a boundary.

**可检验假设**
    Stratifying open-domain QA items by fact rarity, consistency-based detection AUROC exceeds
    0.75 on the most frequent quartile but falls below 0.55 on the rarest quartile, with the
    transition occurring within a 2-quartile span, at a fixed 10 samples per item.

**证伪条件**
    If consistency-based detection AUROC varies by less than 10 points across fact-rarity
    quartiles, or if the rarest quartile still yields AUROC above 0.70, then the claim that a
    knowledge-coverage boundary governs the detector's operating range is falsified.

**证据（6 条）**
- [negative_result] 当模型对查询q3缺乏知识时，即使生成5个语义相同的错误答案，HSE仍为0，无法检测错误。
- [comparison] 在所有模型和数据集上，语义能量一致优于语义熵。
- [dataset] 使用100项TruthfulQA派生的无上下文弃权压力测试进行评估
- [limitation] 语义熵在单聚类场景下完全失效，其值始终为零，无法提供有效的不确定性区分。
- [future_work] 需要在SQuAD 2.0、HaluEval等成熟基准上进行更大规模的评估以获得发表级统计功效。
- [limitation] 受控评估每个模型仅用50个条目、每个regime 10个，样本规模小。

**我的判断**：`[ ] NONE   [ ] PARTIAL   [ ] FULL`

**复述 delta + 理由**：

    

**有相邻工作吗**（有/无）：

---

## 14. `val-006` — gap_type: `contradiction`

**CLAIMED DELTA**
    The apparent advantage of low-rank PEFT over full fine-tuning for knowledge retention is not
    caused by the low-rank constraint itself but is mediated by how much of the base model's
    representational subspace the update occupies; once updates stay within a narrow aligned
    subspace the retention advantage disappears regardless of adapter rank.

**描述**
    The literature disagrees on whether LoRA-style PEFT preserves general knowledge better than
    full fine-tuning. One line of evidence reports LoRA mitigates forgetting more than weight
    decay/dropout and keeps generation diversity, while another reports LoRA underperforms full
    fine-tuning and that the LoRA-vs-full-FT difference may reverse with model scale. The flip is
    unexplained: no account identifies which variable (scale, task complexity, or the forgetting
    metric used) drives the reversal.

**目标设定**
    LoRA (and comparable low-rank PEFT) versus full fine-tuning of decoder LLMs, evaluated on
    general-knowledge retention benchmarks after single-task adaptation.

**观察到的问题**
    Evidence simultaneously supports 'LoRA forgets less than full FT / regularization' and 'LoRA
    performs worse than full FT', with the direction of the LoRA-vs-full-FT gap left as open
    future work and explicitly suspected to depend on model scale.

**已有覆盖**
    Prior work compares LoRA against weight decay/dropout for forgetting, measures
    learning/forgetting via Open LLM Leaderboard benchmarks, and notes that PEFT effectiveness
    varies by architecture and task; one paper flags scale-dependence of the LoRA-vs-full-FT
    difference as unresolved.

**缺失能力**
    A causal account of why the LoRA-versus-full-FT forgetting comparison flips across settings —
    i.e., identifying the mediating variable rather than adding another benchmark row.

**可检验假设**
    Across at least 3 base-model scales spanning an order of magnitude, the sign of the LoRA-
    minus-full-FT forgetting gap is predicted by measured update-subspace overlap with the
    pretrained weights, not by adapter rank or parameter count; controlling for that overlap
    removes the sign flip.

**证伪条件**
    If, after matching update-subspace overlap between LoRA and full fine-tuning runs, the
    forgetting gap retains the same sign and magnitude as the unmatched comparison (i.e., overlap
    explains <10% of the variance in the gap), the mediation claim is false.

**证据（5 条）**
- [result] LoRA有助于保持更多样化的生成。
- [future_work] LoRA与全量微调的差异是否随模型规模变化尚待严格研究，作者将其留作未来工作。
- [comparison] LoRA比常见的正则化技术（如权重衰减和dropout）更能减轻遗忘。
- [problem] 现有PEFT方法（如LoRA）相比全量微调往往表现不佳，尤其在涉及复杂数据集时。
- [comparison] 有证据表明LoRA性能不如全量微调。

**我的判断**：`[ ] NONE   [ ] PARTIAL   [ ] FULL`

**复述 delta + 理由**：

    

**有相邻工作吗**（有/无）：

---
