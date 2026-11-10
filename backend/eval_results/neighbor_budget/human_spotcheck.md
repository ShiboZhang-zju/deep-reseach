# 近邻预算实验 — 人工抽查表（盲）

**在做完判断前不要打开 `spotcheck_key.json`。**

对每个单元独立判断，只依据下方呈现的单位内容与证据，不要推测它来自哪一组。

判断口径（与独立评审一致）：

- `NONE` — 存在可发表空白，未见先行工作覆盖该 delta
- `PARTIAL` — 有先行工作部分覆盖，但该 delta 的特定操作化未被直接报告
- `FULL` — 该 delta 已被先行工作实质覆盖
- `UNCERTAIN` — 现有材料不足以判断

共 21 个单元（分层抽样 ≥20%）。

---

## 1. `nb-010` — gap

- 层：emerging_sparse
- gap_type：boundary_gap

**描述**
    Load-balancing strategies (auxiliary loss, capacity factor, bias-only balancing) are known to
    constrain expert specialization and interact with training stability, but their effect on
    routing behavior and specialization has only been analyzed through coarse expert-selection
    frequency counts over limited configurations, without stricter validation. Analyses of
    specialization are largely restricted to selection-frequency statistics, and the balance-loss
    weight had to be manually adjusted under training instability, indicating the regime where
    standard balancing behaves poorly is not characterized. It remains untested how specialization
    differs systematically across balancing schemes and how sensitive routing is to input
    perturbation under each.

**目标设置**
    Sparse MoE language models trained with different load-balancing regimes (GShard auxiliary loss,
    Switch capacity factor, bias-only Loss-Free balancing), measuring expert specialization and
    routing stability.

**观察到的问题**
    Micro-batch balancing constraints limit expert specialization and negatively affect performance;
    training instability forced lowering peak learning rate and adjusting the load-balancing loss
    weight; and specialization analysis relied mainly on per-domain expert selection frequencies
    without stricter validation. This leaves the boundary of which balancing regimes and
    perturbation levels break down routing specialization uncharacterized.

**已有覆盖**
    Work compares balancing variants (auxiliary loss vs bias-only Loss-Free rule), reports that
    micro-batch balance harms specialization and performance, notes training instability requiring
    balance-loss tuning, and quantifies specialization via routing purity and expert utilization
    metrics across many configurations.

**缺失能力**
    A controlled evaluation relating distinct load-balancing strategies to measured specialization
    and to routing stability under input perturbation, with stricter validation than selection-
    frequency counts.

**声称的 delta**
    Existing evidence ties load balancing to reduced specialization and to training instability but
    only measures specialization via selection-frequency statistics over limited configurations; it
    does not establish where, across balancing schemes and perturbation magnitudes, routing
    specialization degrades or becomes unstable.

**可检验假设**
    Across at least three balancing regimes (auxiliary loss, capacity-factor gating, bias-only Loss-
    Free) applied to the same MoE architecture, routing purity will differ by more than 15
    percentage points between regimes, and routing assignments will change for more than 20% of
    tokens under bounded input perturbations within cluster boundaries.

**证伪条件**
    The gap is closed if a study shows that routing purity varies by less than 5 percentage points
    across auxiliary-loss, capacity-factor, and bias-only balancing regimes and that fewer than 5%
    of token routings change under bounded input perturbations, demonstrating no meaningful
    balancing-dependent boundary.

**证据（6 条，论文身份已隐藏）**
- [method] 对每个R[i]添加有界随机噪声δi得到˜R[i]，作为Xi中token的代理；δi被限制在相邻聚类中心最小距离的一半以内，以模拟输入变化并防止跨越聚类边界。
- [limitation] 对于某些Switch Transformer模型，观察到训练不稳定性，因此降低了峰值学习率并调整了负载均衡损失权重。
- [method] Loss-Free变体使用Wang等人的仅偏置平衡规则，调整每个专家的路由偏置，同时让router权重不受平衡梯度影响
- [limitation] 对专业化的分析主要集中于不同领域的专家选择频率，未进行更严格的验证。
- [problem] 微批次平衡约束不仅限制了专家专业化，还对模型性能产生负面影响。
- [metric] 通过路由纯度和专家利用率指标直接量化专家专业化程度，提供标准困惑度指标无法捕捉的路由动态可见性

**你的判断**：`[ ] NONE  [ ] PARTIAL  [ ] FULL  [ ] UNCERTAIN`

**理由（必填，一句话）**：

    

---

## 2. `nb-099` — gap

- 层：narrow_mature
- gap_type：mechanism_gap

**描述**
    Compression methods report token savings and quality changes, but the causal driver of the
    reported quality effects (why some compressors preserve or even improve EM while others degrade
    it) is not isolated. In particular, redundancy reduction and fragment selection are confounded
    with the choice of retention signal, and no work isolates which property of the retained context
    actually causes the observed gains.

**目标设置**
    Query-focused context compression for long-document RAG under a fixed memory token budget,
    evaluated on QA benchmarks such as NaturalQuestions and HotpotQA.

**观察到的问题**
    Multiple compressors show divergent quality outcomes at comparable compression ratios: COMI
    reports highest EM at 2x-32x compression versus SnapKV/LongLLMLingua/LLMLingua-2 ([paper]), QADC
    improves F1 from 0.371 to 0.435 over truncation ([paper]) with significant HotpotQA gains
    ([paper]), and compression even raises EM for reasoning-type questions ([paper]). Meanwhile
    token-level compression is reported to break syntax and traceability ([paper]) and rolling
    summarization cuts 60.5% of tokens yet harms performance ([paper]). The cause of these opposite
    outcomes is not established.

**已有覆盖**
    Papers compare compressors empirically under shared budgets and reimplement baselines for fair
    comparison ([paper]), measure redundancy differences between MIG-based and relevance-based
    retention ([paper]), and report AUC superiority of MIG over relevance for identifying answer-
    critical spans ([paper]). They document that adaptive retrieval/compression reduces tokens but
    adds latency ([paper]) and that soft-prompt compression needs ~1.05MB per token of activation
    storage ([paper]).

**缺失能力**
    No controlled ablation that holds the token budget and retained-span count fixed while varying
    only the retention signal (relevance vs. information-gain/MIG vs. random) and measuring whether
    redundancy reduction, span locality, or syntactic preservation is the causal mediator of the
    quality change.

**声称的 delta**
    Isolate the causal mediator of compression-induced quality change by a factorial ablation over
    at least 3 retention signals at 2 fixed retention rates (25% and 50%) on 2 QA datasets,
    attributing at least 50% of the EM/F1 delta to a single identified factor rather than reporting
    aggregate compressor comparisons.

**可检验假设**
    At a fixed 25% retention rate on NaturalQuestions and HotpotQA, swapping only the retention
    signal (MIG vs. relevance vs. random) while holding retained-span count and token budget
    constant will produce an EM difference of more than 3 points, and redundancy alone (measured as
    in prior work) will explain less than 30% of that difference.

**证伪条件**
    The gap is closed if a controlled experiment shows that, with token budget and retained-span
    count fixed, changing the retention signal yields an EM difference of 3 points or less across
    both datasets, or that measured redundancy accounts for more than 70% of the observed quality
    variation, indicating the mechanism is already explained.

**证据（6 条，论文身份已隐藏）**
- [problem] Token级压缩提升了信息密度，但常常破坏语法结构并削弱可读性与来源可追溯性。
- [comparison] 在NaturalQuestions上，MIG的AUC为0.5809，高于relevance的0.5423，表明MIG对识别答案关键片段的区分能力更强。
- [result] 滚动摘要减少60.5%的上下文token，但增加约205%的LLM API调用，使其既昂贵又有害于性能。
- [result] 按MIG保留top片段时，冗余显著低于按relevance保留；在25%保留率下MIG冗余为0.5673、relevance为0.6839，在50%保留率下MIG为0.5956、relevance为0.6664。
- [result] 在NaturalQuestions可回答子集上，QADC相对Truncation将F1从0.371提升到0.435（p=0.069）。
- [result] 在HotpotQA上增益具有统计显著性（p=0.011，Cohen's d=0.258）。

**你的判断**：`[ ] NONE  [ ] PARTIAL  [ ] FULL  [ ] UNCERTAIN`

**理由（必填，一句话）**：

    

---

## 3. `nb-015` — gap

- 层：emerging_sparse
- gap_type：mechanism_gap

**描述**
    Load-balancing constraints are repeatedly reported to suppress expert specialization and even
    hurt performance, and some balancing schemes cause training instability, but the causal pathway
    from the balance objective to altered routing behavior is not established. It remains unknown
    whether balancing acts directly on router logits/biases or indirectly reshapes expert
    representations, and why certain configurations destabilize training.

**目标设置**
    Sparse MoE Transformers trained with auxiliary load-balancing losses (GShard/Switch-style)
    versus bias-only loss-free balancing, measuring routing distributions and specialization before
    and after balancing-weight changes.

**观察到的问题**
    Micro-batch balance constraints are reported to limit specialization and negatively affect
    performance; Switch Transformer training instability required lowering peak learning rate and
    adjusting load-balancing loss weight; Expert Choice achieves perfect balance but leaks future
    tokens; the causal mechanism linking the balance objective to routing behavior is unverified.

**已有覆盖**
    Prior work documents that balancing reduces specialization and can harm performance, describes
    loss-free bias-only balancing rules, and reports instability requiring hyperparameter
    adjustment; specialization analysis has largely relied on expert selection frequencies across
    domains.

**缺失能力**
    A controlled causal decomposition separating the direct effect of the balance gradient on router
    parameters from its indirect effect through expert representation drift, plus an account of why
    particular balance weights induce instability.

**声称的 delta**
    First work to causally decompose the balance-loss effect into router-parameter vs. expert-
    representation channels using matched training runs differing only in balancing weight,
    quantifying each channel's contribution to specialization loss across at least 3 balance-weight
    settings.

**可检验假设**
    If load balancing suppresses specialization primarily by directly reshaping router logits rather
    than by altering expert representations, then freezing router parameters while applying the
    balance gradient to experts only will preserve specialization (routing purity drop < 5%) while
    still achieving expert-load imbalance below 1.5x uniform, compared with a drop of more than 20%
    when the balance gradient reaches the router.

**证伪条件**
    The mechanism claim is falsified if freezing the router under the balance objective produces a
    specialization drop comparable to (within 5 percentage points of) the unfrozen-router condition,
    indicating the effect flows mainly through expert representations rather than router parameters.

**证据（6 条，论文身份已隐藏）**
- [limitation] 对于某些Switch Transformer模型，观察到训练不稳定性，因此降低了峰值学习率并调整了负载均衡损失权重。
- [future_work] ERC损失展现出若干理想特性，为未来专家专业化研究提供了有前景的工具（即专家专业化研究尚未完成）
- [method] Loss-Free变体使用Wang等人的仅偏置平衡规则，调整每个专家的路由偏置，同时让router权重不受平衡梯度影响
- [limitation] 对专业化的分析主要集中于不同领域的专家选择频率，未进行更严格的验证。
- [problem] 微批次平衡约束不仅限制了专家专业化，还对模型性能产生负面影响。
- [negative_result] Expert Choice (EC) 通过为每个专家分配相同数量的 token 确保完美负载均衡，但本质上会导致严重的未来 token 泄漏问题。

**你的判断**：`[ ] NONE  [ ] PARTIAL  [ ] FULL  [ ] UNCERTAIN`

**理由（必填，一句话）**：

    

---

## 4. `nb-035` — idea

- 层：narrow_mature

**标题**：Import working-set / locality-of-reference modeling from operating-systems and d

**描述**
    Import working-set / locality-of-reference modeling from operating-systems and database buffer-
    pool management, where the isomorphic problem is selecting a fixed-size working set from a long
    reference stream to maximize hit rate under a capacity constraint. Map the document to the
    reference stream and retained spans to the cached working set: replace the pure relevance/MIG
    ranking with a locality-aware admission policy (analogous to LRU/ARC admission) that scores each
    candidate span by its margi

**动机**
    Cross-domain transfer: cache-miss / working-set analysis from computer systems: Multiple
    compressors show divergent quality outcomes at comparable compression ratios: COMI reports
    highest EM at 2x-32x compression versus SnapKV/LongLLMLingua/LLMLingua-2 ([paper]), QADC
    improves F1 from 0.371 to 0.435 over truncation ([paper]) with significant HotpotQA gains
    ([paper]), and compression even raises EM for reasoning-type questions ([paper]). Meanwhile
    token-level compression is reported to break syntax and traceability ([paper]) and rolling
    summarization cuts 60.5% of tokens yet harms performance ([paper]). The cause of these opposite
    outcomes is not established.  Supporting evidence (verbatim): -
    "Token级压缩提升了信息密度，但常常破坏语法结构并削弱可读性与来源可追溯性。" -
    "在NaturalQuestions上，MIG的AUC为0.5809，高于relevance的0.5423，表明MIG对识别答案关键片段的区分能力更强。"  This direction
    specifically targets the failure mechanism: Compressors that retain spans by a global relevance
    or MIG score ignore the sequential access pattern of multi-hop QA, so retained spans scatter
    across the document and break the local co-reference chains needed to answer bridge questions.
    This is a locality-of-reference failure: the retention policy optimizes per-span value but
    destroys the temporal contiguity that downstream reasoning depends on, which explains why
    HotpotQA gains diverge from NaturalQuestions gains across compressors..
    [research_direction_only] Deferred by independent scientific triage (priority 6 of 12; the
    experiment budget funds the top 2). Best case: If locality-of-reference is the true driver, a
    locality-aware admission policy should close the HotpotQA-NQ divergence and mediation should
    show locality carrying the majority; a failure to close the gap would strongly exonerate
    locality. Strongest objection: Imports an OS/buffer-pool analogy whose mapping to documents is
    loose; the new admission policy adds machinery beyond a clean test, and improved locality may
    coincide with changed answer coverage. Alternative explanation: HotpotQA gains may stem from
    better multi-hop answer-span co-retention rather than contiguity per se.

**方法草图**
    Import working-set / locality-of-reference modeling from operating-systems and database buffer-
    pool management, where the isomorphic problem is selecting a fixed-size working set from a long
    reference stream to maximize hit rate under a capacity constraint. Map the document to the
    reference stream and retained spans to the cached working set: replace the pure relevance/MIG
    ranking with a locality-aware admission policy (analogous to LRU/ARC admission) that scores each
    candidate span by its marginal contribution to contiguous coverage of already-admitted spans,
    subject to the identical fixed token budget and span count. Compare against pure relevance, MIG,
    and random under the same budget.

**预期贡献**
    Measured average retained-span run-length and locality score per condition, correlated with
    HotpotQA bridge-question F1/EM; test whether the locality-aware policy closes the >3-point EM
    gap and whether locality (not redundancy) carries the majority of the delta in the mediation
    analysis.

**你的判断**：`[ ] NONE  [ ] PARTIAL  [ ] FULL  [ ] UNCERTAIN`

**理由（必填，一句话）**：

    

---

## 5. `nb-001` — idea

- 层：broad_ambiguous

**标题**：Insert a decomposition step before ranking: for each UQ method and task-architec

**描述**
    The ECE-vs-NLL rank disagreement is attributable to component mix: after decomposing each proxy
    into calibration, sharpness, discrimination, and coverage components and conditioning on the
    dominant component, the ECE- and NLL-induced rankings converge (Kendall tau increases
    substantially).

**动机**
    Metric-decomposition diagnostic: Papers rank UQ methods by ECE/NLL as if these scores measured
    the same underlying reliability, yet the same method can win on one metric and lose on another,
    and some metrics are explicitly uninformative for whole families of methods.  Supporting
    evidence (verbatim): - "在ECE校准指标上，CBOpt不如IVON和MC Dropout，后者通过测试时多次随机前向传播获得更好的校准" -
    "实验框架评估点预测精度、概率评分、校准、预测区间覆盖率、区间锐度和计算成本。"  This direction specifically targets the failure
    mechanism: ECE and NLL conflate calibration with sharpness/discrimination, so a UQ method can
    appear reliable under one proxy while its uncertainty is driven by a different component (e.g.,
    low NLL from overconfident correct predictions despite poor coverage); without decomposing the
    scores, rankings reflect component mix rather than reliability..  [research_direction_only]
    Deferred by independent scientific triage (priority 4 of 7; the experiment budget funds the top
    2). Best case: Decomposing proxies into calibration/sharpness/discrimination/coverage and
    showing convergence after conditioning would give a mechanistic account of why rankings diverge,
    turning a descriptive finding into an explanatory one. Strongest objection: Conditioning on the
    dominant component is post hoc and risks circularity — the re-ranking procedure can be adjusted
    until tau rises, so convergence is weak evidence unless pre-specified; also depends on the prior
    disagreement existing. Alternative explanation: Convergence after conditioning may be a
    statistical artifact of reducing rank variance through partialling, not evidence that component
    mix caused the original disagreement.

**方法草图**
    Insert a decomposition step before ranking: for each UQ method and task-architecture cell,
    compute calibration (ECE, reliability diagrams), sharpness (predictive entropy/variance),
    discrimination (AUROC/AUPRC for error detection), and coverage at nominal levels; then regress
    each proxy on these components to show which component dominates its ranking, and re-rank
    methods after conditioning on the dominant component to test whether ECE- and NLL-based orders
    converge once component mix is controlled.

**预期贡献**
    Change in Kendall tau between ECE- and NLL-induced rankings after conditioning on decomposed
    components; variance in proxy scores explained by each component (R^2); reduction in rank
    disagreement attributable to component mix.

**你的判断**：`[ ] NONE  [ ] PARTIAL  [ ] FULL  [ ] UNCERTAIN`

**理由（必填，一句话）**：

    

---

## 6. `nb-091` — idea

- 层：broad_ambiguous

**标题**：Monotonic Perturbation Response Diagnostic for ECE vs NLL Construct Validity

**描述**
    We test whether ECE and NLL share a latent reliability construct by sweeping a graded, accuracy-
    preserving perturbation family over fixed base predictors and checking monotonicity and mutual
    rank agreement of the two metrics' response curves. This differs from the sibling's static
    cross-metric ranking comparison: we manipulate perturbation strength and compare response
    trajectories, using an analytic oracle ladder as ground truth.

**动机**
    Metric-sensitivity stress probe (perturbation-based proxy validity test): Papers rank UQ methods
    by ECE/NLL as if these scores measured the same underlying reliability, yet the same method can
    win on one metric and lose on another, and some metrics are explicitly uninformative for whole
    families of methods.  Supporting evidence (verbatim): - "在ECE校准指标上，CBOpt不如IVON和MC
    Dropout，后者通过测试时多次随机前向传播获得更好的校准" - "实验框架评估点预测精度、概率评分、校准、预测区间覆盖率、区间锐度和计算成本。"  This direction
    specifically targets the failure mechanism: A valid universal reliability proxy should respond
    monotonically to controlled degradations of a model's uncertainty quality. If ECE and NLL react
    differently (or non-monotonically) to the same injected miscalibration/sharpness perturbation,
    their rankings reflect metric-specific artifacts rather than a shared latent reliability
    construct, which is exactly why cross-method rankings can invert..  [Triage note — funded at
    priority 2] Best case: Controlled perturbations that hold accuracy fixed isolate the construct-
    validity question cleanly; monotonicity violations would decisively demonstrate that ECE and NLL
    track different constructs, and clean monotonicity would support interchangeability — both
    outcomes are highly informative. Strongest objection: Synthetic perturbations (tempering, logit
    noise) may not correspond to any real UQ method's behavior, so localized failure regimes might
    be artifacts of unrealistic transformations rather than evidence about actual method rankings.

**方法草图**
    Build a synthetic-perturbation diagnostic: take fixed base predictors and apply a graded family
    of controlled transformations known to alter calibration without changing accuracy (confidence
    tempering, label-preserving logit noise, variance rescaling for regression, per-pixel confidence
    smoothing for segmentation). Sweep the perturbation strength and record how ECE and NLL move. A
    proxy is declared 'construct-valid' only if both metrics are monotone in perturbation strength
    and rank-correlated across the sweep; violations localize the failure regime.

**预期贡献**
    Monotonicity violation rate per metric per task, Spearman correlation between ECE and NLL
    trajectories across perturbation sweeps, and identification of perturbation regimes where the
    two metrics rank methods oppositely.

**你的判断**：`[ ] NONE  [ ] PARTIAL  [ ] FULL  [ ] UNCERTAIN`

**理由（必填，一句话）**：

    

---

## 7. `nb-050` — idea

- 层：broad_ambiguous

**标题**：Arm-Matched Factuality Gate Harness for Multi-Agent Scientific Discovery

**描述**
    A four-arm, two-benchmark diagnostic harness (single-LLM, LLM-RAG, multi-agent gate-off, multi-
    agent gate-on) under matched prompts/budgets/seeds, scoring factuality F=(C_s/C_t)x100 via
    claim-level grounding against retrieved spans and rubric quality anchored to held-out gold
    evidence and execution checks, with paired bootstrap CIs on inter-arm deltas.

**动机**
    Diagnostic benchmark instrument (baseline-anchored factuality and hypothesis-quality scoring
    harness): Proposed multi-agent frameworks explicitly defer empirical validation to future work
    and only expect broader research capability than a single language model; meanwhile other work
    reports that strong subtask performance does not translate into reliable end-to-end experimental
    autonomy, and that assistant systems remain unreliable in open-ended applications.  Supporting
    evidence (verbatim): - "未来的实证研究应实现所提出的架构，并使用标准化基准将其与独立LLM和LLM-RAG系统进行比较，评估假设质量、事实性和实验等方面。" -
    "提出包含证据支撑的知识检索、专用科学智能体、独立假设批评、自动化实验设计、计算实验、结果验证、溯源审计日志与人类在环审批的多智能体架构。"  This direction
    specifically targets the failure mechanism: The field has no shared, baseline-anchored
    measurement of end-to-end scientific-discovery output, so reported multi-agent gains are
    unverifiable and cannot be separated from prompt, retrieval, or aggregation artifacts; without a
    fixed instrument, any claimed >10pp advantage over single-LLM and LLM-RAG baselines is
    confounded and non-reproducible..  Includes 2 mechanism variant(s) folded into the same
    experiment as ablation arms (same hypothesis, different mechanism): Multi-agent pipelines report
    end-to-end capability without any controlled baseli; Reported comparisons of multi-agent vs
    single-LLM vs RAG systems rely on point e  [Triage note — funded at priority 1] Best case: A
    shared, baseline-anchored harness would settle the field's central unverified claim: whether the
    >10pp multi-agent advantage exists at all once prompts, seeds, budgets, and rubrics are matched.
    Either outcome is decisive—confirmation gives the community a reproducible instrument and a
    validated effect size; refutation exposes a widely repeated artifact and redirects effort.
    Because it releases a scoring protocol rather than a new agent, its value compounds across
    future work. Strongest objection: The harness measures everything at once (factuality, quality,
    design validity) across three arms and two benchmarks, so a null result is ambiguous: it could
    reflect a weak pipeline implementation, a lenient/strict rubric, or genuinely absent effects.
    Claim-level grounding for F requires retrieved source spans that the single-LLM arm may not
    have, reintroducing the very information-access confound the study aims to control unless
    budgets are carefully matched.

**方法草图**
    We hypothesize that a multi-agent scientific-discovery pipeline's apparent advantage over
    single-LLM and LLM-RAG baselines is partly an artifact of unmatched retrieval and ungrounded
    intermediate artifacts. We build a diagnostic harness that runs the same benchmark items through
    three arms (single-LLM, LLM-RAG, role-decomposed multi-agent with an inserted factuality gate
    F=(C_s/C_t)x100 on each intermediate artifact) under matched prompts, token budgets, and seeds,
    and additionally ablate the gate on/off. Each hypothesis is scored by claim-level grounding
    against retrieved source spans (factuality F) and by a rubric whose correctness anchor is a
    held-out gold evidence span plus a downstream execution check on a verifiable subset. Expected
    outcome: the gated multi-agent arm shows a larger, reproducible delta over both baselines than
    the ungated arm, isolating whether orchestration or grounding drives any gain.

**预期贡献**
    If successful, this establishes whether multi-agent scientific-discovery gains survive arm-
    matched retrieval and intermediate grounding, separating orchestration effects from retrieval
    and single-model effects — a causal claim the literature currently lacks.

**你的判断**：`[ ] NONE  [ ] PARTIAL  [ ] FULL  [ ] UNCERTAIN`

**理由（必填，一句话）**：

    

---

## 8. `nb-058` — idea

- 层：emerging_sparse

**标题**：Construct a controlled multi-segment drift stream (≥5 segments) parameterized by

**描述**
    There exists a quantifiable regime boundary—expressed as a scalar drift rate λ (Wasserstein
    distance between consecutive segments)—beyond which a statically-tuned VLM TTA method loses more
    than half its static-shift advantage, and this breakpoint λ* differs across methods and predicts
    the first-to-last-segment decay.

**动机**
    boundary_mapping_protocol: Evaluation uses three standard static VLM TTA benchmarks ([paper]);
    the framework assumes noise statistics are known ([paper]) and online adaptation to unknown or
    time-varying noise statistics is left as future work ([paper]); feature-collapse failure modes
    are noted but not tested under drift ([paper]). No supplied evidence measures performance under
    continuously changing distributions.  Supporting evidence (verbatim): - "在三个标准VLM
    TTA基准上评估ORIGIN，使用ViT-B-16视觉编码器" - "论文贡献之一是提出一个新颖的持续测试时学习框架，提升预训练基础模型在下游任务上的性能。"  This direction
    specifically targets the failure mechanism: Static-shift validation selects hyperparameters
    (learning rate, update frequency, buffer size) that implicitly assume a stationary shift
    magnitude; under gradual drift these choices cause the method to either over-adapt to transient
    noise or under-adapt to persistent drift, so its advantage over a frozen model decays
    nonlinearly across segments. The regime boundary is the drift rate (or cumulative distribution
    change) beyond which the static-tuned configuration loses more than half its initial gain..
    [research_direction_only] Deferred by independent scientific triage (priority 5 of 6; the
    experiment budget funds the top 2). Best case: Locating a scalar drift-rate breakpoint lambda*
    per method and showing it predicts first-to-last decay would turn a qualitative worry into a
    quantitative regime boundary, useful for both diagnosis and benchmark design; a well-fit
    piecewise model with CIs is a strong deliverable. Strongest objection: Requires sweeping a
    Wasserstein-parameterized drift grid and fitting breakpoints, which is expensive and sensitive
    to the choice of distance metric and segment construction; lambda* may not transfer to natural
    streams. Alternative explanation: An apparent breakpoint could be an artifact of the piecewise
    fit or of the specific lambda parameterization rather than a real regime transition in the
    method.

**方法草图**
    Construct a controlled multi-segment drift stream (≥5 segments) parameterized by a scalar drift-
    rate λ (e.g., Wasserstein distance between consecutive segment distributions). For each of
    several VLM TTT methods previously validated on static TTA benchmarks, sweep λ over a grid and
    record per-segment accuracy of the adapted model versus a frozen static model. Fit a piecewise
    model to the advantage trajectory to locate the breakpoint λ* where the advantage falls below
    50% of its static-shift value, and report the first-to-last-segment decay as a function of λ.

**预期贡献**
    Estimated λ* with confidence interval; proportion of λ grid where advantage loss exceeds 5pp;
    slope of first-to-last-segment decay versus λ; comparison of λ* across methods.

**你的判断**：`[ ] NONE  [ ] PARTIAL  [ ] FULL  [ ] UNCERTAIN`

**理由（必填，一句话）**：

    

---

## 9. `nb-047` — idea

- 层：broad_ambiguous

**标题**：Invert the assumption 'verification agents improve factuality' by injecting cont

**描述**
    A dedicated multi-agent verification stage detects injected false claims and unsupported
    citations at a true-positive rate more than 10 percentage points higher than single-LLM self-
    check and LLM-RAG citation-grounding check, and this detection advantage translates into higher
    downstream factuality F.

**动机**
    Inverted-verification stress test: Proposed multi-agent frameworks explicitly defer empirical
    validation to future work and only expect broader research capability than a single language
    model; meanwhile other work reports that strong subtask performance does not translate into
    reliable end-to-end experimental autonomy, and that assistant systems remain unreliable in open-
    ended applications.  Supporting evidence (verbatim): - "未来的实证研究应实现所提出的架构，并使用标准化基准将其与独立LLM和LLM-
    RAG系统进行比较，评估假设质量、事实性和实验等方面。" -
    "提出包含证据支撑的知识检索、专用科学智能体、独立假设批评、自动化实验设计、计算实验、结果验证、溯源审计日志与人类在环审批的多智能体架构。"  This direction
    specifically targets the failure mechanism: Multi-agent frameworks assume that adding a
    dedicated verification agent increases end-to-end reliability, but verification agents can
    inherit the same parametric biases as generators and thus rubber-stamp plausible-but-false
    claims; the literature reports strong subtask performance without testing whether the verifier
    actually catches injected errors, so reported end-to-end autonomy may reflect correlated failure
    rather than genuine checking..  [research_direction_only] Deferred by independent scientific
    triage (priority 3 of 5; the experiment budget funds the top 2). Best case: Inverting the
    question to verifier error-detection with injected ground-truth false claims yields a crisp,
    objective signal immune to rubric subjectivity. A high true-positive rate would substantiate the
    verification-stage premise; correlated failure (rubber-stamping) would be a striking, highly
    citable negative result explaining why end-to-end autonomy lags subtask performance. Strongest
    objection: Injected synthetic errors may not resemble naturally occurring generator failures, so
    detection rates on injections may not transfer to real pipeline behavior. The downstream-
    factuality link also depends on the pipeline's base error rate, making the >10pp translation
    claim fragile. Alternative explanation: High detection could reflect superficial lexical
    mismatch between injected claims and sources rather than genuine semantic verification; low
    detection could reflect injection artifacts the verifier was never designed to catch.

**方法草图**
    Invert the assumption 'verification agents improve factuality' by injecting controlled,
    benchmark-grounded false claims and unsupported citations into intermediate agent outputs at
    known rates, then measuring whether the multi-agent verifier detects them relative to a single-
    LLM self-check and an LLM-RAG citation-grounding check; the inverted metric is the verifier's
    error-detection rate rather than the pipeline's raw output quality.

**预期贡献**
    Verifier true-positive rate on injected false claims, false-positive rate on correct claims, and
    downstream factuality F=(C_s/C_t)x100 after verification, compared across multi-agent, single-
    LLM, and LLM-RAG conditions on >=2 benchmarks; a multi-agent advantage is claimed only if
    detection rate exceeds both baselines by >10 percentage points.

**你的判断**：`[ ] NONE  [ ] PARTIAL  [ ] FULL  [ ] UNCERTAIN`

**理由（必填，一句话）**：

    

---

## 10. `nb-103` — idea

- 层：broad_ambiguous

**标题**：Run a factorial boundary-mapping experiment on >=2 standardized scientific-disco

**描述**
    The multi-agent advantage over LLM-RAG and single-LLM is a threshold phenomenon located at
    specific retrieval-context budgets and agent counts; outside the mapped regime the advantage
    vanishes or reverses, so the >10pp effect holds only within an identifiable, reproducible
    boundary.

**动机**
    regime-boundary sweep over retrieval-context budget and agent-count scale: Proposed multi-agent
    frameworks explicitly defer empirical validation to future work and only expect broader research
    capability than a single language model; meanwhile other work reports that strong subtask
    performance does not translate into reliable end-to-end experimental autonomy, and that
    assistant systems remain unreliable in open-ended applications.  Supporting evidence (verbatim):
    - "未来的实证研究应实现所提出的架构，并使用标准化基准将其与独立LLM和LLM-RAG系统进行比较，评估假设质量、事实性和实验等方面。" -
    "提出包含证据支撑的知识检索、专用科学智能体、独立假设批评、自动化实验设计、计算实验、结果验证、溯源审计日志与人类在环审批的多智能体架构。"  This direction
    specifically targets the failure mechanism: The claimed multi-agent advantage over single-LLM
    and LLM-RAG baselines is asserted as an average effect without locating the operating-point
    boundary at which it appears or vanishes; if the advantage is actually a threshold phenomenon in
    retrieval-context budget (RAG) and agent count (multi-agent), then mean-only comparisons
    conflate regimes where multi-agent helps with regimes where added agents merely add compounding-
    error surface, so the reported 'broader capability' cannot be attributed to any identifiable
    regime..  [research_direction_only] Deferred by independent scientific triage (priority 4 of 5;
    the experiment budget funds the top 2). Best case: If the multi-agent advantage is regime-
    dependent, mapping the crossover in retrieval budget and agent count converts a contested
    average into an actionable operating envelope. This would reconcile conflicting literature
    claims and tell practitioners where added agents help versus where they only add error
    surface—valuable regardless of whether a threshold is found. Strongest objection: A full
    factorial sweep across three architectures, multiple budget levels, and agent counts on two
    benchmarks is expensive and risks being underpowered per cell; threshold fitting on noisy,
    sparse cells can manufacture apparent crossovers. It also presupposes a threshold exists, so a
    flat response wastes the budget. Alternative explanation: Apparent crossovers may be artifacts
    of finite sampling or of specific budget granularity; a monotone but saturating relationship
    could be misread as a discrete threshold.

**方法草图**
    Run a factorial boundary-mapping experiment on >=2 standardized scientific-discovery benchmarks:
    cross {single-LLM, LLM-RAG, multi-agent} with a swept retrieval-context budget (e.g. k retrieved
    passages / token budget across several levels) and a swept agent count (1..N agents in the
    pipeline), holding the underlying base model fixed. For each cell compute factuality
    F=(C_s/C_t)x100 and expert-rated hypothesis quality, then fit piecewise/threshold models to
    locate crossover points where multi-agent overtakes RAG and where it stops improving or
    degrades.

**预期贡献**
    Estimated crossover retrieval-budget value and agent-count saturation point per benchmark, with
    confidence intervals; the >10pp multi-agent advantage is confirmed only inside the mapped regime
    and its location is reported as a reproducible boundary rather than an average.

**你的判断**：`[ ] NONE  [ ] PARTIAL  [ ] FULL  [ ] UNCERTAIN`

**理由（必填，一句话）**：

    

---

## 11. `nb-003` — idea

- 层：emerging_sparse

**标题**：Frequency-Matched Permutation Calibration of Expert-Category Concentration in Multilingual MoE Routing

**描述**
    This experiment tests whether frequency-matched permutation calibration reveals genuine
    category-conditioned expert routing in a multilingual MoE model, using language identity, POS,
    and semantic class labels. We compare observed expert-category concentration to a null
    distribution generated by permuting category labels within frequency bins, and validate against
    a synthetic no-specialization control.

**动机**
    Frequency-matched null calibration layer for expert-category concentration: One paper explicitly
    limits its specialization analysis to expert selection frequencies across domains without
    stricter validation ([paper]); another offers ERC loss only as a promising tool for future
    specialization study ([paper]); a vision-focused work excludes text/audio modalities entirely
    ([paper]); and a multilingual study defers extending specialization-pattern analysis to multiple
    languages to future work ([paper]).  Supporting evidence (verbatim): -
    "ERC损失展现出若干理想特性，为未来专家专业化研究提供了有前景的工具（即专家专业化研究尚未完成）" -
    "由于无法跨领域配对样本，改为在语料序列上平均激活比例，再计算任务/语言数据集与基线之间的差值Δ"  This direction specifically targets the
    failure mechanism: Specialization claims rest on raw expert selection frequencies per
    domain/category, which conflate genuine routing preference with the marginal token-frequency
    distribution of each category; without a frequency-matched chance baseline, any category
    imbalance in the corpus masquerades as expert specialization, so the reported concentration
    cannot distinguish signal from base-rate artifact..  Includes 4 mechanism variant(s) folded into
    the same experiment as ablation arms (same hypothesis, different mechanism): Existing
    specialization analyses report raw expert-selection frequencies per dom; Raw expert-selection
    frequencies across categories are confounded by the margina; Existing specialization analyses
    (e.g. [paper]'s POS confusion-matrix accuracy,; Expert-category concentration is measured
    against raw frequency or uncalibrated   [Complementary experiment] Same factor (expert-category
    concentration statistic (raw normalized deviation of observed routing mass from expected mass)),
    different manipulation (compare_observed_statistic_to_row_column_preserving_surrogate_band vs
    permute_category_labels_within_frequency_bins) — tests a different causal question on this
    factor and is preserved as a separate experiment under this idea.

**方法草图**
    We hypothesize that apparent expert specialization in MoE language models is largely an artifact
    of token-frequency skew, and that a frequency-matched permutation null will reveal whether
    routing carries category-specific information beyond unigram base rates. We label tokens in a
    multilingual corpus with language identity, POS, and semantic class, record expert assignments
    from an open MoE model, and compute per-expert-per-category concentration statistics. By
    permuting category labels within frequency bins while preserving expert load and category
    marginals, we obtain a calibrated null distribution and z-scores for each expert-category pair.
    We expect a minority of experts to exceed a +2 SD threshold, establishing that some routing
    decisions are genuinely category-conditioned rather than frequency-driven.

**预期贡献**
    If successful, this experiment establishes that frequency-matched permutation calibration
    separates genuine category-conditioned routing from base-rate artifacts in MoE language models,
    providing a statistically grounded criterion for when specialization claims are warranted. It
    further quantifies how much of previously reported specialization survives this calibration
    across language, POS, and semantic categories.

**你的判断**：`[ ] NONE  [ ] PARTIAL  [ ] FULL  [ ] UNCERTAIN`

**理由（必填，一句话）**：

    

---

## 12. `nb-023` — idea

- 层：emerging_sparse

**标题**：Scale-Ladder Change-Point Detection for Null-Calibrated Expert-Category Concentration

**描述**
    Probe a ladder of public MoE checkpoints (inference only) with a frequency-matched permutation
    null for expert-category concentration, then fit a change-point model to the fraction of experts
    exceeding +2 SD versus log parameter count and expert count. The experiment tests whether null-
    calibrated specialization is absent at small scales and emerges past an estimable boundary,
    using gold human-annotated labels for part-of-speech and language identity and a rule-based
    lexical oracle for semantic class, with a synthetic frequency-only-router control fixing the
    nominal false-positive rate at each rung.

**动机**
    Scale-boundary sweep of null-calibrated specialization prevalence: One paper explicitly limits
    its specialization analysis to expert selection frequencies across domains without stricter
    validation ([paper]); another offers ERC loss only as a promising tool for future specialization
    study ([paper]); a vision-focused work excludes text/audio modalities entirely ([paper]); and a
    multilingual study defers extending specialization-pattern analysis to multiple languages to
    future work ([paper]).  Supporting evidence (verbatim): -
    "ERC损失展现出若干理想特性，为未来专家专业化研究提供了有前景的工具（即专家专业化研究尚未完成）" -
    "由于无法跨领域配对样本，改为在语料序列上平均激活比例，再计算任务/语言数据集与基线之间的差值Δ"  This direction specifically targets the
    failure mechanism: Specialization claims are made at a single model scale without checking
    whether the phenomenon emerges only past a parameter-count or expert-count boundary, so reported
    prevalence may be an artifact of the chosen operating point..

**方法草图**
    Evaluate the same frequency-matched null-calibrated concentration test across a ladder of
    publicly available MoE checkpoints spanning parameter counts and expert counts (inference only),
    and fit a change-point model to the fraction of experts exceeding +2 SD versus scale to locate
    the emergence boundary.

**预期贡献**
    Change-point estimate for log parameter count (or expert count) with confidence interval; slope
    of significant-expert fraction versus scale before and after the boundary; replication of the
    >15% prevalence claim only above the estimated boundary.

**你的判断**：`[ ] NONE  [ ] PARTIAL  [ ] FULL  [ ] UNCERTAIN`

**理由（必填，一句话）**：

    

---

## 13. `nb-033` — idea

- 层：emerging_sparse

**标题**：Noise-Statistic Staleness Probe: Oracle-Known vs Estimated Sigma Under Time-Varying Corruption Drift

**描述**
    A factorial ablation crossing oracle-known vs estimated noise statistics with stationary vs
    time-varying corruption schedules inside a 5-segment drift stream, using CLIP ResNet-50 and a
    BayesTTA-style estimator whose sigma is either supplied from the synthesizer's recorded ground
    truth or estimated online. Per-segment accuracy advantage over frozen CLIP and a feature-
    collapse indicator are measured; the primary metric is the absolute percentage point difference
    in accuracy advantage between oracle and estimated arms at the final segment, and the staleness-
    sensitivity probe quantifies how much each arm's accuracy depends on sigma mismatch.

**动机**
    Reconciliation ablation isolating the known-noise-statistics assumption: Evaluation uses three
    standard static VLM TTA benchmarks ([paper]); the framework assumes noise statistics are known
    ([paper]) and online adaptation to unknown or time-varying noise statistics is left as future
    work ([paper]); feature-collapse failure modes are noted but not tested under drift ([paper]).
    No supplied evidence measures performance under continuously changing distributions.  Supporting
    evidence (verbatim): - "在三个标准VLM TTA基准上评估ORIGIN，使用ViT-B-16视觉编码器" -
    "论文贡献之一是提出一个新颖的持续测试时学习框架，提升预训练基础模型在下游任务上的性能。"  This direction specifically targets the failure
    mechanism: Methods assume noise statistics are known and stationary; when drift makes the true
    noise statistics time-varying, the estimator used for adaptation becomes biased, so a quantity
    that improves early (adaptation fit to current segment) degrades later (accumulated mis-
    estimation and feature collapse), producing the observed advantage decay without any change in
    the underlying method..  [research_direction_only] The experiment plan for this direction was
    drafted but rejected by the consistency gate after all retries. Blocking reasons:
    LLM_ONLY_ORACLE: the oracle relies on an LLM's own judgement (oracle = "Execution-grounded
    oracle: the ground-truth Gaussian-noise sigma used by the image synthesizer for each segment is
    recor") — ground it in execution outcomes, hidden test cases, formal verification or human
    labels instead. The full draft plan is preserved in the idea_quality_gate trace
    (rejected_plan_full).

**方法草图**
    Run a factorial ablation crossing (i) oracle-known vs estimated noise statistics with (ii)
    stationary vs time-varying noise schedules inside the drift stream, measuring per-segment
    accuracy and a feature-collapse indicator (e.g., representation variance/rank) for each cell.
    This tests whether the static-vs-drift advantage loss is attributable to the violated known-
    statistics assumption rather than to drift per se.

**预期贡献**
    Difference in final-segment advantage between oracle-statistics and estimated-statistics cells
    (in pp), and correlation between feature-collapse indicator trajectory and per-segment accuracy
    decay; reconciliation holds if oracle statistics recover most of the lost advantage.

**你的判断**：`[ ] NONE  [ ] PARTIAL  [ ] FULL  [ ] UNCERTAIN`

**理由（必填，一句话）**：

    

---

## 14. `nb-039` — gap

- 层：broad_ambiguous
- gap_type：boundary_gap

**描述**
    Conformal prediction is documented to lose its distribution-free coverage guarantee when
    exchangeability is violated (dynamic environments, real-world distribution shift), and multiple
    baseline CP variants show coverage degradation as shift intensity increases. Yet the same
    literature evaluates CP almost exclusively on static image-classification benchmarks with a
    handful of CNN backbones and default train-test splits. The boundary condition is therefore the
    transition from exchangeable to non-exchangeable data streams combined with task diversity: no
    supplied evidence tests whether the coverage guarantee holds jointly across classification,
    regression and detection under evolving shift, nor how quickly coverage decays per task type.

**目标设置**
    Conformal-prediction UQ applied to deep networks across classification, regression and detection
    tasks under non-exchangeable / progressively shifting distributions, rather than static
    exchangeable benchmark splits.

**观察到的问题**
    Naive CP application in dynamic environments yields coverage that degrades over time ([paper]),
    and chi-square, FG-CP and Weight baselines all show coverage degradation as shift intensity
    grows ([paper]); UGNN also fails to reach valid coverage on Trade under semi-inductive settings
    ([paper]). Meanwhile CP evaluations use only VGG16/ResNet50 backbones ([paper]), four CNN
    backbones ([paper]), and default CIFAR-10 splits ([paper]), and the impossibility of
    simultaneously achieving distribution-free, finite-sample-valid and conditional coverage is
    acknowledged ([paper]).

**已有覆盖**
    CP theory and practice are well covered for exchangeable static settings: split vs full
    conformal efficiency trade-offs ([paper]), flexible model wrapping via make_model ([paper]),
    coverage-oriented metrics such as Cov@1% ([paper]), and evaluations on MNIST/ImageNet/iWildCam
    ([paper]) with ResNet50/InceptionV3/DenseNet161/MobileNetV2 ([paper]). Degradation under shift
    is noted qualitatively ([paper], [paper]).

**缺失能力**
    A per-task-type characterization of how far empirical coverage falls below nominal coverage as a
    function of shift intensity, jointly across classification, regression and detection, which the
    current single-task static-split evaluations never measure.

**声称的 delta**
    Existing CP work documents coverage decay only within static image-classification pipelines
    using <=4 CNN backbones and default splits; it never establishes WHERE the distribution-free
    guarantee breaks down when moving from exchangeable static splits to progressively shifting
    streams across >=3 task types (classification, regression, detection).

**可检验假设**
    Under a controlled shift-intensity sweep, empirical CP coverage will fall more than 5 percentage
    points below the 90% nominal level for at least 2 of 3 task types (classification, regression,
    detection) once shift intensity exceeds a measurable threshold, whereas on exchangeable static
    splits the same pipeline stays within 2 percentage points of nominal.

**证伪条件**
    The gap is closed if a CP procedure evaluated across all three task types maintains empirical
    coverage within 2 percentage points of the 90% nominal level at every shift-intensity level
    tested, showing the documented decay does not generalize beyond the reported image-
    classification baselines.

**证据（6 条，论文身份已隐藏）**
- [comparison] 书中对比了分裂共形与完整共形在统计效率与计算效率上的差异
- [limitation] 已有工作表明无法同时获得无分布假设、有限样本有效和条件覆盖这三个条件。
- [negative_result] 在Trade数据集上，UGNN在半归纳设置下未能达到有效覆盖率
- [dataset] 实验使用了ResNet50、InceptionV3、DenseNet161和MobileNetV2四种深度学习模型进行共形预测集大小评估
- [dataset] CIFAR-10实验使用默认的训练-测试划分
- [negative_result] 在可交换性被违反的动态环境中，朴素地应用共形预测会导致覆盖性能随时间退化。

**你的判断**：`[ ] NONE  [ ] PARTIAL  [ ] FULL  [ ] UNCERTAIN`

**理由（必填，一句话）**：

    

---

## 15. `nb-022` — gap

- 层：broad_ambiguous
- gap_type：cross_domain_transfer

**描述**
    In the probabilistic-forecasting / econometrics literature, proper scoring rules and reliability
    diagrams are decomposed into separate calibration and sharpness components, and calibration is
    assessed conditionally (per-region, per-horizon) rather than via a single pooled ECE. The deep-
    UQ target setting evaluates calibration mostly with a single pooled ECE/NLL over 20 bins and
    lacks a systematic framework separating uncertainty estimation from calibration and
    probabilistic prediction; comparisons across methods are hard because datasets, metrics and
    pipelines differ. The mapping is: source = decomposition of proper scoring rules into
    calibration + sharpness (and conditional reliability assessment) from forecasting/statistics;
    target = deep-network UQ evaluation where pooled ECE conflates miscalibration with sharpness and
    hides conditional failures (e.g., EMM under-confident in-distribution smECE=0.193 yet
    overconfident OOD). This decomposition has not been imported into the general deep-UQ benchmark.

**目标设置**
    Cross-task, cross-domain deep-network UQ evaluation (classification, regression, detection)
    needing a systematic framework that separates uncertainty estimation from calibration and
    probabilistic prediction.

**观察到的问题**
    Calibration is measured with a single pooled ECE over 20 bins; methods show contradictory
    behavior (EMM under-confident in-distribution but overconfident OOD; pre-trained/ABNN/Whitebox
    retain <=15% confidence drop), and comparisons are difficult because datasets, architectures,
    training pipelines and metrics differ across studies.

**已有覆盖**
    Papers note the need for systematic evaluation frameworks distinguishing uncertainty estimation
    from calibration and probabilistic prediction, use ECE with 20 variance-minimizing bins, and
    report Cov@1% for selective prediction; they do not adopt a calibration/sharpness decomposition
    or conditional reliability analysis.

**缺失能力**
    A scoring-rule decomposition (calibration term + sharpness term, plus conditional/per-group
    reliability) applied uniformly across tasks and domains in the deep-UQ benchmark, replacing the
    single pooled ECE.

**声称的 delta**
    Current deep-UQ evaluations report one pooled ECE over 20 bins; the transfer would decompose the
    score into calibration and sharpness components and evaluate conditional reliability across at
    least 3 task types and 2 domains, revealing miscalibration that pooled ECE masks by at least a
    factor of 2 in the worst group, which no cited work reports.

**可检验假设**
    Decomposing the proper scoring rule into calibration and sharpness terms reveals at least one
    subgroup (domain or task) whose conditional calibration error exceeds its pooled ECE by a factor
    of 2 or more, for at least 2 of 3 evaluated UQ methods on a shared multi-task benchmark.

**证伪条件**
    The opportunity is closed if a cited or newly found deep-UQ benchmark already reports a
    calibration/sharpness decomposition with conditional per-group reliability across at least 3
    task types, or if conditional calibration error is shown to never exceed pooled ECE by more than
    a factor of 2 for any tested method.

**证据（6 条，论文身份已隐藏）**
- [comparison] EMM位于对角线上方，在ID数据上信心不足（smECE=0.193），但在OOD输入上仍然过度自信。
- [comparison] 预训练模型、ABNN和Whitebox在ID与偏移输入之间的预测置信度几乎没有下降（≤15%），在OOD上仍然过度自信。
- [metric] 使用期望校准误差(ECE)衡量MC-Dropout不确定性估计与模型误差的校准程度，并比较不同Dropout率与层放置配置（编码器稀释、解码器稀释、全局稀释）
- [limitation] 由于研究常采用不同的数据集、架构、训练流程、不确定性度量和校准指标，这些不确定性方法之间的比较十分困难
- [future_work] 现有工作强调需要系统性评估框架，以区分不确定性估计与校准及概率预测
- [metric] 校准测量使用20个分箱，分箱划分以最小化箱内方差

**你的判断**：`[ ] NONE  [ ] PARTIAL  [ ] FULL  [ ] UNCERTAIN`

**理由（必填，一句话）**：

    

---

## 16. `nb-016` — idea

- 层：broad_ambiguous

**标题**：Invert 'uncertainty = spread across members/models' into 'uncertainty = divergen

**描述**
    Residual overconfidence of Bayesian and deep-ensemble UQ on shifted inputs can be reduced by a
    self-consistency or disagreement-gated correction that compares stated uncertainty against
    locally estimated error, lowering OOD smECE and the ID-vs-OOD confidence gap without regressing
    ID calibration.

**动机**
    Assumption inversion: measure disagreement-with-self instead of agreement-with-others: Naive
    conformal prediction loses coverage over time when exchangeability is violated ([paper]); non-
    intrusive methods (Whitebox, EMM) degrade under OOD/adversarial conditions ([paper]); pre-
    trained models, ABNN and Whitebox show <=15% confidence drop between ID and shifted inputs and
    remain overconfident on OOD ([paper]); EMM is underconfident on ID (smECE=0.193) yet
    overconfident on OOD ([paper]); standard ensembles produce overconfident predictions in unseen
    scenarios because members behave similarly ([paper]).  Supporting evidence (verbatim): -
    "EMM位于对角线上方，在ID数据上信心不足（smECE=0.193），但在OOD输入上仍然过度自信。" -
    "标准集成方法（即用不同随机初始化独立训练的模型）在未见过的场景中可能产生过度自信的预测，因为集成中的模型往往表现出相似的行为。"  This direction specifically
    targets the failure mechanism: Ensemble and Bayesian UQ assume that member agreement is
    informative about uncertainty, but members trained on the same exchangeable data share
    correlated errors, so they agree confidently on OOD inputs ([paper]) and the resulting variance
    underestimates true error. Conformal methods separately assume the score distribution is stable;
    neither family checks whether its own uncertainty estimate is internally consistent across the
    shift, so overconfidence on OOD ([paper], [paper]) goes undetected until coverage fails..
    [research_direction_only] Deferred by independent scientific triage (priority 4 of 4; the
    experiment budget funds the top 2). Best case: If within-model self-consistency residual
    predicts subsequent coverage loss and gating on it reduces OOD overconfidence without regressing
    ID calibration, it delivers a genuinely new uncertainty signal that survives correlated-member
    failure — a substantive mechanistic advance over spread-based UQ. Even partial success (residual
    correlates with coverage loss but correction is weak) is informative about where the
    overconfidence originates. Strongest objection: The correction needs 'a small recent labeled
    buffer' to estimate local error scale, reintroducing the label-dependence problem and making the
    method hard to distinguish from ordinary post-hoc recalibration; the self-consistency residual
    may simply be a restatement of the calibration error it purports to predict, yielding circular
    validation. It also stacks multiple components (buffer estimation, gating, inflation) beyond the
    minimal test of the hypothesis. Alternative explanation: Reduced OOD smECE could arise from
    interval inflation per se rather than from the self-consistency signal being informative; a
    random or constant-inflation control matched for average width would reveal whether the specific
    residual carries predictive value.

**方法草图**
    Invert 'uncertainty = spread across members/models' into 'uncertainty = divergence between a
    model's predicted uncertainty and the uncertainty implied by its own held-out residuals under
    the current input regime'. For each family, compute a self-consistency residual: the gap between
    the stated interval/credible width and the locally estimated error scale from a small recent
    labeled buffer; flag and inflate intervals when this residual is positive (overconfident). This
    uses within-model self-disagreement rather than cross-member agreement, which is exactly the
    quantity that survives correlated-member failure.

**预期贡献**
    Residual overconfidence (mean stated-confidence minus empirical accuracy) on OOD inputs reduced
    relative to uncorrected baselines, and smECE reported separately for ID vs OOD per family;
    quantify the correlation between self-consistency residual and subsequent coverage loss over the
    100-step horizon.

**你的判断**：`[ ] NONE  [ ] PARTIAL  [ ] FULL  [ ] UNCERTAIN`

**理由（必填，一句话）**：

    

---

## 17. `nb-036` — idea

- 层：broad_ambiguous

**标题**：On a shared multi-task benchmark (>=3 task types, >=2 domains) with fixed archit

**描述**
    The divergence between worst-group conditional calibration error and pooled ECE is governed by a
    controllable regime variable (subgroup support size / bin occupancy, or shift magnitude /
    frequency band along a graded perturbation ladder), such that there exists a detectable onset
    threshold or breakdown band where worst-group calibration error crosses 2x pooled ECE for >= 2
    of 3 methods.

**动机**
    Regime-boundary mapping of pooled-vs-worst-group calibration divergence: Calibration is measured
    with a single pooled ECE over 20 bins; methods show contradictory behavior (EMM under-confident
    in-distribution but overconfident OOD; pre-trained/ABNN/Whitebox retain <=15% confidence drop),
    and comparisons are difficult because datasets, architectures, training pipelines and metrics
    differ across studies.  Supporting evidence (verbatim): -
    "EMM位于对角线上方，在ID数据上信心不足（smECE=0.193），但在OOD输入上仍然过度自信。" -
    "预训练模型、ABNN和Whitebox在ID与偏移输入之间的预测置信度几乎没有下降（≤15%），在OOD上仍然过度自信。"  This direction specifically
    targets the failure mechanism: Pooled ECE averages over subgroups and bins, so miscalibration
    concentrated in low-density regions (rare domains, tail classes, small object scales) is diluted
    below the reporting threshold; the divergence between pooled ECE and worst-group conditional
    calibration is not constant but switches on past a boundary in subgroup support size / bin
    occupancy, so studies using a single pooled number cannot see where the failure begins..
    [research_direction_only] Deferred by independent scientific triage (priority 4 of 6; the
    experiment budget funds the top 2). Best case: Locating an onset threshold in support size / bin
    occupancy would convert the >=2x claim from a fragile point estimate into a characterized
    regime, and a clean monotonic divergence curve would be strong mechanistic evidence that
    dilution drives the pooled-vs-worst-group gap. Strongest objection: Sweeping support size via
    controlled subsampling changes the very quantity being measured and interacts with ECE's own
    sample-size bias, so any located 'threshold' may be an artifact of the estimator's variance
    rather than a property of the underlying miscalibration. Alternative explanation: An apparent
    onset boundary could simply track where binned ECE becomes statistically unstable at low
    occupancy, i.e. a property of the metric's estimator variance, not of the model's conditional
    reliability.

**方法草图**
    On a shared multi-task benchmark (>=3 task types, >=2 domains) with fixed architectures and
    training pipelines, sweep the regime variable that plausibly governs the divergence: per-
    subgroup effective sample size (via controlled subsampling of domain/task strata) and bin
    occupancy. For each operating point compute the proper-scoring-rule decomposition (calibration
    term vs sharpness term) plus per-group reliability, and locate the support-size/bin-occupancy
    threshold at which worst-group calibration error crosses 2x pooled ECE, separately for each UQ
    method.

**预期贡献**
    Estimated onset threshold (support size and bin occupancy) at which worst-group calibration
    error >= 2x pooled ECE, reported with bootstrap CIs; fraction of methods showing a detectable
    onset boundary; monotonicity check of divergence ratio across the swept regime.

**你的判断**：`[ ] NONE  [ ] PARTIAL  [ ] FULL  [ ] UNCERTAIN`

**理由（必填，一句话）**：

    

---

## 18. `nb-064` — idea

- 层：emerging_sparse

**标题**：Capacity-Matched Routing Permutation Decomposition of Expert Specialization

**描述**
    A randomized-controlled-trial design for MoE routing: on identical evaluation batches, compare
    native top-k routing against a capacity-matched random-routing arm that preserves the exact per-
    expert load, and against a uniform-random arm. Decompose the change in domain-level
    specialization (JS divergence between domain-conditional expert usage distributions, domain-
    expert mutual information) into routing-attributable and capacity-attributable fractions, with
    bootstrap CIs. Separately test whether router statistics (gate logit margin, entropy) predict
    the sign of the specialization change above chance. Run across three sparse-gated MoE families.

**动机**
    Causal attribution via counterfactual routing ablation with matched controls (imported from
    experimental design / clinical trial methodology): Papers report specialization via expert-
    selection frequency across domains and acknowledge this is not strict validation; they also find
    no local router statistic predicts the sign of flip loss above random, leaving the causal link
    between routing and specialization unmeasured.  Supporting evidence (verbatim): -
    "预注册的探针在三种架构上验证发现。" -
    "Engmann等人在OLMoE、Qwen1.5-MoE和DeepSeek上审计群体路由摘要是否能预测专家消融的因果重要性，发现观察指标不足；本文改为在配对干净/量化计算下干预路由中介。"
    This direction specifically targets the failure mechanism: Router statistics are correlational
    and confounded by expert capacity limits: token-to-expert assignment frequency mixes the effect
    of the routing decision with the effect of the fixed per-expert capacity buffer, so no local
    router statistic can isolate whether routing causes specialization or merely reflects it.
    Without a randomized/matched control arm, the sign of any flip-loss effect is unidentified..
    Includes 3 mechanism variant(s) folded into the same experiment as ablation arms (same
    hypothesis, different mechanism): Router statistics are correlational and confounded with expert
    capacity/buffer d; Current work assumes routing decisions are the primary causal driver of
    expert s; Specialization is attributed to routing decisions by convention, but expert capa
    [Triage note — funded at priority 1] Best case: A clean capacity-matched permutation control
    yields a routing-attributable fraction of specialization with bootstrap CIs, directly
    quantifying the causal link the gap says is unmeasured; either a large or a near-zero routing
    share is decisive and interpretable. Strongest objection: The capacity-matched permutation
    control may itself alter effective computation in ways that are not purely 'routing', so the
    treatment-minus-control contrast could absorb second-order effects; the 10% relative threshold
    is arbitrary and untested for sensitivity.  [Complementary experiment] Same factor (token-to-
    expert routing assignment identity), different manipulation
    (forced_topk_swap_with_capacity_cross vs capacity_matched_permutation) — tests a different
    causal question on this factor and is preserved as a separate experiment under this idea.

**方法草图**
    Borrow the randomized-controlled-trial / matched-control design from biostatistics and apply it
    to routing: for each evaluation batch, construct a treatment arm (native top-k routing) and a
    matched random-routing control arm that preserves the exact same per-expert load distribution
    (capacity-matched permutation of assignments). Compute domain specialization metrics (expert-
    selection entropy, domain-expert mutual information) under both arms and decompose the
    difference into a routing-attributable component (treatment minus capacity-matched control)
    versus a capacity-attributable component (control minus uniform-random). Report the fraction of
    specialization attributable to routing as a causal-effect size with bootstrap confidence
    intervals.

**预期贡献**
    Fraction-of-specialization-attributable-to-routing point estimate with 95% CI per architecture;
    hypothesis confirmed if the routing-attributable effect exceeds 10% relative change versus the
    matched random-routing control in at least 2 of 3 MoE families.

**你的判断**：`[ ] NONE  [ ] PARTIAL  [ ] FULL  [ ] UNCERTAIN`

**理由（必填，一句话）**：

    

---

## 19. `nb-014` — idea

- 层：broad_ambiguous

**标题**：Build a pre-registration ledger that timestamps and hashes each AI-generated hyp

**描述**
    The model-over-human novelty advantage is a monotonic function of ground-truth exposure level in
    the judge context, with a locatable crossover exposure level below which the advantage falls
    below the >50% shrinkage threshold.

**动机**
    boundary-mapping sweep over ground-truth exposure timing: Novelty and validity of LLM-generated
    hypotheses are assessed by LLM judges that amplify novelty hallucination and favor model
    outputs, while anchor-overlap metrics cannot distinguish memorization from genuine discovery
    Supporting evidence (verbatim): - "在比较设置下，LLM评委放大了新颖性幻觉，严重偏向模型输出而非作者撰写的基线" -
    "建议建立集中式预注册仓库，收录所有AI生成的假设。"  This direction specifically targets the failure mechanism:
    Retrospective LLM judges rate hypotheses after the judge has already seen ground-truth answers
    and baseline outputs, so the model-vs-baseline novelty advantage is a function of how much
    ground-truth leakage enters the judge context; the location of the crossover point (where the
    advantage vanishes) is unknown, making the 'novelty mirage' look like a uniform bias rather than
    a regime-dependent artifact..  [research_direction_only] Deferred by independent scientific
    triage (priority 4 of 6; the experiment budget funds the top 2). Best case: Maps the dose-
    response of ground-truth exposure to the novelty advantage, yielding a crossover boundary that
    converts a binary bias claim into a quantitative regime characterization; informative regardless
    of where the crossover lands. Strongest objection: Defining discrete exposure levels
    (blinded/partial/exposed) is operationally fuzzy and confounded with prompt length and context
    richness; the sweep multiplies conditions and risks underpowered cells. Alternative explanation:
    A monotonic curve could reflect general context-priming effects on judge ratings rather than
    ground-truth-specific hindsight leakage.

**方法草图**
    Build a pre-registration ledger that timestamps and hashes each AI-generated hypothesis before
    any ground-truth exposure, then run a controlled sweep along the single axis of ground-truth
    exposure level in the judge context (fully blinded / partial-anchor / fully exposed), holding
    the hypothesis set fixed across the 40 RCBench tasks. Score the identical frozen hypothesis set
    at each exposure level both prospectively (ledger-locked, blinded) and retrospectively (post-
    exposure LLM judge), mapping the curve of model-minus-baseline novelty advantage versus exposure
    level to locate the crossover boundary.

**预期贡献**
    A plotted advantage-vs-exposure curve over >=100 hypotheses x 40 tasks with an estimated
    crossover exposure level and confidence interval; report whether prospective (blinded) scoring
    sits below the >50% shrinkage threshold relative to retrospective scoring at matched exposure.

**你的判断**：`[ ] NONE  [ ] PARTIAL  [ ] FULL  [ ] UNCERTAIN`

**理由（必填，一句话）**：

    

---

## 20. `nb-038` — idea

- 层：emerging_sparse

**标题**：Log-Scale Update-Magnitude Sweep for Over-Sharpening Threshold in LoRA Test-Time Adaptation

**描述**
    This experiment isolates the effect of update magnitude (not iteration count) on test-time LoRA
    adaptation quality. By fixing iterations at 10 and sweeping a log-spaced magnitude multiplier,
    we locate the threshold beyond which adaptation quality reverses, distinguishing magnitude-
    induced over-sharpening from iteration-count effects.

**动机**
    Update-magnitude boundary sweep: The field assumes test-time parameter updates improve
    adaptation, but evidence shows high iteration counts are infeasible ([paper]), prompt tuning can
    catastrophically degrade output ([paper]), and online updates are unstable ([paper]). Whether
    additional update iterations actually yield monotone gains, or saturate/reverse, is never
    directly measured.  Supporting evidence (verbatim): -
    "QueST通过生成辅助问题–解答对直接从输入查询构建监督，并在测试时通过LoRA进行高效参数自适应。" - "提出测试时最近邻训练（TTT-
    NN）：对每个测试实例，从大型数据库中检索其最近邻，并在应用模型前在这些邻居上微调模型。"  This direction specifically targets the failure
    mechanism: Even with a fixed number of iterations, the magnitude of parameter updates
    (controlled by learning rate or norm constraint) may cause over-sharpening or catastrophic
    degradation beyond a threshold; the safe operating range is unknown..  [Triage note — funded at
    priority 2] Best case: Sweeping update magnitude at fixed iteration count isolates the magnitude
    dimension and can reveal a critical threshold; a null result (no reversal within a plausible LR
    grid) would be equally informative about whether magnitude alone drives degradation. Strongest
    objection: It fixes iteration count at an arbitrary moderate value (e.g., 10), so it may miss
    the interaction between magnitude and iteration count that the gap's phenomenon centrally
    concerns; the chosen anchor could predetermine the answer.

**方法草图**
    Sweep update magnitude (e.g., learning rate or L2 norm of parameter change) over a logarithmic
    grid while keeping iteration count fixed at a moderate value (e.g., 10). Measure adaptation
    quality to identify the magnitude threshold where performance reverses.

**预期贡献**
    Critical magnitude threshold (e.g., learning rate) where quality drops >5% from its peak,
    reported with confidence intervals; also report interaction with iteration count if both vary.

**你的判断**：`[ ] NONE  [ ] PARTIAL  [ ] FULL  [ ] UNCERTAIN`

**理由（必填，一句话）**：

    

---

## 21. `nb-076` — idea

- 层：broad_ambiguous

**标题**：Using the pre-registration ledger to freeze hypotheses, sweep two resource bound

**描述**
    The model-over-human novelty advantage converges to the prospective estimate once judge-panel
    size and anchor-set density exceed identifiable resource thresholds; below those thresholds the
    advantage is inflated by under-resourced judging.

**动机**
    budget-boundary scaling of judge panel size and anchor density: Novelty and validity of LLM-
    generated hypotheses are assessed by LLM judges that amplify novelty hallucination and favor
    model outputs, while anchor-overlap metrics cannot distinguish memorization from genuine
    discovery  Supporting evidence (verbatim): - "在比较设置下，LLM评委放大了新颖性幻觉，严重偏向模型输出而非作者撰写的基线" -
    "建议建立集中式预注册仓库，收录所有AI生成的假设。"  This direction specifically targets the failure mechanism: The
    model-favoring novelty advantage may be governed by judge-panel aggregation budget and anchor-
    set density: with few judges or sparse anchors, self-preference noise dominates and inflates
    model novelty, but past some panel-size/anchor-density boundary the aggregate should stabilize
    toward the prospective score. Without mapping this boundary, reported advantages confound true
    novelty with under-resourced judging..  [research_direction_only] Deferred by independent
    scientific triage (priority 6 of 6; the experiment budget funds the top 2). Best case:
    Identifies resource thresholds (judge count, anchor density) beyond which retrospective and
    prospective estimates converge, useful for standardizing future evaluations and explaining
    variance across prior studies. Strongest objection: Two-dimensional resource sweep is expensive
    and largely orthogonal to the central pre-registration claim; convergence thresholds may be
    task- and judge-specific and not generalize, and the grid dilutes power for the primary >=50%
    question. Alternative explanation: Convergence at high panel size could reflect averaging-out of
    idiosyncratic judge noise rather than correction of a systematic model-favoring bias.

**方法草图**
    Using the pre-registration ledger to freeze hypotheses, sweep two resource boundaries
    independently: number of independent LLM judges aggregated per hypothesis (1..k) and anchor-set
    density used for overlap scoring (sparse..dense). At each grid point compute the model-minus-
    baseline novelty advantage under prospective (ledger-locked) and retrospective protocols,
    identifying the panel-size and anchor-density thresholds beyond which retrospective and
    prospective estimates converge.

**预期贡献**
    Advantage estimates across the judge-count x anchor-density grid with identified convergence
    thresholds; report the minimum judge count and anchor density at which prospective-retrospective
    gap falls below a stated tolerance, and whether the >=50% shrinkage persists above that
    boundary.

**你的判断**：`[ ] NONE  [ ] PARTIAL  [ ] FULL  [ ] UNCERTAIN`

**理由（必填，一句话）**：

    

---
