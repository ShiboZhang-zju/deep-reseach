# 从"研究方向"到"Idea"的难点分析与优化建议

> 分析日期：2026-08-03
> 分析对象：F:\deep research（Evidence-grounded Research Gap Discovery and Idea Validation System）
> 分析方法：从头审视 runner / steps / policy / gate 代码，结合 docs/refactor 重构进度与历史链路实测报告

---

## 一、结论先行

"从研究方向到 idea 很难做"不是工程 bug，而是**问题本身的难度 + 当前架构把这个难度放大了**。

具体来说，你已经做了一件正确的事：把系统从"检索完就让 LLM 编 idea"（V1，幻觉严重、5 个 idea 全 reject）重构成"证据驱动的 Gap 发现 + 对抗审计 + 干预设计"（V2）。V2 的方向是对的，但它把一条**串行的、每一环都可能返回空**的长链路暴露了出来：

```
方向 → Contract → Questions → 多轮检索 → Evidence 抽取 → Coverage
     → Gap 挖掘 → Gap 审计 → Intervention → 最小实验 → Idea
```

这条链上有 **7 个"硬闸门"**，任何一个闸门不通过，整个任务就以 `more_research_required` / `insufficient_evidence` / `abstained` 结束，**产不出 idea**。所以你的主观感受"很难做"，本质是：**闸门太严 + 上游供给不足 + 没有降级路径 = 大概率空手而归**。

优化的核心思路不是"放宽闸门骗一个 idea 出来"（那是 V1 的老路），而是：**提升上游证据供给质量、把硬闸门改成带反馈的软闸门、增加"定向补检索"的回环、并给出分级可信度的 idea 而非 0/1 的通过/拒绝**。

---

## 二、当前链路到底卡在哪（逐闸门诊断）

以下每一个都是 runner.py 里会导致任务提前终止、产不出 idea 的真实分支。

### 闸门 1：Phase2 Readiness Gate（`readiness_gate.py`）

位置：`run_task` 检索循环之后。要求：
- 有 active contract
- 有 active questions
- 有非 rejected/conflicted 的 evidence
- **每一个 importance ≥ 0.7 的高重要性问题都必须有最新一轮的 coverage snapshot**（否则直接 `failed`）
- 至少一个高重要性问题 coverage_score > 0（否则 `more_research_required`）

**卡点**：`missing_snapshot_ids` 一旦非空就直接 `failed`。也就是说只要某个高重要性 question 在最后一轮没有生成 coverage 记录（LLM 抽 evidence 失败、PDF 没下下来、限流跳过），整个任务就 failed，**连 Gap 挖掘都进不去**。这是"很难做"的第一道墙。

### 闸门 2：Gap Mining Admission（`mine_gaps.py` → `evaluate_gap_mining_admission`）

每个 question 想进入 Gap 挖掘，必须同时满足：
- `INSUFFICIENT_INDEPENDENT_PAPERS`：支撑证据来自 **≥ 2 篇不同论文**
- `NO_FULLTEXT_LOCATABLE_EVIDENCE`：**≥ 1 条全文可定位证据**（要求 verified/upgraded + original_span + source_chunk_hash + 页码 + span 偏移）
- `NO_LIMITATION_SIGNAL`：**必须有 limitation / negative_result 类型的证据**
- `UNRESOLVED_VERIFIED_CONTRADICTION`：不能有未解决的矛盾证据

**卡点**：这是最致命的一道。"必须有 limitation 信号"意味着：如果检索到的论文里没人明确写"我们这个方法有 XX 局限 / 在 YY 场景失败"，那这个 question 永远 PASS 不了，永远挖不出 Gap。而现实是：
1. 论文摘要里很少直白写 limitation（都在正文 discussion 里）；
2. `extract_evidence.py` 在 Windows 上 PDF 全文抽取经常失败，退化成 abstract-only；
3. abstract-only 证据的 `verification_status` 是 `abstract_only`，虽然在 `_ADMISSIBLE_STATUSES` 里，但 `_is_fulltext_locatable` 要求 verified/upgraded，所以 `NO_FULLTEXT_LOCATABLE_EVIDENCE` 极易触发。

结果：**上游 evidence 供给（尤其是全文 + limitation）不足 → admission 全 UNKNOWN → 挖不出 Gap → 任务 `more_research_required`**。

### 闸门 3：Gap Candidate 校验（`mine_gaps.py` 循环内）

即使 admission 过了、LLM 生成了 gap 候选，每个候选还要过 6 项校验：gap_type 白名单、question_id 合法、evidence_id 合法、无矛盾证据、**≥ 2 篇论文支撑**、**≥ 1 条全文可定位**、**≥ 1 条 limitation 信号**。任何一项不满足直接丢弃。

**卡点**：与闸门 2 叠加，双重"全文 + limitation + 2 篇"约束。LLM 只要引对了 id 但恰好那几条证据不是全文/不是 limitation，候选就被扔掉。

### 闸门 4：Gap Search Admission（`audit_gaps.py` → `evaluate_gap_search_admission`）

这是你现在 git 正在改的文件。每个存活 Gap 要做对抗审计前，必须先跑 3 个对抗 query family，并且：
- `INSUFFICIENT_COMPLETED_QUERIES`：完成的 query ≥ 2
- `INSUFFICIENT_QUERY_FAMILIES`：完成的 family ≥ 2
- `SEARCH_SUCCESS_RATE_TOO_LOW`：成功率 ≥ 0.5
- `NO_SUCCESSFUL_SOURCE`：至少一个有效源
- `INSUFFICIENT_GAP_SPECIFIC_PAPERS`：命中论文 ≥ 3
- `NO_EXTERNAL_NEIGHBOR`：至少一篇非支撑论文的外部邻居

**卡点**：这一步强依赖检索源可用性。历史报告显示 Semantic Scholar 无 key 时大量 429 限流；一旦对抗检索被限流，`SEARCH_SUCCESS_RATE_TOO_LOW` / `NO_SUCCESSFUL_SOURCE` 触发，审计返回 `uncertain` + `more_search`，Gap 拿不到 `surviving` 状态。

### 闸门 5：审计判定（`audit_gap_candidate`）

只有 LLM 判 `audit_result == "confirmed"` 且 `recommended_action == "continue"` 时，Gap 才 `surviving`。`closed` → rejected，`partially_closed` → narrow（不 surviving），`uncertain` → more_search（不 surviving）。

**卡点**：审计 prompt 明确要求"证据不足时返回 uncertain"，这是对的（防幻觉），但意味着**只要邻居论文信息不够，就永远 uncertain，Gap 永远 survive 不了**。runner 里 `if not state.surviving_gap_ids` 直接 `more_research_required` 返回。

### 闸门 6：Intervention Hard Gates（`generate_interventions.py`）

存活 Gap 才生成 intervention，且每个 intervention 要过 3 个硬闸门：
- evidence gate：Gap 有 ≥ 2 条证据
- novelty gate：审计 confirmed 且有 remaining_delta
- feasibility gate：符合 contract 资源约束（不允许训练却提到 training → FAIL）

三门全 PASS 才 `passed`。runner 里 `if not passed_intervention_ids` → `more_research_required`。

**卡点**：feasibility gate 用**关键词子串匹配**（" training"、"训练"、"benchmark"）判 FAIL。LLM 只要在方案里顺口写了"训练一个小分类器"，哪怕只是辅助步骤，也会被 contract 的 `allow_model_training=False` 一票否决。

### 闸门 7：最小实验生成（`generate_minimal_experiments.py`）

通过 intervention 才生成 idea（decision=`conditional_go`）。生不出 → `abstained`。

**小结**：7 道闸门串联，每道通过率假设都有 70%（乐观），端到端通过率 ≈ 0.7^7 ≈ **8%**。这就是"很难做"的数学根源——**不是某一处坏了，是串行长链 + 全严闸门的结构性问题**。

---

## 三、上游供给的三个真实短板

闸门严不严是一方面，更根本的是**喂给闸门的东西质量不够**。三个短板，全部有代码/历史证据：

### 短板 A：检索源限流导致证据覆盖不全

`.env` 里 `SEMANTIC_SCHOLAR_API_KEY` 为空（历史报告实测大量 429）。S2 是唯一带 venue + citation + 引用关系的高质量源，限流后经典论文（如 TOGA）缺失，Gap 审计的"近邻对比"失去参照物，只能返回 uncertain。

### 短板 B：Windows 上 PDF/RAG 全文能力被禁用

`runner.py` 第 489 行硬编码 `skipping RAG indexing (disabled on Windows, using abstract fallback)`。`extract_evidence.py` 虽然会尝试 `download_pdf_multi_source` + PyMuPDF 抽全文，但一旦失败就退化成 abstract-only。而闸门 2/3 又强制要求"全文可定位证据"——**供给（abstract）和需求（full-text locatable）根本对不上**。这是当前最大的结构性矛盾。

### 短板 C：Limitation 信号稀缺且抽取困难

Gap 挖掘的整个哲学是"从别人写的 limitation/negative result 里找机会"。但：
1. 大部分论文摘要不写 limitation；
2. `EVIDENCE_EXTRACT` 要抽出 `evidence_type == "limitation"` 的证据，本身就依赖全文（见短板 B）；
3. 没有 limitation 证据 → 闸门 2 的 `NO_LIMITATION_SIGNAL` 必然触发。

---

## 四、优化建议（按投入产出比排序）

### 第一优先级：让链路"能出东西"（结构性，1-2 天）

这一档不改闸门的严格性，而是**改链路形态**，让"卡住"变成"降级 + 反馈"而不是"终止"。

**O1. 把 7 道硬闸门改成"软闸门 + 分级产出"**
不要在每个闸门后 `return`。而是让 Gap / Intervention / Idea 都带一个 `confidence_tier`（如 A=证据充分可信 / B=证据部分支撑待验证 / C=推测性方向）。闸门不通过的候选降级到 B/C 档，**照样入库、照样展示给用户**，只是标注"证据不足，需人工判断/补检索"。

理由：你的原则是"允许 0 个可信 idea"，这是对的（防幻觉）。但"允许 0 个 A 档 idea"不等于"必须返回 0 个 idea"。用户要的是"给我几个有依据、标清置信度的方向让我挑"，而不是"要么完美要么空手"。

**O2. 在每个 `more_research_required` 出口挂一个"定向补检索"回环**
现在闸门 2/4/5 失败都直接终止。应该改成：失败时读取失败原因（reason_codes），生成针对性的补充 query（缺 limitation → 检索 "limitations of X" / "failure cases of X"；缺近邻 → 检索 Gap 的 claimed_delta），跑 1 轮定向检索后**回到失败的那个闸门重试**，最多 N 次。runner 里其实已有 `_idea_retry_search_round` 的雏形，但 V2 链路没接。

**O3. 修复 Readiness Gate 的"一票 failed"**
闸门 1 里 `missing_snapshot_ids` 非空就 failed 太脆。改成：允许部分高重要性问题缺 snapshot（降级为 `more_research_required` 而非 `failed`），只要**整体覆盖达到某个比例**（如高重要性问题 ≥ 60% 有 coverage）就放行进入 Gap 挖掘。

### 第二优先级：补上游供给（1-2 天，见效最快）

**O4. 配置 Semantic Scholar API Key**（10 分钟，收益最大）
免费申请（https://www.semanticscholar.org/product/api#api-key-form），填进 `.env`。限速从 20 req/min 提到 1 req/s，429 基本消失，经典论文和引用关系回来了，闸门 4/5 的近邻审计立刻有料。

**O5. 解决"全文供给 vs 全文需求"的矛盾**，二选一或都做：
- (a) 用 **API embedding**（OpenAI / Venus 的 embedding 接口）替代本地 PyTorch，绕开 Windows segfault，重新启用 RAG 全文；
- (b) 若短期不启用全文，就**降低闸门 2/3 对 full-text 的硬要求**：把 `NO_FULLTEXT_LOCATABLE_EVIDENCE` 从"必须"改成"加分项"，允许高质量 abstract 证据（多篇一致 + 明确 limitation 措辞）也能支撑 Gap，但标注为 B 档。

**O6. 增强 limitation 信号抽取**
在 `EVIDENCE_EXTRACT` prompt 里，除了正文，专门针对论文的 "Limitations" / "Threats to Validity" / "Future Work" / "Discussion" 章节做定向抽取（`extract_pdf_text_by_section` 已能分章节）。这些章节是 limitation 的富矿。同时对 abstract 也让 LLM 判断"隐含的能力边界"（如"we focus on X"隐含"未覆盖 non-X"）。

### 第三优先级：让检索更精准（2-3 天）

**O7. 检索结果预过滤**（历史报告 P1-2）
6 源 × 5 query × 15 篇 = 450 篇/轮，大量噪声（肝癌、人脸识别混进 test oracle 主题）。用 embedding 算 topic-paper 相似度，< 0.5 直接丢弃。噪声减少后，evidence 抽取和 Gap 挖掘的信噪比大幅提升。

**O8. Query 生成绑定 Question + Coverage**
V2 里 `generate_queries` 已经是 coverage-driven，确认它真的优先给**低覆盖的高重要性 question** 生成 query，而不是泛泛检索。让检索资源集中在"最可能挖出 Gap 的问题"上。

### 第四优先级：产出体验（1 天）

**O9. 无论成败都产出一份"研究态势简报"（Research Landscape Brief）**
目标架构文档里已经设计了这个（Evidence + Coverage 之后生成），但 V2 里 `pipeline_version >= 2` 分支跳过了 report。哪怕最终 0 个 surviving gap，也应该给用户：研究问题树、论文角色分布、覆盖矩阵、已知 limitation 清单、未覆盖区域、以及"为什么没产出可信 idea"的诚实解释 + 建议的补充方向。**这样用户即使没拿到 idea，也拿到了有价值的领域地图和明确的下一步。**

---

## 五、我建议的落地顺序

如果只做一件事：**O4（配 S2 key）**，10 分钟，立刻缓解限流，闸门 4/5 通过率显著上升。

如果做一个下午（约 4 小时）：**O4 + O3 + O9**。配 key + 放松 readiness 一票否决 + 无论如何产出态势简报。这三个组合能让"大概率空手"变成"大概率至少有领域地图 + 部分 B 档方向"。

如果做一周：加上 **O1（软闸门分级）+ O2（定向补检索回环）+ O5(a)（API embedding 启用全文）**。这才是根治——把 8% 的端到端通过率结构性地提上来，同时不牺牲防幻觉原则。

---

## 六、一句话总结

你没做错方向——V2 的"证据驱动 + 对抗审计"是对抗 LLM 幻觉的正解。你现在的痛苦来自**把一条防御性极强的长链做成了全严硬闸门串联，且上游（检索限流 + Windows 无全文 + limitation 稀缺）供给不足**。解法不是退回 V1 让 LLM 乱编，而是：**软闸门分级产出 + 失败即定向补检索回环 + 补齐上游证据供给 + 永远至少交付一份领域态势简报**。这样既守住"允许 0 个可信 idea"的底线，又不会让用户"很难做"到大概率空手而归。

---

## 附录：已实施的修复（2026-08-03）

以下五项已落地并通过测试，直接缓解"很难做"：

O4 配置层——`config.py` 新增 `effective_s2_rate_per_min` / `effective_openalex_rate_per_min` 属性：一旦 `.env` 填入 S2 key，限速自动从 20/min 提到 5000/min，无需手改任何限速值；`rate_limiter.py` 改用该有效值；`.env` 补了 key 申请指引。（S2 key 需你自行免费申请：https://www.semanticscholar.org/product/api#api-key-form ）

O3 Readiness Gate 一票 failed 修复——`readiness_gate.py` 引入 `MIN_HIGH_IMPORTANCE_SNAPSHOT_RATIO=0.6`。高重要性问题缺 coverage snapshot 时不再一律 failed：全部缺失才 failed，部分缺失（比例 < 0.6）降级为 `more_research_required`（可恢复），达标则放行。任务不再因单个问题缺快照而整体死亡。

O6 limitation 抽取增强——evidence 抽取 prompt 增加"局限/边界优先"指令；对 abstract 引导 LLM 识别隐含能力边界（"we focus on X" 隐含未覆盖 non-X），对 conclusion 章节引导聚焦 Limitations / Threats to Validity / Future Work。缓解 Gap 挖掘的 limitation 信号饥荒。

O5(b) 降低全文硬要求（A/B 分级）——`mine_gaps.py` 的准入不再把"全文可定位证据"当硬门槛：有全文 span 的 Gap 记为 A 档（`provenance_status=complete`），仅摘要级强证据（≥2 篇 + limitation）记为 B 档（`partial`）并交由后续审计确认，而不是直接拒绝。Windows 无全文时也能产出候选 Gap。

O9 Research Landscape Brief——新增 `generate_landscape_brief.py`（确定性生成、不依赖 LLM、不抛异常），在 runner 的 6 个终止出口前调用。无论任务以 `more_research_required` / `abstained` / 成功结束，用户都能拿到一份含研究问题树、覆盖矩阵、论文角色分布、已知 limitation、候选 Gap 及审计状态、以及诚实的"为何没产出可信 idea + 建议下一步"的领域态势简报。

测试：改动涉及的单元/集成测试全部通过（含新增 `test_landscape_brief.py` 3 项、更新 `test_phase2a_final_closure.py` 2 项）。全套 166 passed；唯一失败的 `test_opportunity_pipeline_e2e` 为既有 WIP（gap search admission gate 与 `perform_search=False` 冲突）导致，与本次修复无关，已用 git stash 验证。

尚未实施（建议后续）：O1 全链路 confidence_tier 贯通到 intervention/idea、O2 失败即定向补检索回环、O5(a) 用 API embedding 启用全文 RAG。
