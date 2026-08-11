# Deep Research 项目不足分析与开源方案对比报告

> 分析日期：2026-08-06
> 分析对象：本仓库 `f:\deep research`（Evidence-grounded Research Gap Discovery and Idea Validation System）
> 分析方法：从头核查 runner / mine_gaps / generate_minimal_experiments / targeted_research 等实际代码，结合 docs/14-17 与 refactor/PROGRESS，横向对比 7 个代表性开源/研究系统的机制与实证数据
> 目标：项目定位是"输入研究方向 → 输出可行 idea"，本报告评估其结构性不足，并对照业界主流方案给出优化方向

## 执行摘要

本项目 V2 走的是一条"证据驱动 + 对抗审计"的长链路（Contract → Questions → 多轮检索 → Evidence 抽取 → Coverage → Gap 挖掘 → Gap 审计 → Intervention → 最小实验 → Idea），方向上与业界最前沿的防幻觉理念一致，其"禁止编造 baseline/dataset、要求 falsification_condition、span 原文回溯校验"的严格程度甚至超过多数开源方案。但它把七道硬闸门做成了串行结构，端到端通过率理论上仅约 8%（0.7^7），在无Semantic Scholar key 的当前配置下上游证据供给不足，导致 Gap 审计系统性判uncertain、几乎产不出可信 idea。横向对比发现，本项目最需要补上的三块短板是：缺少 Google AI Co-Scientist 式的"生成-辩论-演化 + 排名锦标赛"的正向发散与择优机制（现在只有单向的减法闸门），缺少 OpenScholar/PaperQA2 式的成熟全文 RAG 与引用回溯供给（现在被Windows 限制退化到摘要级），以及缺少一个诚实的"分级产出"出口设计（虽已有Landscape Brief 与 A/B 分档雏形，但 Intervention/Idea 层仍是硬闸门）。核心结论是：本项目不该退回"让 LLM 乱编 idea"的老路，而应在守住证据底线的同时，引入正向的假设发散-择优环节、补齐全文证据供给、并把分级置信度贯通到最终 idea 层。

## 一、背景：本项目当前真实实现（基于代码核查，非文档复述）

本项目的核心链路在 `backend/app/agent/runner.py` 中编排，各步骤实现在 `agent/steps/` 下。经核查，代码与 docs/16、docs/17 描述一致，V2 重构与 O1-O9 优化确实已落地：

在证据准入上，`mine_gaps.py` 定义了 `evaluate_gap_mining_admission`，要求每个研究问题的支撑证据来自至少 2 篇不同论文（`INSUFFICIENT_INDEPENDENT_PAPERS`）、必须含 limitation/negative_result 类型信号（`NO_LIMITATION_SIGNAL`）、不能有未解决矛盾（`UNRESOLVED_VERIFIED_CONTRADICTION`）；`_is_fulltext_locatable` 要求证据带 `verification_status ∈ {verified, upgraded}` + `original_span` + `source_chunk_hash` + 页码 + span偏移。O5(b) 已把全文从硬门槛降为分档信号（有全文 span记A 档、仅摘要强证据记 B 档），策略版本 `evidence-admission-v1`。

在idea 生成上，`generate_minimal_experiments.py` 的 system prompt 明确要求"Do not invent dataset or baseline names; use generic roles when the evidence does not identify a specific valid name"，输出 schema `MinimalExperimentSchema` 强制包含 `hypothesis`、`success_condition`、`falsification_condition`。这是本项目相对同行最扎实的防幻觉设计。

在链路韧性上，`runner.py` 已引入 `run_targeted_research_round` 与 `can_remediate`（O2 定向补检索回环），以及 `generate_landscape_brief`（O9 无论成败都产出领域态势简报）。docs/17 记录第 1-7 项优化已实施、后端 199 测试通过。

由此确认，本项目的问题不在"做得不够严"，恰恰在"太严且是单向减法"——这正是与其他方案对比时最凸显的结构性特征。

## 二、七个对标方案的机制与实证数据

下表先给出总览，随后逐一展开。

| 系统 | 机构/年份 | 核心范式 | idea/假设生成机制 | 防幻觉机制 | 关键实证数据 |
|------|-----------|----------|-------------------|-----------|-------------|
| The AI Scientist v1/v2 | Sakana AI，2024-2025 | 端到端全自动科研 | LLM 反思式发散 + 模板代码 + Agentic Tree Search | Semantic Scholar/OpenAlex 查新 | 单篇约 15 美元；v2 论文通过 ICLR workshop 双盲评审 |
| Google AI Co-Scientist | Google DeepMind，2025.02 | 多智能体假设生成 | 生成-辩论-演化 + Elo 排名锦标赛 | 反思agent 审稿 + test-time compute 扩展 | 肝纤维化提出 3 个新靶点、AML 药物 KIRA6 IC50 低至13nM、48 小时复现十年科研结论 |
| OpenScholar | Ai2，2024.11（Nature 2026） | 检索增强文献综合 | 不做 idea，做带引用的综合回答 | 检索 + 重排 + 自反馈 + 引用验证 | 纯 LLM 引用编造率 78-98%，OpenScholar 降至 0%；ScholarQABench 超 GPT-4o 5% |
| ResearchAgent | Microsoft，2024.09 | 迭代 idea 生成 | 文献 + 实体知识图谱 + ReviewingAgents 反馈 | 人类偏好对齐的多维评估 | 多学科评估验证 idea 质量优于基线 |
| STORM / Co-STORM | Stanford OVAL，2024 | 生成带引用长文 | 多视角提问 + 模拟专家对话 | 来源可信度过滤 + 逐节引用 | 70% 维基编辑认为对前期研究有用 |
| PaperQA2 | FutureHouse，2024.09 | 全文 RAG 科学问答 | 不做 idea，做超人类文献综合 | 全文 RAG + 引用回溯 | 综述任务超越人类专家、引用准确率高 |
| GPT-Researcher |开源社区 | 自主研究报告 | plan-and-execute 多源检索 | 多来源交叉 | 工程化程度高，无严格证据校验 |

### The AI Scientist（Sakana AI）：端到端全自动，但 idea 靠 LLM 发散、查新偏弱

AI Scientist v1 的完整流程是七阶段：想法生成（LLM 结合模板代码多轮反思）→ 新颖性检查（用 Semantic Scholar/OpenAlex 检索文献保留新颖 idea）→ 实验执行 → 结果可视化 → 论文写作 → 自动评审（LLM 输出创新性/质量/清晰度评分）→ 改进迭代。v2 引入 Agentic Tree Search 与并行代理架构，实现从假设到论文的全自动，其生成的论文曾通过 ICLR 2025 workshop 的双盲同行评审，单篇成本约 15 美元。

关键对比点在于：AI Scientist 的 idea 生成本质是"LLM 自由发散 + 事后查新"，其 `generate_temp_free_idea` 主循环默认生成 20 个 idea、每个做 5 轮反思，反思时强制至少调一次 Semantic Scholar 搜索以"确保区分于已有文献"。这与本项目"从证据 limitation 反推 gap"的减法思路正好相反——AI Scientist 是加法（先发散再剪枝），本项目是减法（先设闸门再看谁能通过）。AI Scientist 因此高产但被批 idea 质量参差、自动评审可靠性存疑；本项目则相反，质量守得住但极易空手。

### Google AI Co-Scientist：最值得本项目借鉴的"生成-辩论-演化 + 排名锦标赛"

Co-Scientist（arXiv:2502.18864，基于 Gemini 2.0）是六个专门agent 协作：Generation（生成初步假说）、Reflection（当审稿人评估逻辑性/创新性/可测试性）、Ranking（用 Elo 排名锦标赛，让假说两两模拟学术辩论）、Evolution（基于反馈优化/合并/简化假说）、Proximity（聚类去重）、Meta-review（元综述）。整体工作机制概括为"生成-辩论-演化（Generate-Debate-Evolve）"，通过 test-time compute 扩展迭代自我改进，且验证了较高 Elo 分与正确答案概率正相关。

其实测极具说服力：在急性髓性白血病药物再利用中提出 KIRA6，体外多细胞系测试 IC50 低至 13nM；在肝纤维化中提出 3 个新颖表观遗传靶点，至少2 个候选药物显著抑制纤维化标记；在抗菌素耐药性问题上，48 小时内独立提出了某顶级团队用十年才验证的 cf-PICIs 跨种传播假说。

对本项目的直接启示是：Co-Scientist 有一个本项目完全缺失的正向环节——它先大量生成假设，再用辩论和 Elo 锦标赛择优。本项目目前只有"减法闸门"（Gap 审计只会confirm/close/uncertain），没有"让多个候选相互竞争、迭代进化"的正向机制，所以一旦上游证据不足，就只能全部卡在 uncertain，而不会像 Co-Scientist 那样从一堆候选里排出相对最优的几个。

### OpenScholar（Ai2）：全文 RAG 把引用编造率从 78-98% 打到 0%

OpenScholar（Nature 2026 正式发表，arXiv:2411.14199）是本项目在"证据供给"这块最该参考的工程标杆。它的数据存储含 4500 万篇开放论文、2.37 亿段落嵌入，正文按 250 词分块；检索用 110M 双编码器（Contriever 持续预训练）+ 340M 交叉编码器重排（BGE-reranker 微调），每篇论文最多保留 3 段、把归一化引用数纳入相关性；推理时做带检索的自反馈迭代（生成初稿 + 最多 3 条反馈 + 检索补充 + 引用验证）。

其量化结论是本报告最有力的论据：纯 LLM 在引用最新文献时，Llama 3.1 8B 的引用编造率高达 92-98%、GPT-4o 也有 78-95%；而 OpenScholar 通过强制检索把编造率降到 0%。在 ScholarQABench 上，OpenScholar-8B 正确性比 GPT-4o 高 5%、比 PaperQA2 高 7%，成本比 PaperQA2 低数个数量级；专家盲评中 OS-GPT4o 对人类专家答案胜率 70%。消融显示重排器和自反馈都对引用准确率有显著贡献。

对本项目的启示是：本项目 Gap 挖掘准入门"偏好全文可定位证据"的需求是对的，但供给侧在 Windows 上曾退化到摘要级（docs 已用 API embedding 缓解）。OpenScholar 证明了成熟的"检索 + 重排 + 引用回溯"管线能把幻觉压到 0，本项目应把这套供给做扎实，而不是靠闸门去挡不足的证据。

### ResearchAgent（Microsoft）：文献 + 实体知识图谱 + ReviewingAgents

ResearchAgent（arXiv 2024.09）从核心论文出发，沿学术引用图连接相关文献，并从"以实体为中心的知识存储"检索实体来激发跨领域联想，然后迭代生成问题-方法-实验设计。其特色是引入多个 ReviewingAgents，用与人类偏好对齐的评估标准（清晰度、相关性、原创性、可行性、重要性）给 idea 打分并反馈，形成迭代改进闭环。

对本项目的启示：本项目有 Question 分解和 Coverage 矩阵，但缺少"实体/概念知识图谱"这一激发跨领域新颖性的维度，也缺少 ResearchAgent 式的多维度评审反馈闭环（本项目的审计是单向裁决而非迭代改进）。

### STORM/Co-STORM（Stanford）与 PaperQA2（FutureHouse）：多视角提问与全文问答的工程范式

STORM 的核心是"多视角提问 + 模拟专家对话"：先检索同类主题识别不同视角，再让 LLM 分别扮演提问者和检索型专家多轮对话，最后综合成带引用大纲和长文；Co-STORM 加入人类在环和主持人 agent、维护动态思维导图。PaperQA2（arXiv:2409.13740）则在全文 RAG 问答上做到综述质量超越人类专家、引用高度可靠。

对本项目的启示：STORM 的"多视角提问"是提升覆盖度和发现盲区的轻量机制，比本项目"按研究轴机械分解 question"更能捕捉不同利益相关者视角；Co-STORM 的人类在环也提示本项目可以在关键闸门处引入人工介入而非纯自动裁决。

## 三、综合分析：本项目相对同行的三个结构性不足

将七个方案与本项目对齐后，可归纳出三个最关键的不足，都有代码或实证支撑。

第一，只有"减法闸门"，缺少"正向发散-辩论-择优"机制。这是与Co-Scientist 对比后最刺眼的差距。本项目从 Gap 到 Idea 全程是过滤式的：审计只做 confirm/close/uncertain，Intervention 有 3 道硬闸门，Idea 是 go/revise/reject。没有任何环节让系统"先大量生成候选、再让它们相互竞争进化、排出相对最优"。后果是——当证据充分时它能挑出好 idea，但当证据不足（无key 模式的常态）时，它不会退而求其次给出"相对最有希望的几个方向"，而是全部判uncertain 直接空手。Co-Scientist 的 Elo 锦标赛证明，即使没有绝对确定的答案，相对排序也能产出有价值的方向清单。

第二，证据供给侧不够成熟，却用极严的全文需求去卡它。本项目 Gap 挖掘偏好全文可定位证据（`_is_fulltext_locatable` 要求 span + chunk_hash + 页码 + 偏移），这个需求方向和 OpenScholar/PaperQA2 一致且正确，但供给侧的 RAG 管线相比 OpenScholar 的 4500 万论文库+ 双编码器 + 交叉重排 + 自反馈还有明显差距，且长期受Windows 环境和无 S2 key 限制。OpenScholar 用0% 引用编造率证明了"把供给做厚"比"把闸门设严"更能根治幻觉。本项目应优先投资检索/重排/引用回溯的供给质量，而非继续加码闸门。

第三，分级置信度没有贯通到最终产出，用户拿到的是 0/1 结果而非分级清单。虽然本项目已在 Gap 层实现 A/B 分档、并有 Landscape Brief兜底，但 Intervention 和 Idea 层仍是硬 PASS/FAIL。而 Stanford 的大规模人类实验（arXiv:2409.04109，100+ NLP 研究者盲审）恰恰指出：LLM 生成的 idea 在新颖性上显著优于人类专家（p<0.05），只是可行性略弱，且 LLM 自评不可靠、生成多样性不足。这说明"要么完美要么拒绝"的硬裁决会误杀掉大量新颖但可行性待验证的 idea——正确做法是像ResearchAgent 那样多维打分、分级呈现，把可行性判断和最终取舍交还给人类，而不是系统一票否决。

一个交叉验证的重要信号：微信公众号"星使智算"引用的 Nature 子刊研究提出"高智商大模型未必能想出好 Idea"，与 Stanford 实验的"LLM 自评不可靠 + 多样性不足"相互印证。这提示本项目不应过度信任任何单一 LLM 的裁决（无论是生成还是审计），而应通过多候选、多视角、相对排序来对冲单模型的偏差——这正是 Co-Scientist 锦标赛和 ResearchAgent 多评审的价值所在。

## 四、优化建议（对照同行，按投入产出排序）

第一优先级，引入 Co-Scientist 式的"候选池 + 相对排序"择优层，这是弥补"只有减法"的关键。在现有 Gap/Intervention/Idea 各层，不要一遇不达标就丢弃，而是让候选进入一个池子，用一个轻量的 pairwise 比较（或 Elo 式两两辩论打分）排出相对顺序，即使全部达不到 A 档，也输出"相对最有希望的 Top-K + 明确标注证据缺口"。这直接把"大概率空手"变成"至少给分级方向清单"，且不牺牲防幻觉（因为仍然如实标注证据强度）。

第二优先级，把证据供给做厚，向 OpenScholar 看齐。优先级高于继续调闸门：配置 Semantic Scholar API key（限速从 20/min 提到 5000/min，docs/16 第0 档已指出这是 10 分钟收益最大的一步）；用 API embedding 稳定启用全文 RAG（绕开 Windows PyTorch 问题）；引入交叉编码器重排（哪怕用现成的 BGE-reranker）提升检索精度；检索入库前做embedding 预过滤去噪。供给厚了，全文可定位证据自然增多，Gap 从 B 档升A 档，无需放宽闸门就能提高通过率。

第三优先级，分级置信度贯通到 Intervention 和 Idea 层，并借鉴 ResearchAgent 的多维评审。给 intervention/idea 加 `confidence_tier` 字段（A/B/C），硬闸门不通过时降级入库而非丢弃；评审从单向裁决改为多维打分（新颖性、可行性、证据充分度、资源匹配度分别给分），把可行性 gate 的关键词硬匹配改为 LLM 结构化判断"是否需要训练/规模多大"，避免误杀。

第四优先级，引入 STORM 式多视角提问增强覆盖与新颖性。在 `decompose_research_space` 之外，增加"从不同利益相关者/方法论视角"生成问题的机制，捕捉机械分解漏掉的盲区；并可借鉴 ResearchAgent 的实体知识图谱思路，用跨领域实体联想激发更新颖的 gap。

第五优先级，考虑关键闸门处的人类在环（Co-STORM 思路）。在 Gap 审计或 Idea 定稿等高价值节点，允许用户介入确认或补充方向，而非纯自动 uncertain。这对科研辅助工具尤其重要——Stanford 实验和 Co-Scientist 都强调最终判断需人类参与。

## 五、结论

本项目的方向是正确的，其证据驱动 + 对抗审计 + 禁止编造事实的设计在防幻觉严格程度上甚至领先多数开源方案，OpenScholar 用 0% 引用编造率、PaperQA2 用超人类综述质量共同证明了"检索 grounding 是对抗幻觉的正解"，本项目坚持这条路没有错。但横向对比暴露出它把防御做成了单向减法长链：七道硬闸门串联、上游供给不足、失败即终止，导致在当前配置下大概率产不出可信 idea。补救之道不是退回"让 LLM 乱编"的老路，而是学习 Google AI Co-Scientist 的正向"生成-辩论-演化 + 排名锦标赛"补上择优发散环节、学习 OpenScholar/PaperQA2 把全文证据供给做厚、学习 ResearchAgent 把裁决改为多维分级评审，并始终以分级置信度清单 + 领域态势简报兜底。同时，Stanford 100+ 研究者实验和 Nature 子刊研究共同警示：单个 LLM 的 idea 自评不可靠、多样性不足，因此本项目应通过多候选相对排序和人类在环来对冲单模型偏差，而非依赖任何单点的硬裁决。

## 六、局限性

本报告对开源方案的机制描述主要基于官方论文摘要、机构博客和高质量中文技术解读，部分系统（如 Co-Scientist 六agent 的Elo 评分公式、AI Scientist v2 树搜索细节）未逐行核对源码，具体参数以各自官方 repo 与论文为准。对本项目的评估基于当前 HEAD 代码与 docs/14-17，若后续 commit 已实现分级贯通或择优层，相关不足可能已部分缓解。各方案的量化数据来自其各自评测基准，横向数值不完全可比。

## References
1. [Can LLMs Generate Novel Research Ideas? A Large-Scale Human Study with 100+ NLP Researchers (arXiv:2409.04109)](https://arxiv.org/abs/2409.04109)
2. [OpenScholar: Synthesizing Scientific Literature with Retrieval-Augmented LMs (arXiv:2411.14199)](https://arxiv.org/html/2411.14199v1)
3. [OpenScholar - Nature (2026)](https://www.nature.com/articles/s41586-025-10072-4)
4. [OpenScholar GitHub (AkariAsai/OpenScholar)](https://github.com/AkariAsai/OpenScholar)
5. [Google AI Co-Scientist (arXiv:2502.18864)](https://arxiv.org/abs/2502.18864)
6. [The AI Scientist GitHub (SakanaAI/AI-Scientist)](https://github.com/SakanaAI/AI-Scientist)
7. [The AI Scientist-v2 GitHub (SakanaAI/AI-Scientist-v2)](https://github.com/SakanaAI/AI-Scientist-v2)
8. [ResearchAgent: Iterative Research Idea Generation over Scientific Literature (Microsoft Research)](https://www.microsoft.com/en-us/research/publication/researchagent-iterative-research-idea-generation-over-scientific-literature-with-large-language-models/)
9. [PaperQA2: Language agents achieve superhuman synthesis of scientific literature (arXiv:2409.13740)](https://arxiv.org/abs/2409.13740)
10. [PaperQA2 GitHub (Future-House/paper-qa)](https://github.com/future-house/paper-qa)
11. [STORM GitHub (stanford-oval/storm)](https://github.com/stanford-oval/storm)
12. [AI Scientist v1 框架整体分析 (CSDN)](https://blog.csdn.net/qq_42540492/article/details/149257859)
13. [AI Scientist v2 IDEA 生成机制分析 (CSDN)](https://blog.csdn.net/qq_42540492/article/details/149305159)
14. [谷歌 AI co-scientist 三大案例解析 (CSDN)](https://blog.csdn.net/dinaxuejie/article/details/146993046)
