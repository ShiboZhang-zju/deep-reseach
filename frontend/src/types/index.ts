// TypeScript types matching backend Pydantic schemas

export interface Task {
  id: string;
  user_input: string;
  normalized_topic: string | null;
  status: string;
  current_round: number;
  max_rounds: number;
  stop_reason: string | null;
  state_json: string | null;
  created_at: string;
  updated_at: string;
}

export interface ResearchState {
  task_id: string;
  user_input: string;
  normalized_topic: string;
  keywords: string[];
  research_questions: string[];
  current_round: number;
  used_queries: string[];
  knowledge_gaps: string[];
  collected_paper_ids: string[];
  high_priority_paper_ids: string[];
  medium_priority_paper_ids: string[];
  low_priority_paper_ids: string[];
  round_summaries: string[];
  selected_idea_ids: string[];
  user_feedback: string;
  stop_reason: string;
}

export interface Paper {
  id: string;
  title: string;
  abstract: string | null;
  authors_json: string | null;
  year: number | null;
  venue: string | null;
  doi: string | null;
  arxiv_id: string | null;
  url: string | null;
  citation_count: number;
  sources_json: string | null;
  final_score: number | null;
  priority: string | null;
  reason: string | null;
  summary: string | null;
}

export interface Round {
  id: string;
  round_number: number;
  queries_json: string | null;
  papers_found: number;
  new_papers: number;
  duplicate_rate: number | null;
  summary: string | null;
  knowledge_gaps_json: string | null;
  created_at: string;
}

export interface Report {
  id: string;
  task_id: string;
  content_markdown: string;
  content_json: string | null;
  created_at: string;
}

export interface Idea {
  id: string;
  task_id: string;
  title: string | null;
  description: string | null;
  motivation: string | null;
  method_sketch: string | null;
  expected_contribution: string | null;
  novelty: number | null;
  feasibility: number | null;
  significance: number | null;
  evidence_support: number | null;
  differentiation: number | null;
  experimentability: number | null;
  potential_impact: number | null;
  risk: number | null;
  final_score: number | null;
  decision: string | null;
  related_paper_ids_json: string | null;
  user_selected: boolean;
  created_at: string;
}

export interface Experiment {
  id: string;
  task_id: string;
  idea_id: string;
  hypothesis: string | null;
  dataset: string | null;
  baselines: string | null;
  metrics: string | null;
  steps_markdown: string | null;
  steps_json: string | null;
  risks: string | null;
  created_at: string;
}

export interface Trace {
  id: string;
  step_name: string;
  step_type: string;
  round_number: number | null;
  input_json: string | null;
  output_json: string | null;
  llm_tokens_used: number | null;
  duration_ms: number | null;
  created_at: string;
}

export interface SSEEvent {
  event: string;
  data: any;
}

// Status label mapping
export const STATUS_LABELS: Record<string, string> = {
  pending: '待启动',
  clarifying: '分析方向中',
  waiting_for_clarification: '等待澄清',
  searching: '检索中',
  summarizing: '摘要中',
  analyzing_papers: '论文深度分析',
  reporting: '生成报告',
  generating_ideas: '生成 Ideas',
  waiting_for_user_review: '等待用户审阅',
  judging_ideas: '深度评估',
  generating_experiment: '生成实验方案',
  done: '已完成',
  stopped: '已停止',
  failed: '失败',
  // Phase 0: New statuses
  insufficient_evidence: '证据不足',
  more_research_required: '需补充检索',
  mining_gaps: '挖掘 Gap',
  auditing_gaps: 'Gap 审计中',
  checking_feasibility: '可行性检查',
  synthesizing_ideas: '合成 Ideas',
};

export const STATUS_COLORS: Record<string, string> = {
  pending: 'bg-gray-100 text-gray-700',
  clarifying: 'bg-blue-100 text-blue-700',
  waiting_for_clarification: 'bg-amber-100 text-amber-700',
  searching: 'bg-indigo-100 text-indigo-700',
  summarizing: 'bg-indigo-100 text-indigo-700',
  analyzing_papers: 'bg-indigo-100 text-indigo-700',
  reporting: 'bg-purple-100 text-purple-700',
  generating_ideas: 'bg-purple-100 text-purple-700',
  waiting_for_user_review: 'bg-amber-100 text-amber-700',
  judging_ideas: 'bg-purple-100 text-purple-700',
  generating_experiment: 'bg-purple-100 text-purple-700',
  done: 'bg-green-100 text-green-700',
  stopped: 'bg-gray-100 text-gray-700',
  failed: 'bg-red-100 text-red-700',
  // Phase 0: New statuses
  insufficient_evidence: 'bg-orange-100 text-orange-700',
  more_research_required: 'bg-amber-100 text-amber-700',
  mining_gaps: 'bg-purple-100 text-purple-700',
  auditing_gaps: 'bg-purple-100 text-purple-700',
  checking_feasibility: 'bg-purple-100 text-purple-700',
  synthesizing_ideas: 'bg-purple-100 text-purple-700',
};
