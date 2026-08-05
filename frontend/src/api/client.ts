import type { Task, Paper, Round, Report, Idea, Experiment, Trace, Gap, Intervention, CoverageRecord, GapAudit, NeighborComparison } from '../types';

const BASE = '/api';

async function fetchJSON<T>(url: string, options?: RequestInit): Promise<T> {
  const resp = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`接口错误 ${resp.status}: ${text}`);
  }
  return resp.json();
}

export const api = {
  // Tasks
  createTask: (userInput: string) =>
    fetchJSON<Task>(`${BASE}/tasks`, {
      method: 'POST',
      body: JSON.stringify({ user_input: userInput }),
    }),

  listTasks: (limit = 50) =>
    fetchJSON<Task[]>(`${BASE}/tasks?limit=${limit}`),

  getTask: (id: string) =>
    fetchJSON<Task>(`${BASE}/tasks/${id}`),

  startTask: (id: string) =>
    fetchJSON<{ status: string }>(`${BASE}/tasks/${id}/start`, { method: 'POST' }),

  stopTask: (id: string) =>
    fetchJSON<{ status: string }>(`${BASE}/tasks/${id}/stop`, { method: 'POST' }),

  submitClarification: (id: string, answers: string[]) =>
    fetchJSON<{ status: string }>(`${BASE}/tasks/${id}/clarify`, {
      method: 'POST',
      body: JSON.stringify({ answers }),
    }),

  submitFeedback: (id: string, content: string, needMoreResearch: boolean) =>
    fetchJSON<{ status: string }>(`${BASE}/tasks/${id}/feedback`, {
      method: 'POST',
      body: JSON.stringify({ content, need_more_research: needMoreResearch }),
    }),

  // Papers
  getPapers: (taskId: string, priority?: string, limit = 100, offset = 0) =>
    fetchJSON<Paper[]>(
      `${BASE}/tasks/${taskId}/papers?limit=${limit}&offset=${offset}${priority ? `&priority=${priority}` : ''}`
    ),

  getRounds: (taskId: string) =>
    fetchJSON<Round[]>(`${BASE}/tasks/${taskId}/rounds`),

  // Report
  getReport: (taskId: string) =>
    fetchJSON<Report>(`${BASE}/tasks/${taskId}/report`),

  // Ideas
  getIdeas: (taskId: string) =>
    fetchJSON<Idea[]>(`${BASE}/tasks/${taskId}/ideas`),

  getGaps: (taskId: string) =>
    fetchJSON<Gap[]>(`${BASE}/tasks/${taskId}/gaps`),

  getGapAudits: (gapId: string) =>
    fetchJSON<GapAudit[]>(`${BASE}/gaps/${gapId}/audits`),

  getGapNeighbors: (gapId: string) =>
    fetchJSON<NeighborComparison[]>(`${BASE}/gaps/${gapId}/neighbors`),

  getInterventions: (taskId: string) =>
    fetchJSON<Intervention[]>(`${BASE}/tasks/${taskId}/interventions`),

  getCoverage: (taskId: string) =>
    fetchJSON<CoverageRecord[]>(`${BASE}/tasks/${taskId}/coverage`),

  selectIdeas: (taskId: string, ideaIds: string[]) =>
    fetchJSON<{ status: string }>(`${BASE}/tasks/${taskId}/ideas/select`, {
      method: 'POST',
      body: JSON.stringify({ idea_ids: ideaIds }),
    }),

  judgeIdeas: (taskId: string) =>
    fetchJSON<any>(`${BASE}/tasks/${taskId}/ideas/judge`, { method: 'POST' }),

  // Experiments
  getExperiments: (taskId: string) =>
    fetchJSON<Experiment[]>(`${BASE}/tasks/${taskId}/experiments`),

  generateExperiments: (taskId: string) =>
    fetchJSON<any>(`${BASE}/tasks/${taskId}/experiments`, { method: 'POST' }),

  exportExperiment: (taskId: string, planId: string, format: string = 'markdown') =>
    fetchJSON<any>(`${BASE}/tasks/${taskId}/experiments/${planId}/export?format=${format}`),

  // Traces
  getTraces: (taskId: string) =>
    fetchJSON<Trace[]>(`${BASE}/tasks/${taskId}/traces`),

  // Wiki
  getWikiPages: (taskId: string, pageType?: string) =>
    fetchJSON<any[]>(`${BASE}/tasks/${taskId}/wiki${pageType ? `?page_type=${pageType}` : ''}`),

  getWikiStats: (taskId: string) =>
    fetchJSON<any>(`${BASE}/tasks/${taskId}/wiki/stats`),

  // SSE URL
  sseUrl: (taskId: string) => `${BASE}/tasks/${taskId}/events`,
};
