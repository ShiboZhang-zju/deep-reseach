import { useState, useEffect, useCallback } from 'react';
import { api } from '../api/client';
import type { Task, Round } from '../types';
import { StatusBadge } from './StatusBadge';
import { formatDateTime } from '../utils/time';
import { EventLog } from './EventLog';
import { ReportView } from './ReportView';
import { PapersView } from './PapersView';
import { IdeasView } from './IdeasView';
import { GapsView } from './GapsView';
import { InterventionsView } from './InterventionsView';
import { CoverageView } from './CoverageView';
import { ExperimentsView } from './ExperimentsView';
import { TracesView } from './TracesView';
import { WikiView } from './WikiView';
import { ClarifyDialog } from './ClarifyDialog';
import { useTaskEvents } from '../hooks/useTaskEvents';

interface Props {
  taskId: string;
  onBack: () => void;
}

type Tab = 'overview' | 'report' | 'papers' | 'coverage' | 'gaps' | 'interventions' | 'ideas' | 'experiments' | 'traces' | 'wiki';

const TABS: { key: Tab; label: string; icon: string }[] = [
  { key: 'overview', label: '概览', icon: '📋' },
  { key: 'report', label: '研究报告', icon: '📄' },
  { key: 'papers', label: '论文', icon: '📚' },
  { key: 'wiki', label: '知识库', icon: '🧠' },
  { key: 'coverage', label: '覆盖度', icon: '📊' },
  { key: 'gaps', label: '研究缺口', icon: '🔍' },
  { key: 'interventions', label: '干预方案', icon: '🛠️' },
  { key: 'ideas', label: '创意', icon: '💡' },
  { key: 'experiments', label: '实验方案', icon: '🧪' },
  { key: 'traces', label: '执行轨迹', icon: '🔧' },
];

const RUNNING_STATUSES = [
  'clarifying',
  'building_contract',
  'decomposing',
  'searching',
  'summarizing',
  'reporting',
  'generating_ideas',
  'judging_ideas',
  'generating_experiment',
  'mining_gaps',
  'auditing_gaps',
  'synthesizing_ideas',
  'waiting_for_clarification',
  'waiting_for_user_review',
];

export function TaskDetail({ taskId, onBack }: Props) {
  const [task, setTask] = useState<Task | null>(null);
  const [rounds, setRounds] = useState<Round[]>([]);
  const [tab, setTab] = useState<Tab>('overview');
  const [submittingClarify, setSubmittingClarify] = useState(false);

  const { events, connected, clarificationQuestions, clearClarification } = useTaskEvents(taskId);

  // Reload task when SSE pushes a status event (reduces reliance on polling)
  const lastStatusEvent = events.filter(e => e.event === 'status').pop();
  useEffect(() => {
    if (lastStatusEvent) {
      loadTask();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [lastStatusEvent?.timestamp]);

  // Recover clarification questions from state_json if SSE missed them
  const [recoveredQuestions, setRecoveredQuestions] = useState<string[] | null>(null);
  const [clarifySubmitted, setClarifySubmitted] = useState(false);

  useEffect(() => {
    if (clarificationQuestions) {
      setRecoveredQuestions(null); // SSE has them, clear fallback
      setClarifySubmitted(false);  // New questions from SSE, reset flag
      return;
    }
    if (clarifySubmitted) return; // User just submitted, don't re-trigger
    if (!task || task.status !== 'waiting_for_clarification') return;
    if (!task.state_json) return;
    try {
      const state = JSON.parse(task.state_json);
      if (state.research_questions && state.research_questions.length > 0) {
        setRecoveredQuestions(state.research_questions);
      }
    } catch {
      // ignore
    }
  }, [task, clarificationQuestions, clarifySubmitted]);

  const activeQuestions = clarificationQuestions || recoveredQuestions;

  const loadTask = useCallback(async () => {
    try {
      const [t, r] = await Promise.all([api.getTask(taskId), api.getRounds(taskId)]);
      setTask(t);
      setRounds(r);
    } catch {
      // ignore
    }
  }, [taskId]);

  useEffect(() => {
    loadTask();
  }, [loadTask]);

  // Poll while running (fallback for SSE disconnection; SSE is primary)
  useEffect(() => {
    if (!task) return;
    if (!RUNNING_STATUSES.includes(task.status)) return;
    // 10s polling as SSE fallback (SSE pushes status events in real-time)
    const interval = setInterval(loadTask, 10000);
    return () => clearInterval(interval);
  }, [task, loadTask]);

  // Auto-switch tab based on status
  useEffect(() => {
    if (!task) return;
    // Reset clarifySubmitted when agent moves past clarification
    if (task.status !== 'waiting_for_clarification' && clarifySubmitted) {
      setClarifySubmitted(false);
    }
    if (task.status === 'reporting' || task.status === 'generating_ideas') {
      if (tab === 'overview') setTab('report');
    }
    if (task.status === 'waiting_for_user_review') {
      setTab('ideas');
    }
    if (task.status === 'done' && tab === 'overview') {
      setTab('report');
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [task?.status]);

  const handleClarifySubmit = async (answers: string[]) => {
    setSubmittingClarify(true);
    setClarifySubmitted(true);  // Prevent dialog from reappearing before agent restarts
    try {
      await api.submitClarification(taskId, answers);
      clearClarification();
      setRecoveredQuestions(null);
      await loadTask();
    } catch (e: any) {
      setClarifySubmitted(false);  // Reset on error so user can retry
      alert('提交失败: ' + e.message);
    } finally {
      setSubmittingClarify(false);
    }
  };

  const handleStop = async () => {
    if (!task) return;
    if (!confirm('确定要停止该任务吗？')) return;
    await api.stopTask(taskId);
    await loadTask();
  };

  if (!task) {
    return (
      <div className="flex items-center justify-center py-20 text-gray-400">
        <div className="animate-spin w-6 h-6 border-2 border-gray-300 border-t-blue-500 rounded-full mr-3" />
        加载中...
      </div>
    );
  }

  const isRunning = RUNNING_STATUSES.includes(task.status);

  return (
    <div className="max-w-6xl mx-auto p-6">
      {/* Header */}
      <div className="mb-5">
        <button onClick={onBack} className="text-sm text-gray-500 hover:text-gray-700 mb-3 flex items-center gap-1">
          ← 返回任务列表
        </button>
        <div className="bg-white rounded-xl border border-gray-200 p-4">
          <div className="flex items-start justify-between gap-4">
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-gray-900">{task.user_input}</p>
              {task.normalized_topic && (
                <p className="text-xs text-gray-500 mt-1">{task.normalized_topic}</p>
              )}
            </div>
            <div className="flex items-center gap-3 shrink-0">
              <StatusBadge status={task.status} />
              {isRunning && (
                <button
                  onClick={handleStop}
                  className="px-3 py-1 text-xs text-red-600 border border-red-300 rounded-lg hover:bg-red-50"
                >
                  停止
                </button>
              )}
            </div>
          </div>

          {/* Progress bar */}
          <div className="mt-3 flex items-center gap-4">
            <div className="flex-1">
              <div className="flex items-center justify-between text-xs text-gray-400 mb-1">
                <span>检索进度</span>
                <span>
                  {task.current_round} / {task.max_rounds} 轮
                </span>
              </div>
              <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
                <div
                  className="h-full bg-blue-500 rounded-full transition-all duration-500"
                  style={{ width: `${Math.min((task.current_round / task.max_rounds) * 100, 100)}%` }}
                />
              </div>
            </div>
            {task.stop_reason && (task.status === 'failed' || task.status === 'stopped') && (
              <span className="text-xs text-gray-400">终止原因: {task.stop_reason}</span>
            )}
          </div>
        </div>
      </div>

      {/* Clarification dialog */}
      {activeQuestions && activeQuestions.length > 0 && (
        <ClarifyDialog
          questions={activeQuestions}
          onSubmit={handleClarifySubmit}
          onClose={() => {
            clearClarification();
            setRecoveredQuestions(null);
          }}
        />
      )}

      {/* Tabs */}
      <div className="flex gap-1 mb-4 bg-white rounded-lg border border-gray-200 p-1">
        {TABS.map((t) => (
          <button
            key={t.key}
            onClick={() => setTab(t.key)}
            className={`flex-1 px-3 py-2 rounded-md text-sm font-medium transition-colors ${
              tab === t.key
                ? 'bg-blue-600 text-white'
                : 'text-gray-600 hover:bg-gray-50'
            }`}
          >
            <span className="mr-1">{t.icon}</span>
            {t.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      {tab === 'overview' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4" style={{ minHeight: '500px' }}>
          {/* Rounds summary */}
          <div className="space-y-4">
            <div className="bg-white rounded-lg border border-gray-200 p-4">
              <h3 className="text-sm font-semibold text-gray-700 mb-3">检索轮次</h3>
              {rounds.length === 0 ? (
                <p className="text-sm text-gray-400 text-center py-6">暂无轮次数据</p>
              ) : (
                <div className="space-y-2">
                  {rounds.map((r) => (
                    <div key={r.id} className="border border-gray-100 rounded-lg p-3">
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-sm font-medium text-gray-700">第 {r.round_number} 轮</span>
                        <span className="text-xs text-gray-400">
                          {formatDateTime(r.created_at)}
                        </span>
                      </div>
                      <div className="flex gap-4 text-xs text-gray-500">
                        <span>找到 {r.papers_found} 篇</span>
                        <span>新增 {r.new_papers} 篇</span>
                        {r.duplicate_rate !== null && (
                          <span>重复率 {(r.duplicate_rate * 100).toFixed(0)}%</span>
                        )}
                      </div>
                      {r.queries_json && (
                        <div className="mt-1.5 flex flex-wrap gap-1">
                          {JSON.parse(r.queries_json).map((q: string, i: number) => (
                            <span key={i} className="text-xs bg-gray-100 px-2 py-0.5 rounded text-gray-600">
                              {q}
                            </span>
                          ))}
                        </div>
                      )}
                      {r.summary && (
                        <p className="text-xs text-gray-500 mt-1.5 line-clamp-2">{r.summary}</p>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Event log */}
          <div style={{ minHeight: '400px' }}>
            <EventLog events={events} connected={connected} />
          </div>
        </div>
      )}

      {tab === 'report' && <ReportView taskId={taskId} status={task.status} />}
      {tab === 'papers' && <PapersView taskId={taskId} />}
      {tab === 'wiki' && <WikiView taskId={taskId} />}
      {tab === 'coverage' && <CoverageView taskId={taskId} status={task.status} />}
      {tab === 'gaps' && <GapsView taskId={taskId} status={task.status} />}
      {tab === 'interventions' && <InterventionsView taskId={taskId} status={task.status} />}
      {tab === 'ideas' && <IdeasView taskId={taskId} status={task.status} />}
      {tab === 'experiments' && <ExperimentsView taskId={taskId} status={task.status} />}
      {tab === 'traces' && <TracesView taskId={taskId} />}
    </div>
  );
}
