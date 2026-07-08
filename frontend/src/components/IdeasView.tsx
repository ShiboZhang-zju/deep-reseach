import { useState, useEffect } from 'react';
import { api } from '../api/client';
import type { Idea } from '../types';

interface Props {
  taskId: string;
  status: string;
}

export function IdeasView({ taskId, status }: Props) {
  const [ideas, setIdeas] = useState<Idea[]>([]);
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [submitting, setSubmitting] = useState(false);
  const [judging, setJudging] = useState(false);
  const [expanded, setExpanded] = useState<string | null>(null);

  const load = async () => {
    setLoading(true);
    try {
      const data = await api.getIdeas(taskId);
      setIdeas(data);
      setSelected(new Set(data.filter((i) => i.user_selected).map((i) => i.id)));
    } catch {
      setIdeas([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
    // Reset judging when status changes (e.g. done, or back to waiting_for_user_review)
    if (status !== 'judging_ideas' && status !== 'generating_experiment') {
      setJudging(false);
    }
  }, [taskId, status]);

  const toggleSelect = (id: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const handleSelectAndJudge = async () => {
    if (selected.size === 0) return;
    setSubmitting(true);
    try {
      await api.selectIdeas(taskId, [...selected]);
      await api.judgeIdeas(taskId);  // Now returns immediately
      setJudging(true);
      setSubmitting(false);
      // Don't load() immediately — TaskDetail polling will detect status change
      // and this component will reload when status prop changes (see useEffect below)
    } catch (e: any) {
      setSubmitting(false);
      alert('提交失败: ' + e.message);
    }
  };

  const scoreColor = (score: number | null): string => {
    if (score === null) return 'text-gray-400';
    if (score >= 0.75) return 'text-green-600';
    if (score >= 0.55) return 'text-amber-600';
    return 'text-red-500';
  };

  const decisionBadge = (decision: string | null): { label: string; class: string } => {
    if (decision === 'go') return { label: '推荐', class: 'bg-green-100 text-green-700' };
    if (decision === 'revise') return { label: '需改进', class: 'bg-amber-100 text-amber-700' };
    if (decision === 'no_go') return { label: '不推荐', class: 'bg-red-100 text-red-700' };
    return { label: '待评估', class: 'bg-gray-100 text-gray-500' };
  };

  const selectableIdeas = ideas.filter((i) => i.decision === 'go');
  const hasGoIdeas = selectableIdeas.length > 0;
  const showSelectUI = ['waiting_for_user_review'].includes(status);
  const showJudgeUI = ideas.some((i) => i.final_score !== null) && ideas.some((i) => i.final_score === null);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12 text-gray-400">
        <div className="animate-spin w-6 h-6 border-2 border-gray-300 border-t-blue-500 rounded-full mr-3" />
        加载创意...
      </div>
    );
  }

  if (ideas.length === 0) {
    return (
      <div className="text-center py-12 text-gray-400">
        <p className="text-lg mb-1">暂无创意</p>
        <p className="text-sm">
          {['generating_ideas', 'searching', 'reporting'].includes(status)
            ? '研究进行中，创意将在报告生成后产生...'
            : '需先完成研究报告'}
        </p>
      </div>
    );
  }

  return (
    <div>
      {showSelectUI && (
        <div className="bg-amber-50 border border-amber-200 rounded-lg p-4 mb-4">
          <div className="flex items-center justify-between">
            <div>
              {hasGoIdeas ? (
                <>
                  <p className="text-sm font-medium text-amber-800">
                    请选择 {selected.size} 个感兴趣的创意进行深度评估和实验方案生成
                  </p>
                  <p className="text-xs text-amber-600 mt-0.5">
                    已选择 {selected.size} / {selectableIdeas.length} 推荐（{ideas.length - selectableIdeas.length} 个分数不足不可选）
                  </p>
                </>
              ) : (
                <>
                  <p className="text-sm font-medium text-amber-800">
                    当前无高质量创意（≥0.70），系统已完成自动优化
                  </p>
                  <p className="text-xs text-amber-600 mt-0.5">
                    所有 {ideas.length} 个创意分数均低于0.70。系统已尝试3轮优化，当前展示的是最佳创意。
                  </p>
                </>
              )}
            </div>
            {hasGoIdeas && (
              <button
                onClick={handleSelectAndJudge}
                disabled={selected.size === 0 || submitting || judging}
                className="px-4 py-2 bg-amber-600 text-white rounded-lg text-sm font-medium hover:bg-amber-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {submitting ? '提交中...' : judging ? '深度评估中...' : '提交并深度评估'}
              </button>
            )}
          </div>
        </div>
      )}

      <div className="space-y-3">
        {ideas.map((idea, idx) => {
          const dec = decisionBadge(idea.decision);
          const isExpanded = expanded === idea.id;
          return (
            <div
              key={idea.id}
              className={`bg-white rounded-lg border-2 transition-colors ${
                selected.has(idea.id) ? 'border-amber-300' : 'border-gray-200'
              }`}
            >
              <div className="p-4">
                <div className="flex items-start gap-3">
                  {showSelectUI && (() => {
                    const isGo = idea.decision === 'go';
                    return isGo ? (
                      <input
                        type="checkbox"
                        checked={selected.has(idea.id)}
                        onChange={() => toggleSelect(idea.id)}
                        className="mt-1 w-4 h-4 rounded accent-amber-600"
                        title="推荐创意"
                      />
                    ) : (
                      <span className="mt-1 w-4 h-4 flex items-center justify-center text-gray-300 text-xs" title="分数不足0.70，不可选择">
                        🔒
                      </span>
                    );
                  })()}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-start justify-between gap-2">
                      <h3 className="text-sm font-semibold text-gray-900">
                        {idx + 1}. {idea.title || '未命名'}
                      </h3>
                      <div className="flex items-center gap-2 shrink-0">
                        <span className={`px-2 py-0.5 rounded text-xs font-medium ${dec.class}`}>
                          {dec.label}
                        </span>
                        {idea.final_score !== null && (
                          <span className={`text-sm ${scoreColor(idea.final_score)}`}>
                            {idea.final_score.toFixed(2)}
                          </span>
                        )}
                      </div>
                    </div>
                    {idea.description && (
                      <p className={`text-sm text-gray-600 mt-1 ${isExpanded ? '' : 'line-clamp-2'}`}>
                        {idea.description}
                      </p>
                    )}
                  </div>
                </div>

                {isExpanded && (
                  <div className="mt-3 ml-7 space-y-2 text-sm">
                    {idea.motivation && (
                      <div>
                        <span className="text-xs font-semibold text-gray-500">动机</span>
                        <p className="text-gray-700">{idea.motivation}</p>
                      </div>
                    )}
                    {idea.method_sketch && (
                      <div>
                        <span className="text-xs font-semibold text-gray-500">方法概述</span>
                        <p className="text-gray-700">{idea.method_sketch}</p>
                      </div>
                    )}
                    {idea.expected_contribution && (
                      <div>
                        <span className="text-xs font-semibold text-gray-500">预期贡献</span>
                        <p className="text-gray-700">{idea.expected_contribution}</p>
                      </div>
                    )}
                    {(() => {
                      let paperIds: string[] = [];
                      try {
                        paperIds = JSON.parse(idea.related_paper_ids_json || '[]');
                      } catch { /* ignore */ }
                      if (paperIds.length === 0) return null;
                      return (
                        <div>
                          <span className="text-xs font-semibold text-gray-500">关联论文</span>
                          <div className="flex flex-wrap gap-1 mt-1">
                            {paperIds.map((pid: string, i: number) => (
                              <span key={i} className="px-2 py-0.5 bg-blue-50 text-blue-600 rounded text-xs font-mono">
                                {pid.substring(0, 8)}
                              </span>
                            ))}
                          </div>
                        </div>
                      );
                    })()}
                    {idea.final_score !== null && (
                      <div className="grid grid-cols-4 gap-2 pt-2">
                        {[
                          { label: '新颖性', val: idea.novelty },
                          { label: '可行性', val: idea.feasibility },
                          { label: '重要性', val: idea.significance },
                          { label: '证据支撑', val: idea.evidence_support },
                          { label: '差异化', val: idea.differentiation },
                          { label: '可实验性', val: idea.experimentability },
                          { label: '潜在影响', val: idea.potential_impact },
                          { label: '风险', val: idea.risk },
                        ].map((m) => (
                          <div key={m.label} className="text-center bg-gray-50 rounded p-1.5">
                            <div className="text-xs text-gray-500">{m.label}</div>
                            <div className={`text-sm font-medium ${scoreColor(m.val ?? null)}`}>
                              {m.val !== null ? m.val.toFixed(2) : '-'}
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}

                <button
                  onClick={() => setExpanded(isExpanded ? null : idea.id)}
                  className="ml-7 mt-2 text-xs text-blue-500 hover:underline"
                >
                  {isExpanded ? '收起' : '展开详情'}
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
