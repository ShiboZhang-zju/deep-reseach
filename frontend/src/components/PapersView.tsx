import { useState, useEffect } from 'react';
import { api } from '../api/client';
import type { Paper } from '../types';

interface Props {
  taskId: string;
}

const PRIORITY_TABS = [
  { key: '', label: '全部' },
  { key: 'high', label: '高优先级' },
  { key: 'medium', label: '中优先级' },
  { key: 'low', label: '低优先级' },
  { key: 'null', label: '未评分' },
];

export function PapersView({ taskId }: Props) {
  const [papers, setPapers] = useState<Paper[]>([]);
  const [loading, setLoading] = useState(true);
  const [priority, setPriority] = useState('');
  const [expanded, setExpanded] = useState<string | null>(null);
  const [visibleCount, setVisibleCount] = useState(50);  // incremental rendering

  useEffect(() => {
    setLoading(true);
    setVisibleCount(50);  // reset on filter change
    api
      .getPapers(taskId, priority || undefined, 200, 0)
      .then(setPapers)
      .catch(() => setPapers([]))
      .finally(() => setLoading(false));
  }, [taskId, priority]);

  const visiblePapers = papers.slice(0, visibleCount);
  const hasMore = visibleCount < papers.length;

  const parseAuthors = (json: string | null): string[] => {
    if (!json) return [];
    try {
      return JSON.parse(json);
    } catch {
      return [];
    }
  };

  const scoreColor = (score: number | null): string => {
    if (score === null) return 'text-gray-400';
    if (score >= 0.75) return 'text-green-600 font-semibold';
    if (score >= 0.5) return 'text-amber-600 font-semibold';
    return 'text-red-500';
  };

  const priorityBadge = (p: string | null): string => {
    if (p === 'high') return 'bg-green-100 text-green-700';
    if (p === 'medium') return 'bg-amber-100 text-amber-700';
    if (p === 'low') return 'bg-gray-100 text-gray-600';
    return 'bg-gray-50 text-gray-400';
  };

  return (
    <div>
      {/* Priority filter tabs */}
      <div className="flex gap-2 mb-4">
        {PRIORITY_TABS.map((tab) => (
          <button
            key={tab.key}
            onClick={() => setPriority(tab.key)}
            className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
              priority === tab.key
                ? 'bg-blue-600 text-white'
                : 'bg-white text-gray-600 border border-gray-200 hover:bg-gray-50'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {loading ? (
        <div className="flex items-center justify-center py-12 text-gray-400">
          <div className="animate-spin w-6 h-6 border-2 border-gray-300 border-t-blue-500 rounded-full mr-3" />
          加载论文列表...
        </div>
      ) : papers.length === 0 ? (
        <div className="text-center py-12 text-gray-400">暂无论文数据</div>
      ) : (
        <div className="space-y-2">
          {visiblePapers.map((paper) => (
            <div
              key={paper.id}
              className="bg-white rounded-lg border border-gray-200 overflow-hidden"
            >
              <div
                className="p-4 cursor-pointer hover:bg-gray-50"
                onClick={() => setExpanded(expanded === paper.id ? null : paper.id)}
              >
                <div className="flex items-start justify-between gap-3">
                  <div className="flex-1 min-w-0">
                    <h3 className="text-sm font-medium text-gray-900 line-clamp-2">
                      {paper.title}
                    </h3>
                    <p className="text-xs text-gray-500 mt-1">
                      {parseAuthors(paper.authors_json).slice(0, 3).join(', ')}
                      {parseAuthors(paper.authors_json).length > 3 && ' et al.'}
                      {paper.year && ` · ${paper.year}`}
                      {paper.venue && ` · ${paper.venue}`}
                    </p>
                    {paper.summary && expanded !== paper.id && (
                      <p className="text-xs text-gray-400 mt-1 line-clamp-1">{paper.summary}</p>
                    )}
                  </div>
                  <div className="flex flex-col items-end gap-1 shrink-0">
                    {paper.priority && (
                      <span className={`px-2 py-0.5 rounded text-xs font-medium ${priorityBadge(paper.priority)}`}>
                        {paper.priority === 'high' ? '高' : paper.priority === 'medium' ? '中' : '低'}
                      </span>
                    )}
                    <span className={`text-sm ${scoreColor(paper.final_score)}`}>
                      {paper.final_score !== null ? paper.final_score.toFixed(2) : '-'}
                    </span>
                  </div>
                </div>
              </div>
              {expanded === paper.id && (
                <div className="px-4 pb-4 border-t border-gray-100 pt-3 space-y-2">
                  {paper.abstract && (
                    <div>
                      <span className="text-xs font-semibold text-gray-500">摘要</span>
                      <p className="text-sm text-gray-700 mt-1 leading-relaxed">{paper.abstract}</p>
                    </div>
                  )}
                  {paper.summary && (
                    <div>
                      <span className="text-xs font-semibold text-gray-500">AI 评估</span>
                      <p className="text-sm text-gray-700 mt-1">{paper.summary}</p>
                    </div>
                  )}
                  {paper.reason && (
                    <div>
                      <span className="text-xs font-semibold text-gray-500">评分理由</span>
                      <p className="text-sm text-gray-600 mt-1">{paper.reason}</p>
                    </div>
                  )}
                  <div className="flex flex-wrap gap-3 pt-1 text-xs text-gray-400">
                    {paper.doi && <span>DOI: {paper.doi}</span>}
                    {paper.arxiv_id && <span>arXiv: {paper.arxiv_id}</span>}
                    {paper.citation_count > 0 && <span>引用: {paper.citation_count}</span>}
                    {paper.url && (
                      <a href={paper.url} target="_blank" rel="noopener noreferrer" className="text-blue-500 hover:underline">
                        原文链接 →
                      </a>
                    )}
                  </div>
                </div>
              )}
            </div>
          ))}
          {hasMore && (
            <div className="text-center py-4">
              <button
                onClick={() => setVisibleCount((c) => c + 50)}
                className="px-4 py-2 text-sm text-blue-600 border border-blue-300 rounded-lg hover:bg-blue-50"
              >
                加载更多（剩余 {papers.length - visibleCount} 篇）
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
