import { useState, useEffect } from 'react';
import { api } from '../api/client';
import type { CoverageRecord } from '../types';

interface Props {
  taskId: string;
  status: string;
}

function scoreColor(score: number): string {
  if (score >= 0.7) return 'bg-green-500';
  if (score >= 0.4) return 'bg-amber-500';
  return 'bg-red-400';
}

export function CoverageView({ taskId, status }: Props) {
  const [records, setRecords] = useState<CoverageRecord[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api
      .getCoverage(taskId)
      .then(setRecords)
      .catch(() => setRecords([]))
      .finally(() => setLoading(false));
  }, [taskId]);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12 text-gray-400">
        <div className="animate-spin w-6 h-6 border-2 border-gray-300 border-t-blue-500 rounded-full mr-3" />
        加载覆盖度矩阵...
      </div>
    );
  }

  if (records.length === 0) {
    return (
      <div className="text-center py-12 text-gray-400">
        <p className="text-lg mb-1">暂无覆盖度数据</p>
        <p className="text-sm">检索与证据抽取过程中会为每个研究问题计算覆盖度</p>
      </div>
    );
  }

  // Keep only the latest record per question (highest round).
  const latestByQuestion = new Map<string, CoverageRecord>();
  for (const r of records) {
    const cur = latestByQuestion.get(r.question_id);
    if (!cur || r.round_number > cur.round_number) latestByQuestion.set(r.question_id, r);
  }
  const latest = Array.from(latestByQuestion.values()).sort((a, b) => b.coverage_score - a.coverage_score);
  const avg = latest.reduce((s, r) => s + r.coverage_score, 0) / latest.length;

  return (
    <div className="space-y-4">
      <div className="text-sm text-gray-500">
        共 {latest.length} 个研究问题，平均覆盖度 {(avg * 100).toFixed(0)}%。覆盖度反映每个问题被已抽取证据支撑的程度。
      </div>
      <div className="bg-white rounded-lg border border-gray-200 divide-y divide-gray-100">
        {latest.map((r, idx) => (
          <div key={r.id} className="p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-gray-700">问题 {idx + 1}</span>
              <span className="text-xs text-gray-400">
                第 {r.round_number} 轮{r.evidence_count != null ? ` · ${r.evidence_count} 条证据` : ''}
                {r.status ? ` · ${r.status}` : ''}
              </span>
            </div>
            <div className="flex items-center gap-3">
              <div className="flex-1 h-2 bg-gray-100 rounded-full overflow-hidden">
                <div className={`h-full ${scoreColor(r.coverage_score)}`} style={{ width: `${Math.round(r.coverage_score * 100)}%` }} />
              </div>
              <span className="text-sm font-medium text-gray-700 w-12 text-right">{(r.coverage_score * 100).toFixed(0)}%</span>
            </div>
          </div>
        ))}
      </div>
      {status === 'searching' && (
        <p className="text-xs text-gray-400">检索仍在进行，覆盖度会随每轮更新。</p>
      )}
    </div>
  );
}
