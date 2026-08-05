import { useState, useEffect } from 'react';
import { api } from '../api/client';
import type { Gap } from '../types';
import { formatDateTime } from '../utils/time';

interface Props {
  taskId: string;
  status: string;
}

const STATUS_STYLES: Record<string, string> = {
  surviving: 'bg-green-100 text-green-700',
  candidate: 'bg-blue-100 text-blue-700',
  rejected: 'bg-gray-100 text-gray-500',
  superseded: 'bg-gray-100 text-gray-400',
};

function tierBadge(provenance: string): { label: string; cls: string } {
  if (provenance === 'complete') {
    return { label: 'A · 全文支撑', cls: 'bg-emerald-100 text-emerald-700' };
  }
  return { label: 'B · 摘要级', cls: 'bg-amber-100 text-amber-700' };
}

export function GapsView({ taskId, status }: Props) {
  const [gaps, setGaps] = useState<Gap[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api
      .getGaps(taskId)
      .then(setGaps)
      .catch(() => setGaps([]))
      .finally(() => setLoading(false));
  }, [taskId]);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12 text-gray-400">
        <div className="animate-spin w-6 h-6 border-2 border-gray-300 border-t-blue-500 rounded-full mr-3" />
        加载研究缺口...
      </div>
    );
  }

  if (gaps.length === 0) {
    return (
      <div className="text-center py-12 text-gray-400">
        <p className="text-lg mb-1">暂无研究缺口</p>
        <p className="text-sm">
          {['mining_gaps', 'auditing_gaps'].includes(status)
            ? '缺口挖掘/审计进行中，请稍候...'
            : '当证据充分时，系统会挖掘并审计出证据支撑的研究缺口'}
        </p>
      </div>
    );
  }

  const surviving = gaps.filter((g) => g.status === 'surviving').length;

  return (
    <div className="space-y-4">
      <div className="text-sm text-gray-500">
        共 {gaps.length} 个缺口候选，其中 {surviving} 个通过审计存活。分级：A = 有全文证据可定位；B = 仅摘要级证据，需进一步确认。
      </div>
      {gaps.map((gap, idx) => {
        const tier = tierBadge(gap.provenance_status);
        const statusCls = STATUS_STYLES[gap.status] || 'bg-gray-100 text-gray-600';
        return (
          <div key={gap.id} className="bg-white rounded-lg border border-gray-200 p-5">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-base font-semibold text-gray-900">
                缺口 {idx + 1}
                <span className="text-xs font-normal text-gray-400 ml-2">{gap.gap_type}</span>
              </h3>
              <div className="flex items-center gap-2">
                <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${tier.cls}`}>{tier.label}</span>
                <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${statusCls}`}>{gap.status}</span>
              </div>
            </div>

            <p className="text-sm text-gray-800 mb-3">{gap.description}</p>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-3">
              {gap.target_setting && (
                <Field label="目标场景" value={gap.target_setting} />
              )}
              {gap.observed_problem && (
                <Field label="已观察到的问题" value={gap.observed_problem} />
              )}
              {gap.existing_coverage && (
                <Field label="现有工作覆盖" value={gap.existing_coverage} />
              )}
              {gap.missing_capability && (
                <Field label="缺失的能力" value={gap.missing_capability} />
              )}
              {gap.claimed_delta && (
                <Field label="声称的增量" value={gap.claimed_delta} />
              )}
              {gap.testable_hypothesis && (
                <Field label="可检验假设" value={gap.testable_hypothesis} highlight />
              )}
              {gap.falsification_condition && (
                <Field label="证伪条件" value={gap.falsification_condition} highlight />
              )}
            </div>

            <div className="flex items-center gap-4 mt-4 pt-3 border-t border-gray-100 text-xs text-gray-400">
              {gap.novelty_score != null && <span>新颖性 {gap.novelty_score.toFixed(2)}</span>}
              {gap.feasibility_score != null && <span>可行性 {gap.feasibility_score.toFixed(2)}</span>}
              {gap.significance_score != null && <span>重要性 {gap.significance_score.toFixed(2)}</span>}
              {gap.risk_score != null && <span>风险 {gap.risk_score.toFixed(2)}</span>}
              <span className="ml-auto">挖掘轮次 {gap.mining_round} · {formatDateTime(gap.created_at)}</span>
            </div>
          </div>
        );
      })}
    </div>
  );
}

function Field({ label, value, highlight }: { label: string; value: string; highlight?: boolean }) {
  return (
    <div>
      <span className="text-xs font-semibold text-gray-500 uppercase tracking-wider">{label}</span>
      <p className={`text-sm mt-0.5 ${highlight ? 'text-indigo-700 font-medium' : 'text-gray-700'}`}>{value}</p>
    </div>
  );
}
