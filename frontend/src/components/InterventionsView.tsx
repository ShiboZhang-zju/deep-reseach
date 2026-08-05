import { useState, useEffect } from 'react';
import { api } from '../api/client';
import type { Intervention } from '../types';
import { formatDateTime } from '../utils/time';

interface Props {
  taskId: string;
  status: string;
}

const TIER_STYLE: Record<string, { label: string; cls: string }> = {
  A: { label: 'A · 可信', cls: 'bg-emerald-100 text-emerald-700' },
  B: { label: 'B · 待确认', cls: 'bg-amber-100 text-amber-700' },
  C: { label: 'C · 推测', cls: 'bg-gray-100 text-gray-500' },
};

function gateStyle(v: string): string {
  if (v === 'PASS') return 'bg-green-100 text-green-700';
  if (v === 'FAIL') return 'bg-red-100 text-red-700';
  if (v === 'WARN') return 'bg-amber-100 text-amber-700';
  return 'bg-gray-100 text-gray-500';
}

export function InterventionsView({ taskId, status }: Props) {
  const [items, setItems] = useState<Intervention[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api
      .getInterventions(taskId)
      .then(setItems)
      .catch(() => setItems([]))
      .finally(() => setLoading(false));
  }, [taskId]);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12 text-gray-400">
        <div className="animate-spin w-6 h-6 border-2 border-gray-300 border-t-blue-500 rounded-full mr-3" />
        加载干预方案...
      </div>
    );
  }

  if (items.length === 0) {
    return (
      <div className="text-center py-12 text-gray-400">
        <p className="text-lg mb-1">暂无干预方案</p>
        <p className="text-sm">
          {status === 'synthesizing_ideas' ? '正在基于存活的研究缺口生成干预方案...' : '当研究缺口通过审计后，系统会针对性生成干预方案'}
        </p>
      </div>
    );
  }

  const tierOrder: Record<string, number> = { A: 0, B: 1, C: 2 };
  const sorted = [...items].sort((a, b) => (tierOrder[a.confidence_tier] ?? 2) - (tierOrder[b.confidence_tier] ?? 2));

  return (
    <div className="space-y-4">
      <div className="text-sm text-gray-500">
        共 {items.length} 个干预方案，按置信度排序。每个方案标注证据/新颖性/可行性三道硬闸门结果。
      </div>
      {sorted.map((item, idx) => {
        const tier = TIER_STYLE[item.confidence_tier] || TIER_STYLE.C;
        return (
          <div key={item.id} className="bg-white rounded-lg border border-gray-200 p-5">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-base font-semibold text-gray-900">
                方案 {idx + 1}
                <span className="text-xs font-normal text-gray-400 ml-2">{item.intervention_type}</span>
              </h3>
              <div className="flex items-center gap-2">
                <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${tier.cls}`}>{tier.label}</span>
                <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${item.status === 'passed' ? 'bg-blue-100 text-blue-700' : 'bg-gray-100 text-gray-400'}`}>{item.status}</span>
              </div>
            </div>

            <p className="text-sm text-gray-800 mb-3">{item.proposed_intervention}</p>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-3">
              <Field label="针对的失效机制" value={item.failure_mechanism} />
              <Field label="预期中间效果" value={item.intermediate_effect} />
              <Field label="可测量结果" value={item.measurable_outcome} />
              {item.implementation_cost && <Field label="实现成本" value={item.implementation_cost} />}
            </div>

            <div className="flex items-center gap-2 mt-4 pt-3 border-t border-gray-100">
              <span className="text-xs text-gray-400 mr-1">硬闸门</span>
              <span className={`text-xs px-2 py-0.5 rounded ${gateStyle(item.evidence_gate)}`}>证据 {item.evidence_gate}</span>
              <span className={`text-xs px-2 py-0.5 rounded ${gateStyle(item.novelty_gate)}`}>新颖性 {item.novelty_gate}</span>
              <span className={`text-xs px-2 py-0.5 rounded ${gateStyle(item.feasibility_gate)}`}>可行性 {item.feasibility_gate}</span>
              <span className="ml-auto text-xs text-gray-400">{formatDateTime(item.created_at)}</span>
            </div>
          </div>
        );
      })}
    </div>
  );
}

function Field({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <span className="text-xs font-semibold text-gray-500 uppercase tracking-wider">{label}</span>
      <p className="text-sm mt-0.5 text-gray-700">{value}</p>
    </div>
  );
}
