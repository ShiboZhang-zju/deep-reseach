import { useState, useEffect } from 'react';
import { api } from '../api/client';
import type { Trace } from '../types';

interface Props {
  taskId: string;
}

const STEP_TYPE_COLORS: Record<string, string> = {
  llm: 'bg-blue-100 text-blue-700',
  search: 'bg-cyan-100 text-cyan-700',
  scoring: 'bg-amber-100 text-amber-700',
  logic: 'bg-gray-100 text-gray-600',
  io: 'bg-green-100 text-green-700',
};

const STEP_TYPE_LABELS: Record<string, string> = {
  llm: 'LLM',
  search: '检索',
  scoring: '评分',
  logic: '逻辑',
  io: 'I/O',
  action: '操作',
};

export function TracesView({ taskId }: Props) {
  const [traces, setTraces] = useState<Trace[]>([]);
  const [loading, setLoading] = useState(true);
  const [expanded, setExpanded] = useState<string | null>(null);

  useEffect(() => {
    api
      .getTraces(taskId)
      .then(setTraces)
      .catch(() => setTraces([]))
      .finally(() => setLoading(false));
  }, [taskId]);

  const formatJson = (json: string | null): string => {
    if (!json) return '';
    try {
      return JSON.stringify(JSON.parse(json), null, 2);
    } catch {
      return json;
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12 text-gray-400">
        <div className="animate-spin w-6 h-6 border-2 border-gray-300 border-t-blue-500 rounded-full mr-3" />
        加载执行轨迹...
      </div>
    );
  }

  if (traces.length === 0) {
    return <div className="text-center py-12 text-gray-400">暂无执行轨迹</div>;
  }

  return (
    <div className="space-y-2">
      {traces.map((trace, idx) => {
        const isExpanded = expanded === trace.id;
        const typeColor = STEP_TYPE_COLORS[trace.step_type] || 'bg-gray-100 text-gray-600';
        return (
          <div key={trace.id} className="bg-white rounded-lg border border-gray-200 overflow-hidden">
            <div
              className="p-3 cursor-pointer hover:bg-gray-50 flex items-center gap-3"
              onClick={() => setExpanded(isExpanded ? null : trace.id)}
            >
              <span className="text-xs text-gray-400 font-mono w-8 shrink-0">#{idx + 1}</span>
              <span className={`px-2 py-0.5 rounded text-xs font-medium shrink-0 ${typeColor}`}>
                {STEP_TYPE_LABELS[trace.step_type] || trace.step_type}
              </span>
              <span className="text-sm text-gray-700 flex-1 truncate">{trace.step_name}</span>
              {trace.round_number !== null && (
                <span className="text-xs text-gray-400 shrink-0">R{trace.round_number}</span>
              )}
              {trace.duration_ms !== null && (
                <span className="text-xs text-gray-400 shrink-0">{(trace.duration_ms / 1000).toFixed(1)}秒</span>
              )}
              {trace.llm_tokens_used !== null && (
                <span className="text-xs text-gray-400 shrink-0">{trace.llm_tokens_used} 令牌</span>
              )}
            </div>
            {isExpanded && (
              <div className="border-t border-gray-100 p-3 space-y-3">
                {trace.input_json && (
                  <div>
                    <span className="text-xs font-semibold text-gray-500">输入</span>
                    <pre className="text-xs bg-gray-900 text-gray-100 p-3 rounded mt-1 overflow-x-auto">
                      {formatJson(trace.input_json)}
                    </pre>
                  </div>
                )}
                {trace.output_json && (
                  <div>
                    <span className="text-xs font-semibold text-gray-500">输出</span>
                    <pre className="text-xs bg-gray-900 text-gray-100 p-3 rounded mt-1 overflow-x-auto">
                      {formatJson(trace.output_json)}
                    </pre>
                  </div>
                )}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
