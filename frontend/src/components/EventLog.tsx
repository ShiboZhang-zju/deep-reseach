import { useEffect, useRef } from 'react';
import type { SSEEvent } from '../hooks/useTaskEvents';
import { formatTime } from '../utils/time';

interface Props {
  events: SSEEvent[];
  connected: boolean;
}

const EVENT_LABELS: Record<string, string> = {
  status: '状态变更',
  clarification_needed: '需要澄清',
  round_start: '轮次开始',
  queries_generated: '查询生成',
  search_done: '检索完成',
  round_done: '轮次结束',
  stopping: '即将停止',
  report_ready: '报告就绪',
  idea_generated: '创意生成',
  ideas_judged: '创意评估',
  experiment_generated: '实验方案生成',
  error: '错误',
};

const EVENT_COLORS: Record<string, string> = {
  status: 'text-blue-600',
  clarification_needed: 'text-amber-600',
  round_start: 'text-indigo-600',
  queries_generated: 'text-cyan-600',
  search_done: 'text-teal-600',
  round_done: 'text-indigo-600',
  stopping: 'text-orange-600',
  report_ready: 'text-green-600',
  idea_generated: 'text-purple-600',
  ideas_judged: 'text-purple-600',
  experiment_generated: 'text-pink-600',
  error: 'text-red-600',
};

export function EventLog({ events, connected }: Props) {
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [events]);

  return (
    <div className="bg-gray-900 rounded-lg p-4 font-mono text-sm overflow-hidden flex flex-col h-full">
      <div className="flex items-center justify-between mb-3 shrink-0">
        <span className="text-gray-400 text-xs uppercase tracking-wider">实时事件流</span>
        <span className={`flex items-center gap-1.5 text-xs ${connected ? 'text-green-400' : 'text-gray-500'}`}>
          <span className={`w-2 h-2 rounded-full ${connected ? 'bg-green-400 animate-pulse' : 'bg-gray-600'}`} />
          {connected ? '已连接' : '未连接'}
        </span>
      </div>
      <div className="flex-1 overflow-y-auto space-y-1 min-h-0">
        {events.length === 0 && (
          <div className="text-gray-600 text-center py-8">等待事件...</div>
        )}
        {events.map((evt, i) => {
          const color = EVENT_COLORS[evt.event] || 'text-gray-400';
          const label = EVENT_LABELS[evt.event] || evt.event;
          const time = formatTime(evt.timestamp);
          let detail = '';
          if (evt.event === 'status') detail = evt.data.status;
          else if (evt.event === 'round_start' || evt.event === 'round_done')
            detail = `第 ${evt.data.round} 轮`;
          else if (evt.event === 'search_done')
            detail = `第 ${evt.data.round} 轮: ${evt.data.found} 篇`;
          else if (evt.event === 'queries_generated')
            detail = `第 ${evt.data.round} 轮: ${evt.data.queries?.join(' | ')}`;
          else if (evt.event === 'idea_generated')
            detail = evt.data.title;
          else if (evt.event === 'experiment_generated')
            detail = evt.data.title;
          else if (evt.event === 'error')
            detail = evt.data.message;
          else if (evt.event === 'clarification_needed')
            detail = `${evt.data.questions?.length} 个问题`;
          else if (evt.event === 'report_ready')
            detail = `${evt.data.length} 字符`;
          else if (evt.event === 'stopping')
            detail = evt.data.reason;

          return (
            <div key={i} className="flex gap-2 leading-relaxed">
              <span className="text-gray-600 shrink-0">{time}</span>
              <span className={`shrink-0 font-semibold ${color}`}>[{label}]</span>
              <span className="text-gray-300 break-all">{detail}</span>
            </div>
          );
        })}
        <div ref={endRef} />
      </div>
    </div>
  );
}
