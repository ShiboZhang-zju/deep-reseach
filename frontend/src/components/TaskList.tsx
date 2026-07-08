import { useState } from 'react';
import { api } from '../api/client';
import type { Task } from '../types';
import { StatusBadge } from './StatusBadge';
import { formatDateTime } from '../utils/time';

interface Props {
  tasks: Task[];
  onSelect: (taskId: string) => void;
  onCreated: () => void;
}

export function TaskList({ tasks, onSelect, onCreated }: Props) {
  const [input, setInput] = useState('');
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleCreate = async () => {
    if (input.trim().length < 2) return;
    setCreating(true);
    setError(null);
    try {
      const task = await api.createTask(input.trim());
      await api.startTask(task.id);
      setInput('');
      onCreated();
      onSelect(task.id);
    } catch (e: any) {
      setError(e.message || '创建失败');
    } finally {
      setCreating(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto p-6">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-1">Deep Research</h1>
        <p className="text-gray-500">AI 驱动的学术论文深度研究助手</p>
      </div>

      {/* Create form */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-5 mb-6">
        <label className="block text-sm font-medium text-gray-700 mb-2">创建新的研究任务</label>
        <div className="flex gap-3">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="描述你的研究方向，例如：基于大语言模型的Python代码自动化测试Oracle生成"
            rows={2}
            className="flex-1 px-4 py-2.5 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none text-sm"
            onKeyDown={(e) => {
              if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) handleCreate();
            }}
          />
          <button
            onClick={handleCreate}
            disabled={creating || input.trim().length < 2}
            className="px-5 py-2.5 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors whitespace-nowrap"
          >
            {creating ? '创建中...' : '开始研究'}
          </button>
        </div>
        {error && <p className="mt-2 text-sm text-red-600">{error}</p>}
        <p className="mt-2 text-xs text-gray-400">Ctrl + Enter 快速提交</p>
      </div>

      {/* Task list */}
      <div className="space-y-3">
        {tasks.length === 0 && (
          <div className="text-center py-12 text-gray-400">
            <p className="text-lg">暂无研究任务</p>
            <p className="text-sm mt-1">在上方输入研究方向开始</p>
          </div>
        )}
        {tasks.map((task) => (
          <div
            key={task.id}
            onClick={() => onSelect(task.id)}
            className="bg-white rounded-xl shadow-sm border border-gray-200 p-4 hover:shadow-md hover:border-blue-300 cursor-pointer transition-all"
          >
            <div className="flex items-start justify-between gap-4">
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-gray-900 line-clamp-2">{task.user_input}</p>
                {task.normalized_topic && (
                  <p className="text-xs text-gray-500 mt-1 truncate">{task.normalized_topic}</p>
                )}
              </div>
              <div className="flex flex-col items-end gap-1.5 shrink-0">
                <StatusBadge status={task.status} />
                <span className="text-xs text-gray-400">
                  {formatDateTime(task.created_at)}
                </span>
              </div>
            </div>
            {task.current_round > 0 && (
              <div className="mt-2 flex items-center gap-3 text-xs text-gray-400">
                <span>轮次 {task.current_round}/{task.max_rounds}</span>
                {task.stop_reason && <span>· {task.stop_reason}</span>}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
