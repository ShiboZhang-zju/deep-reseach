import { useState, useEffect, useCallback } from 'react';
import { api } from '../api/client';
import type { Report } from '../types';
import ReactMarkdown from 'react-markdown';
import { formatDateTime } from '../utils/time';

interface Props {
  taskId: string;
  status: string;
}

export function ReportView({ taskId, status }: Props) {
  const [report, setReport] = useState<Report | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const r = await api.getReport(taskId);
      setReport(r);
      setError(null);
    } catch (e: any) {
      if (e.message?.includes('404')) {
        setReport(null);
      } else {
        setError(e.message);
      }
    } finally {
      setLoading(false);
    }
  }, [taskId]);

  useEffect(() => {
    load();
  }, [load]);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12 text-gray-400">
        <div className="animate-spin w-6 h-6 border-2 border-gray-300 border-t-blue-500 rounded-full mr-3" />
        加载报告中...
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-center py-12">
        <p className="text-red-500 mb-3">{error}</p>
        <button onClick={load} className="text-blue-600 hover:underline text-sm">重试</button>
      </div>
    );
  }

  if (!report) {
    return (
      <div className="text-center py-12 text-gray-400">
        <p className="text-lg mb-1">报告尚未生成</p>
        {['searching', 'clarifying', 'summarizing', 'reporting'].includes(status) ? (
          <p className="text-sm">研究进行中，请稍候...</p>
        ) : (
          <button onClick={load} className="text-blue-600 hover:underline text-sm mt-2">刷新</button>
        )}
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-gray-900">研究报告</h2>
        <span className="text-xs text-gray-400">
          {formatDateTime(report.created_at)}
        </span>
      </div>
      <div className="markdown-body text-sm">
        <ReactMarkdown>{report.content_markdown}</ReactMarkdown>
      </div>
    </div>
  );
}
