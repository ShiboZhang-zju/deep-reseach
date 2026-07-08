import { useState, useEffect } from 'react';
import { api } from '../api/client';
import type { Experiment, Idea } from '../types';
import ReactMarkdown from 'react-markdown';
import { formatDateTime } from '../utils/time';

interface Props {
  taskId: string;
  status: string;
}

export function ExperimentsView({ taskId, status }: Props) {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [ideas, setIdeas] = useState<Idea[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([api.getExperiments(taskId), api.getIdeas(taskId)])
      .then(([exps, ids]) => {
        setExperiments(exps);
        setIdeas(ids);
      })
      .catch(() => {
        setExperiments([]);
        setIdeas([]);
      })
      .finally(() => setLoading(false));
  }, [taskId]);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12 text-gray-400">
        <div className="animate-spin w-6 h-6 border-2 border-gray-300 border-t-blue-500 rounded-full mr-3" />
        加载实验方案...
      </div>
    );
  }

  if (experiments.length === 0) {
    return (
      <div className="text-center py-12 text-gray-400">
        <p className="text-lg mb-1">暂无实验方案</p>
        <p className="text-sm">
          {['judging_ideas', 'generating_experiment'].includes(status)
            ? '实验方案生成中，请稍候...'
            : '请先在 Ideas 页面选择并深度评估感兴趣的 Ideas'}
        </p>
      </div>
    );
  }

  const ideaMap = new Map(ideas.map((i) => [i.id, i]));

  return (
    <div className="space-y-4">
      {experiments.map((exp, idx) => {
        const idea = ideaMap.get(exp.idea_id);
        return (
          <div key={exp.id} className="bg-white rounded-lg border border-gray-200 p-5">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-base font-semibold text-gray-900">
                实验方案 {idx + 1}
                {idea?.title && <span className="text-sm font-normal text-gray-500 ml-2">— {idea.title}</span>}
              </h3>
              <span className="text-xs text-gray-400">
                {formatDateTime(exp.created_at)}
              </span>
            </div>

            <div className="space-y-3">
              {exp.hypothesis && (
                <div>
                  <span className="text-xs font-semibold text-gray-500 uppercase tracking-wider">假设</span>
                  <p className="text-sm text-gray-700 mt-0.5">{exp.hypothesis}</p>
                </div>
              )}
              {exp.dataset && (
                <div>
                  <span className="text-xs font-semibold text-gray-500 uppercase tracking-wider">数据集</span>
                  <p className="text-sm text-gray-700 mt-0.5">{exp.dataset}</p>
                </div>
              )}
              {exp.baselines && (
                <div>
                  <span className="text-xs font-semibold text-gray-500 uppercase tracking-wider">基线方法</span>
                  <p className="text-sm text-gray-700 mt-0.5">{exp.baselines}</p>
                </div>
              )}
              {exp.metrics && (
                <div>
                  <span className="text-xs font-semibold text-gray-500 uppercase tracking-wider">评估指标</span>
                  <p className="text-sm text-gray-700 mt-0.5">{exp.metrics}</p>
                </div>
              )}
              {exp.steps_markdown && (
                <div>
                  <span className="text-xs font-semibold text-gray-500 uppercase tracking-wider">实验步骤</span>
                  <div className="markdown-body text-sm mt-1">
                    <ReactMarkdown>{exp.steps_markdown}</ReactMarkdown>
                  </div>
                </div>
              )}
              {exp.risks && (
                <div className="bg-red-50 rounded-lg p-3">
                  <span className="text-xs font-semibold text-red-600 uppercase tracking-wider">风险与注意事项</span>
                  <p className="text-sm text-red-700 mt-0.5">{exp.risks}</p>
                </div>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}
