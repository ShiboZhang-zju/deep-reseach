import { useState, useEffect } from 'react';
import { api } from '../api/client';

interface WikiPage {
  id: string;
  page_type: string;
  title: string;
  content_markdown: string;
  paper_ids: string[];
  links: string[];
  contradictions: string[];
  created_at: string;
  updated_at: string;
}

interface WikiStats {
  total_pages: number;
  by_type: Record<string, number>;
  contradictions: number;
}

const TYPE_LABELS: Record<string, string> = {
  concept: '研究概念',
  method: '方法',
  dataset: '数据集',
  model: '模型',
  synthesis: '综合分析',
};

const TYPE_COLORS: Record<string, string> = {
  concept: 'bg-blue-100 text-blue-700 border-blue-200',
  method: 'bg-green-100 text-green-700 border-green-200',
  dataset: 'bg-purple-100 text-purple-700 border-purple-200',
  model: 'bg-orange-100 text-orange-700 border-orange-200',
  synthesis: 'bg-pink-100 text-pink-700 border-pink-200',
};

export function WikiView({ taskId }: { taskId: string }) {
  const [pages, setPages] = useState<WikiPage[]>([]);
  const [stats, setStats] = useState<WikiStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<string>('all');
  const [expandedPage, setExpandedPage] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([
      api.getWikiPages(taskId),
      api.getWikiStats(taskId),
    ]).then(([p, s]) => {
      setPages(p);
      setStats(s);
      setLoading(false);
    }).catch(() => setLoading(false));
  }, [taskId]);

  if (loading) {
    return <div className="text-center py-8 text-gray-400">加载中...</div>;
  }

  if (pages.length === 0) {
    return (
      <div className="text-center py-12">
        <p className="text-gray-400 text-lg">Wiki 尚未构建</p>
        <p className="text-gray-300 text-sm mt-2">
          研究任务完成后，LLM Wiki 会自动从论文中编译结构化知识页面
        </p>
      </div>
    );
  }

  const filteredPages = filter === 'all'
    ? pages
    : pages.filter(p => p.page_type === filter);

  const typeCounts: Record<string, number> = {};
  pages.forEach(p => {
    typeCounts[p.page_type] = (typeCounts[p.page_type] || 0) + 1;
  });

  return (
    <div className="space-y-4">
      {/* Stats bar */}
      {stats && (
        <div className="flex flex-wrap items-center gap-3 bg-white rounded-lg border border-gray-200 px-4 py-3">
          <span className="text-sm font-semibold text-gray-700">
            {stats.total_pages} 个 Wiki 页面
          </span>
          {stats.contradictions > 0 && (
            <span className="text-sm text-red-600">
              {stats.contradictions} 个矛盾
            </span>
          )}
          <div className="flex gap-2 ml-auto">
            <button
              onClick={() => setFilter('all')}
              className={`text-xs px-3 py-1 rounded-full border transition ${
                filter === 'all'
                  ? 'bg-gray-700 text-white border-gray-700'
                  : 'bg-white text-gray-600 border-gray-200 hover:border-gray-300'
              }`}
            >
              全部 ({pages.length})
            </button>
            {Object.entries(typeCounts).map(([type, count]) => (
              <button
                key={type}
                onClick={() => setFilter(type)}
                className={`text-xs px-3 py-1 rounded-full border transition ${
                  filter === type
                    ? TYPE_COLORS[type] || 'bg-gray-100 text-gray-700 border-gray-300'
                    : 'bg-white text-gray-600 border-gray-200 hover:border-gray-300'
                }`}
              >
                {TYPE_LABELS[type] || type} ({count})
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Wiki pages */}
      <div className="space-y-3">
        {filteredPages.map(page => (
          <div
            key={page.id}
            className="bg-white rounded-lg border border-gray-200 overflow-hidden"
          >
            {/* Page header */}
            <div
              className="flex items-center gap-3 px-4 py-3 cursor-pointer hover:bg-gray-50 transition"
              onClick={() => setExpandedPage(expandedPage === page.id ? null : page.id)}
            >
              <span className={`text-xs px-2 py-0.5 rounded-full border ${
                TYPE_COLORS[page.page_type] || 'bg-gray-100 text-gray-600 border-gray-200'
              }`}>
                {TYPE_LABELS[page.page_type] || page.page_type}
              </span>
              <span className="text-sm font-medium text-gray-800 flex-1">
                {page.title}
              </span>
              <span className="text-xs text-gray-400">
                {page.paper_ids.length} 篇论文
              </span>
              {page.contradictions.length > 0 && (
                <span className="text-xs text-red-500">
                  {page.contradictions.length} 个矛盾
                </span>
              )}
              <span className="text-gray-400 text-xs">
                {expandedPage === page.id ? '▼' : '▶'}
              </span>
            </div>

            {/* Page content (expandable) */}
            {expandedPage === page.id && (
              <div className="border-t border-gray-100 px-4 py-3">
                <div className="prose prose-sm max-w-none">
                  <pre className="whitespace-pre-wrap text-sm text-gray-700 font-sans leading-relaxed">
                    {page.content_markdown}
                  </pre>
                </div>

                {/* Links */}
                {page.links.length > 0 && (
                  <div className="mt-4 pt-3 border-t border-gray-50">
                    <span className="text-xs text-gray-400">交叉引用: </span>
                    {page.links.map((link, i) => (
                      <span key={i} className="text-xs text-blue-500 mr-2">
                        [[{link}]]
                      </span>
                    ))}
                  </div>
                )}

                {/* Contradictions */}
                {page.contradictions.length > 0 && (
                  <div className="mt-3 pt-3 border-t border-gray-50">
                    <span className="text-xs text-red-500 font-semibold">发现的矛盾:</span>
                    <ul className="mt-1 space-y-1">
                      {page.contradictions.map((c, i) => (
                        <li key={i} className="text-xs text-red-600">
                          {c}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
