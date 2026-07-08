import { useState } from 'react';

interface Props {
  questions: string[];
  onSubmit: (answers: string[]) => void;
  onClose: () => void;
}

export function ClarifyDialog({ questions, onSubmit, onClose }: Props) {
  const [answers, setAnswers] = useState<string[]>(questions.map(() => ''));
  const [submitting, setSubmitting] = useState(false);

  const handleSubmit = async () => {
    setSubmitting(true);
    try {
      await onSubmit(answers);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-xl shadow-xl max-w-2xl w-full max-h-[80vh] overflow-y-auto">
        <div className="p-5 border-b border-gray-200">
          <h2 className="text-lg font-semibold text-gray-900">研究方向澄清</h2>
          <p className="text-sm text-gray-500 mt-1">请回答以下问题，帮助 AI 更好地理解你的研究需求</p>
        </div>
        <div className="p-5 space-y-4">
          {questions.map((q, i) => (
            <div key={i}>
              <label className="block text-sm font-medium text-gray-700 mb-1.5">
                {i + 1}. {q}
              </label>
              <textarea
                value={answers[i]}
                onChange={(e) => {
                  const next = [...answers];
                  next[i] = e.target.value;
                  setAnswers(next);
                }}
                rows={2}
                placeholder="输入你的回答..."
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none text-sm"
              />
            </div>
          ))}
        </div>
        <div className="p-5 border-t border-gray-200 flex justify-end gap-3">
          <button
            onClick={onClose}
            className="px-4 py-2 text-gray-600 hover:text-gray-800 text-sm"
          >
            跳过
          </button>
          <button
            onClick={handleSubmit}
            disabled={submitting}
            className="px-5 py-2 bg-blue-600 text-white rounded-lg text-sm font-medium hover:bg-blue-700 disabled:opacity-50"
          >
            {submitting ? '提交中...' : '提交回答'}
          </button>
        </div>
      </div>
    </div>
  );
}
