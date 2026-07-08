import { useState, useEffect, useCallback } from 'react';
import { api } from './api/client';
import type { Task } from './types';
import { TaskList } from './components/TaskList';
import { TaskDetail } from './components/TaskDetail';

export default function App() {
  const [tasks, setTasks] = useState<Task[]>([]);
  const [selectedTaskId, setSelectedTaskId] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const loadTasks = useCallback(async () => {
    try {
      const data = await api.listTasks(50);
      setTasks(data);
    } catch {
      setTasks([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadTasks();
  }, [loadTasks]);

  // Auto-refresh task list when no task is selected
  useEffect(() => {
    if (selectedTaskId) return;
    const interval = setInterval(loadTasks, 5000);
    return () => clearInterval(interval);
  }, [selectedTaskId, loadTasks]);

  if (selectedTaskId) {
    return <TaskDetail taskId={selectedTaskId} onBack={() => {
      setSelectedTaskId(null);
      loadTasks();
    }} />;
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen text-gray-400">
        <div className="animate-spin w-8 h-8 border-2 border-gray-300 border-t-blue-500 rounded-full mr-3" />
        加载中...
      </div>
    );
  }

  return <TaskList tasks={tasks} onSelect={setSelectedTaskId} onCreated={loadTasks} />;
}
