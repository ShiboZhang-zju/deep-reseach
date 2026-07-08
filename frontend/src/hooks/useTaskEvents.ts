import { useEffect, useRef, useState, useCallback } from 'react';

export interface SSEEvent {
  event: string;
  data: any;
  timestamp: number;
}

const EVENT_TYPES = [
  'status',
  'clarification_needed',
  'round_start',
  'queries_generated',
  'search_done',
  'round_done',
  'stopping',
  'report_ready',
  'idea_generated',
  'ideas_judged',
  'experiment_generated',
  'error',
];

export function useTaskEvents(taskId: string | null) {
  const [events, setEvents] = useState<SSEEvent[]>([]);
  const [connected, setConnected] = useState(false);
  const [clarificationQuestions, setClarificationQuestions] = useState<string[] | null>(null);
  const eventSourceRef = useRef<EventSource | null>(null);

  useEffect(() => {
    if (!taskId) return;

    setEvents([]);
    setClarificationQuestions(null);
    setConnected(false);

    const es = new EventSource(`/api/tasks/${taskId}/events`);
    eventSourceRef.current = es;

    es.onopen = () => setConnected(true);
    es.onerror = () => setConnected(false);

    EVENT_TYPES.forEach((type) => {
      es.addEventListener(type, (e: MessageEvent) => {
        try {
          const data = JSON.parse(e.data);
          setEvents((prev) => [...prev, { event: type, data, timestamp: Date.now() }]);

          if (type === 'clarification_needed') {
            setClarificationQuestions(data.questions || []);
          }
        } catch {
          // ignore parse errors
        }
      });
    });

    return () => {
      es.close();
      eventSourceRef.current = null;
    };
  }, [taskId]);

  const clearClarification = useCallback(() => setClarificationQuestions(null), []);
  const clearEvents = useCallback(() => setEvents([]), []);

  return { events, connected, clarificationQuestions, clearClarification, clearEvents };
}
