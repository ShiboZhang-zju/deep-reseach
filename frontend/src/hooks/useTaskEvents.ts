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

// P2-12: SSE reconnection with exponential backoff
const MAX_RECONNECT_ATTEMPTS = 5;
const BASE_RECONNECT_DELAY = 1000; // 1s
const MAX_RECONNECT_DELAY = 30000; // 30s

export function useTaskEvents(taskId: string | null) {
  const [events, setEvents] = useState<SSEEvent[]>([]);
  const [connected, setConnected] = useState(false);
  const [clarificationQuestions, setClarificationQuestions] = useState<string[] | null>(null);
  const eventSourceRef = useRef<EventSource | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Track terminal statuses to stop reconnecting once task is done
  const isTerminalRef = useRef(false);

  const connect = useCallback((tid: string) => {
    // Don't reconnect if task already reached terminal state
    if (isTerminalRef.current) return;

    const es = new EventSource(`/api/tasks/${tid}/events`);
    eventSourceRef.current = es;

    es.onopen = () => {
      setConnected(true);
      reconnectAttemptsRef.current = 0; // reset backoff on successful connect
    };

    es.onerror = () => {
      setConnected(false);
      es.close();
      eventSourceRef.current = null;

      // Don't reconnect if task is terminal
      if (isTerminalRef.current) return;

      // Exponential backoff reconnect
      const attempt = reconnectAttemptsRef.current;
      if (attempt >= MAX_RECONNECT_ATTEMPTS) {
        console.warn(`SSE: max reconnect attempts (${MAX_RECONNECT_ATTEMPTS}) reached, giving up`);
        return;
      }

      const delay = Math.min(
        BASE_RECONNECT_DELAY * Math.pow(2, attempt),
        MAX_RECONNECT_DELAY,
      );
      reconnectAttemptsRef.current = attempt + 1;
      console.warn(`SSE: reconnecting in ${delay}ms (attempt ${attempt + 1}/${MAX_RECONNECT_ATTEMPTS})`);

      reconnectTimerRef.current = setTimeout(() => connect(tid), delay);
    };

    EVENT_TYPES.forEach((type) => {
      es.addEventListener(type, (e: MessageEvent) => {
        try {
          const data = JSON.parse(e.data);
          setEvents((prev) => [...prev, { event: type, data, timestamp: Date.now() }]);

          if (type === 'clarification_needed') {
            setClarificationQuestions(data.questions || []);
          }

          // P2-12: stop reconnecting once task reaches terminal state
          if (type === 'status' && data.status) {
            const terminalStatuses = ['done', 'stopped', 'failed'];
            if (terminalStatuses.includes(data.status)) {
              isTerminalRef.current = true;
            }
          }
        } catch {
          // ignore parse errors
        }
      });
    });
  }, []);

  useEffect(() => {
    if (!taskId) return;

    // Reset state for new task
    setEvents([]);
    setClarificationQuestions(null);
    setConnected(false);
    reconnectAttemptsRef.current = 0;
    isTerminalRef.current = false;

    connect(taskId);

    return () => {
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
    };
  }, [taskId, connect]);

  const clearClarification = useCallback(() => setClarificationQuestions(null), []);
  const clearEvents = useCallback(() => setEvents([]), []);

  return { events, connected, clarificationQuestions, clearClarification, clearEvents };
}
