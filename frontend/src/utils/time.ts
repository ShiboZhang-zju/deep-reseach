/**
 * Format a date/time string to Beijing time (UTC+8).
 * All timestamps from the backend are ISO 8601.
 */

const BEIJING_TZ = 'Asia/Shanghai';

export function formatDateTime(iso: string): string {
  return new Date(iso).toLocaleString('zh-CN', { timeZone: BEIJING_TZ });
}

export function formatTime(iso: string): string {
  return new Date(iso).toLocaleTimeString('zh-CN', { timeZone: BEIJING_TZ });
}
