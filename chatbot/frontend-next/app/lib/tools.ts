/**
 * Tool activity as streamed by the backend (`{"tool": {...}}` SSE frames).
 *
 * Each call arrives twice: once as "running" before the tool executes, then
 * as "done" or "error" with a trimmed JSON result when there is one. The two
 * frames share an id, so the second replaces the first in place.
 */
export type ToolStatus = "running" | "done" | "error";

export interface ToolActivity {
  id: string;
  name: string;
  status: ToolStatus;
  args: Record<string, unknown>;
  /** The tool's result, only when it was a small JSON object. */
  data: Record<string, unknown> | null;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** Validate one frame's payload; anything malformed is dropped, not shown. */
export function parseToolActivity(raw: unknown): ToolActivity | null {
  if (!isRecord(raw)) return null;
  const { id, name, status, args, data } = raw;
  if (typeof id !== "string" || typeof name !== "string") return null;
  if (status !== "running" && status !== "done" && status !== "error") return null;
  return {
    id,
    name,
    status,
    args: isRecord(args) ? args : {},
    data: isRecord(data) ? data : null,
  };
}

/** Insert a new call, or replace the earlier frame for the same call. */
export function upsertToolActivity(
  list: ToolActivity[] | undefined,
  next: ToolActivity
): ToolActivity[] {
  const current = list ?? [];
  const index = current.findIndex((t) => t.id === next.id);
  if (index < 0) return [...current, next];
  const updated = [...current];
  updated[index] = next;
  return updated;
}
