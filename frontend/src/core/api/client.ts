/** Typed fetch client for the FastAPI backend (127.0.0.1:30211). */

export const API_BASE_URL: string =
  (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? 'http://127.0.0.1:30211';

export interface ApiErrorEnvelope {
  error: {
    code: string;
    message: string;
    details: Record<string, unknown>;
  };
}

export class ApiError extends Error {
  code: string;
  httpStatus: number;
  details: Record<string, unknown>;

  constructor(code: string, message: string, httpStatus: number, details: Record<string, unknown> = {}) {
    super(message);
    this.code = code;
    this.httpStatus = httpStatus;
    this.details = details;
  }
}

async function parseJsonSafe(res: Response): Promise<unknown> {
  const text = await res.text();
  if (!text) return null;
  try {
    return JSON.parse(text) as unknown;
  } catch {
    return { raw: text };
  }
}

export async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const started = performance.now();
  let res: Response;
  try {
    res = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
    });
  } catch (err) {
    throw new ApiError('BACKEND_UNREACHABLE', `Backend unreachable at ${API_BASE_URL}`, 0, {
      cause: String(err),
    });
  }
  const latencyMs = performance.now() - started;
  const data = await parseJsonSafe(res);
  if (!res.ok) {
    const envelope = data as ApiErrorEnvelope | null;
    throw new ApiError(
      envelope?.error?.code ?? `HTTP_${res.status}`,
      envelope?.error?.message ?? `Request failed (${res.status})`,
      res.status,
      (envelope?.error?.details as Record<string, unknown>) ?? {},
    );
  }
  return { ...(data as Record<string, unknown>), _latencyMs: latencyMs } as unknown as T;
}

export const api = {
  get: <T>(path: string) => apiFetch<T>(path, { method: 'GET' }),
  post: <T>(path: string, body?: unknown) =>
    apiFetch<T>(path, { method: 'POST', body: body === undefined ? undefined : JSON.stringify(body) }),
  del: <T>(path: string) => apiFetch<T>(path, { method: 'DELETE' }),
};
