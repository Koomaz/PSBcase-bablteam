/**
 * Клиент API юридического ассистента.
 *
 * Контракт (docs/backend-contract.md):
 *   GET  /health/ready            -> {status} | 503
 *   POST /api/chat                -> {kind, answer, sources[]}
 *       {question, history:[{role:"user"|"assistant", content}]}
 *   GET  /api/sources/{chunk_id}  -> источник + дословный текст фрагмента
 * Ошибки: {"detail": {"code", "message"}}.
 */

import { API_BASE_URL } from "./config";

export type AnswerKind =
  | "answer"
  | "partial"
  | "clarification"
  | "insufficient"
  | "refusal";

export interface ChatTurn {
  role: "user" | "assistant";
  content: string;
}

export interface Source {
  chunk_id: string;
  source_reference: string;
  doc_id: string;
  doc_title: string;
  doc_type: string;
  doc_number: string | null;
  doc_date: string | null;
  file_path: string;
  article: string | null;
  part: string | null;
  clause: string | null;
  subclause: string | null;
  section: string | null;
  appendix: string | null;
  amends_clause: string | null;
  edition_date: string | null;
  effective_date: string | null;
  snapshot_date: string | null;
  version_status: string | null;
  temporal_status: string | null;
  parent_doc_id: string | null;
  provision_effective_date: string | null;
  amendment_application_date: string | null;
  amendment_application_relation: string | null;
  char_start: number;
  char_end: number;
}

export interface Fragment extends Source {
  text: string;
}

export interface DocumentInfo {
  doc_id: string;
  doc_title: string;
  file_path?: string;
  text: string;
}

export interface ChatResponse {
  kind: AnswerKind;
  answer: string;
  sources: Source[];
}

export class ApiError extends Error {
  code: string;

  constructor(code: string, message: string) {
    super(message);
    this.code = code;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(API_BASE_URL + path, init);
  } catch {
    throw new ApiError(
      "network",
      "Нет связи с сервисом. Проверьте, что он запущен."
    );
  }

  let body: unknown = null;
  try {
    body = await response.json();
  } catch {
    throw new ApiError(
      "invalid_response",
      "Сервис вернул некорректный ответ."
    );
  }

  if (!response.ok) {
    const raw = (body as { detail?: unknown })?.detail;
    const detail = raw && typeof raw === "object" ? (raw as { code?: string; message?: string }) : {};
    throw new ApiError(
      detail.code ?? (response.status === 404 ? "not_found" : "internal"),
      detail.message ??
        (typeof raw === "string" ? raw : "Не удалось обработать запрос.")
    );
  }
  return body as T;
}

export const api = {
  health(): Promise<{ status: string }> {
    return request("/health/ready");
  },

  chat(question: string, history: ChatTurn[]): Promise<ChatResponse> {
    return request("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, history }),
    });
  },

  source(chunkId: string): Promise<Fragment> {
    return request(`/api/sources/${encodeURIComponent(chunkId)}`);
  },

  document(docId: string): Promise<DocumentInfo> {
    return request(`/api/documents/${encodeURIComponent(docId)}`);
  },
};
