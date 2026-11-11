const readEnvString = (value: unknown): string | undefined =>
  typeof value === "string" && value.trim().length > 0 ? value.trim() : undefined;

/** Базовый адрес API. По умолчанию тот же origin (прокси Vite или serve.py). */
export const API_BASE_URL = readEnvString(import.meta.env.VITE_API_URL) ?? "";
