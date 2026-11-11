/**
 * Флаги возможностей интерфейса.
 *
 * Управление — через переменные сборки Vite (VITE_*) или .env при `npm run build`:
 *   VITE_FEATURE_DOCUMENT_VIEWER=1   — включить просмотр полного документа
 * По умолчанию фичи выключены. Значения "1" / "true" / "on" / "yes" (без учёта
 * регистра) включают фичу, всё остальное, включая отсутствие переменной, — нет.
 */

const flag = (name: string): boolean => {
  const raw = String(import.meta.env[name] ?? "").trim().toLowerCase();
  return ["1", "true", "on", "yes"].includes(raw);
};

export const FEATURES = {
  /** Просмотр полного документа с подсветкой фрагмента (нужен эндпоинт /api/documents/{doc_id}). */
  documentViewer: flag("VITE_FEATURE_DOCUMENT_VIEWER"),
};
