import { useEffect, useRef, useState } from "react";
import { api, type DocumentInfo, type Source } from "../lib/api";

/**
 * Полноэкранный просмотр документа с подсветкой фрагмента.
 *
 * Границы фрагмента берём из char_start/char_end источника: они указывают на
 * позиции в том тексте документа, который разбирает индекс, поэтому выделение
 * совпадает побуквенно. Если границы не сошлись — ищем фрагмент по тексту.
 */
export function DocumentModal({
  source,
  onClose,
}: {
  source: Source;
  onClose: () => void;
}) {
  const [doc, setDoc] = useState<DocumentInfo | null>(null);
  const [error, setError] = useState<string | null>(null);
  const markRef = useRef<HTMLElement>(null);

  useEffect(() => {
    let cancelled = false;
    setDoc(null);
    setError(null);
    api
      .document(source.doc_id)
      .then((data) => {
        if (!cancelled) setDoc(data);
      })
      .catch((err: Error) => {
        if (!cancelled) setError(err.message);
      });
    return () => {
      cancelled = true;
    };
  }, [source.doc_id]);

  useEffect(() => {
    markRef.current?.scrollIntoView({ block: "center" });
  }, [doc]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  function highlight(): { before: string; target: string; after: string } {
    const text = doc?.text ?? "";
    const characters = Array.from(text);
    let start = source.char_start ?? 0;
    let end = source.char_end ?? 0;
    if (!(start < end && end <= characters.length)) {
      const found = text.indexOf(source.source_reference?.split(", ")[0] ?? "");
      const fallback = found >= 0 ? text.indexOf("\n", found) : -1;
      start = Array.from(text.slice(0, Math.max(0, fallback))).length;
      end = start;
    }
    return {
      before: characters.slice(0, start).join(""),
      target: characters.slice(start, end).join(""),
      after: characters.slice(end).join(""),
    };
  }

  const parts = doc ? highlight() : null;

  return (
    <div className="doc-modal" role="dialog" aria-label="Документ">
      <div className="doc-modal-card">
        <div className="doc-modal-head">
          <h2>{doc?.doc_title ?? source.doc_title}</h2>
          <button type="button" className="panel-close" onClick={onClose} aria-label="Закрыть">
            ×
          </button>
        </div>
        <div className="doc-modal-body">
          {error && <div className="fragment-error">Полный текст документа недоступен: {error}</div>}
          {!error && !doc && <div className="fragment-loader">Загрузка документа…</div>}
          {parts && (
            <div className="doc-text">
              {parts.before}
              <mark className="doc-highlight" ref={markRef}>
                {parts.target || "\u00a0"}
              </mark>
              {parts.after}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
