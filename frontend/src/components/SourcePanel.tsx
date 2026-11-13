import {
  useEffect,
  useRef,
  useState,
  type CSSProperties,
  type PointerEvent as ReactPointerEvent,
} from "react";
import { api, type Fragment, type Source } from "../lib/api";
import { FEATURES } from "../lib/features";
import { DocumentModal } from "./DocumentModal";

function formatReference(source: Source | Fragment): string {
  const parts: string[] = [];
  if (source.article) parts.push(`ст. ${source.article}`);
  if (source.part) parts.push(`ч. ${source.part}`);
  if (source.clause) parts.push(`п. ${source.clause}`);
  if (source.subclause) parts.push(`пп. ${source.subclause}`);
  if (source.section) parts.push(`разд. ${source.section}`);
  if (source.appendix) parts.push(`прил. ${source.appendix}`);
  return parts.join(", ");
}

function formatDate(value?: string | null): string | undefined {
  if (!value) return undefined;
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(value);
  return match ? `${match[3]}.${match[2]}.${match[1]}` : value;
}

const STATUS_LABELS: Record<string, string> = {
  undated: "без указания даты",
  current: "действующая редакция",
  current_snapshot: "действующая редакция",
  superseded: "утратила силу",
};

function statusLabel(value?: string | null): string | undefined {
  if (!value) return undefined;
  return STATUS_LABELS[value] ?? value;
}

function MetaRow({ label, value }: { label: string; value?: string | null }) {
  if (!value) return null;
  return (
    <div className="meta-row">
      <span className="meta-label">{label}</span>
      <span className="meta-value">{value}</span>
    </div>
  );
}

export function SourcePanel({
  source,
  onClose,
}: {
  source: Source | null;
  onClose: () => void;
}) {
  const [fragment, setFragment] = useState<Fragment | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showDocument, setShowDocument] = useState(false);

  // мобильная шторка: высота свободная, без фиксированных положений —
  // тянется за шапку или за контент (когда контент не скроллится),
  // остаётся там, где отпустили
  const [sheetHeight, setSheetHeight] = useState(() =>
    typeof window === "undefined" ? 500 : Math.round(window.innerHeight * 0.44)
  );
  const [dragging, setDragging] = useState(false);
  const dragRef = useRef<{ y: number; start: number } | null>(null);

  const scrollable = sheetHeight > (typeof window === "undefined" ? 560 : window.innerHeight * 0.7);

  useEffect(() => {
    if (!source) return;
    let cancelled = false;
    setFragment(null);
    setError(null);
    api
      .source(source.chunk_id)
      .then((data) => {
        if (!cancelled) setFragment(data);
      })
      .catch((err: Error) => {
        if (!cancelled) setError(err.message);
      });
    return () => {
      cancelled = true;
    };
  }, [source]);

  function onDragStart(event: ReactPointerEvent) {
    if (!window.matchMedia("(max-width: 720px)").matches) return;
    const target = event.target as HTMLElement;
    if (target.closest(".panel-close")) return;
    // когда контент скроллится — скроллим его, шторку не тянем
    if (target.closest(".source-panel-body") && scrollable) return;
    event.preventDefault(); // чтобы браузер не перехватил указатель
    dragRef.current = { y: event.clientY, start: sheetHeight };
    setDragging(true);
    (event.currentTarget as HTMLElement).setPointerCapture(event.pointerId);
  }

  function onDragMove(event: ReactPointerEvent) {
    const drag = dragRef.current;
    if (!drag) return;
    const min = 60; // ниже — шторку можно опустить почти до закрытия
    const max = Math.round(window.innerHeight * 0.94);
    const next = drag.start - (event.clientY - drag.y);
    setSheetHeight(Math.min(max, Math.max(min, next)));
  }

  function onDragEnd() {
    if (!dragRef.current) return;
    dragRef.current = null;
    setDragging(false);
    // опустили ниже предела — закрываем (только вертикальный режим)
    if (window.matchMedia("(max-width: 720px)").matches && sheetHeight < window.innerHeight * 0.25) {
      onClose();
    }
  }

  const grip = {
    onPointerDown: onDragStart,
    onPointerMove: onDragMove,
    onPointerUp: onDragEnd,
    onPointerCancel: onDragEnd,
  };

  if (!source) return null;

  return (
    <aside
      className={
        "source-panel" +
        (dragging ? " dragging" : "") +
        (scrollable ? " scrollable" : "")
      }
      style={{ "--sheet-h": `${sheetHeight}px` } as CSSProperties}
      aria-label="Источник"
    >
      <div className="sheet-grip" {...grip}>
        <div className="sheet-handle" />
        <button
          type="button"
          className="panel-close sheet-close"
          onClick={onClose}
          aria-label="Закрыть"
        >
          ×
        </button>
      </div>
      <div className="source-panel-body" {...grip}>
        <h3>{source.doc_title}</h3>
        <div className="meta">
          <MetaRow label="Положение" value={formatReference(source) || "—"} />
          <MetaRow label="Ссылка" value={source.source_reference} />
          <MetaRow label="Номер документа" value={source.doc_number} />
          <MetaRow label="Дата документа" value={formatDate(source.doc_date)} />
          <MetaRow label="Редакция от" value={formatDate(source.edition_date)} />
          <MetaRow label="Действует с" value={formatDate(source.effective_date)} />
          <MetaRow label="Статус редакции" value={statusLabel(source.version_status)} />
          <MetaRow label="Временной статус" value={statusLabel(source.temporal_status)} />
        </div>
        <div className="fragment">
          <h4>Подтверждающий фрагмент</h4>
          {error ? (
            <div className="fragment-error">{error}</div>
          ) : fragment ? (
            <blockquote>{fragment.text}</blockquote>
          ) : (
            <div className="fragment-loader">Загрузка фрагмента…</div>
          )}
          {FEATURES.documentViewer && (
            <button
              type="button"
              className="doc-open"
              onClick={() => setShowDocument(true)}
            >
              Открыть документ целиком
            </button>
          )}
        </div>
      </div>
      {showDocument && (
        <DocumentModal source={source} onClose={() => setShowDocument(false)} />
      )}
    </aside>
  );
}
