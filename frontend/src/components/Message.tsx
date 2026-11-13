import { useState } from "react";
import type { AnswerKind, Source } from "../lib/api";
import { renderRichText } from "../lib/markdown";

const KIND_LABELS: Record<AnswerKind, string> = {
  answer: "Ответ",
  partial: "Частичный ответ",
  clarification: "Уточнение",
  insufficient: "Недостаточно данных",
  refusal: "Отказ",
};

export interface MessageView {
  id: number;
  role: "user" | "assistant" | "error";
  kind?: AnswerKind;
  text: string;
  sources?: Source[];
  errorMessage?: string;
  at?: number; // момент сообщения (epoch ms)
}

function formatTime(at?: number): string {
  if (!at) return "";
  return new Date(at).toLocaleTimeString("ru-RU", {
    hour: "2-digit",
    minute: "2-digit",
  });
}

function SourceList({
  sources,
  onOpenSource,
}: {
  sources: Source[];
  onOpenSource: (source: Source) => void;
}) {
  const [showAll, setShowAll] = useState(false);
  const visible = showAll ? sources : sources.slice(0, 3);
  return (
    <div className="source-list">
      {visible.map((source, index) => (
        <button
          key={source.chunk_id}
          type="button"
          className="source-chip"
          onClick={() => onOpenSource(source)}
        >
          <span className="chip-num">[{index + 1}]</span>
          <span>{source.doc_title}</span>
          {source.clause && <span className="chip-ref">п. {source.clause}</span>}
        </button>
      ))}
      {sources.length > 3 && (
        <button
          type="button"
          className="source-more"
          onClick={() => setShowAll((value) => !value)}
        >
          {showAll ? "Свернуть" : `Ещё ${sources.length - 3}`}
        </button>
      )}
    </div>
  );
}

export function Message({
  message,
  onOpenSource,
  onRetry,
}: {
  message: MessageView;
  onOpenSource: (source: Source) => void;
  onRetry?: () => void;
}) {
  if (message.role === "user") {
    return (
      <div className="msg msg-user">
        <div className="bubble bubble-user">
          {message.text}
          <div className="msg-time">{formatTime(message.at)}</div>
        </div>
      </div>
    );
  }

  if (message.role === "error") {
    return (
      <div className="msg msg-assistant">
        <div className="bubble bubble-error">
          <strong>Не получилось обработать запрос</strong>
          <p>{message.errorMessage}</p>
          {onRetry && <button type="button" className="retry" onClick={onRetry}>
            Повторить
          </button>}
        </div>
      </div>
    );
  }

  const sources = message.sources ?? [];
  return (
    <div className="msg msg-assistant">
      <div className="bubble">
        {message.kind && message.kind !== "answer" && (
          <div className={`badge badge-${message.kind}`}>
            {KIND_LABELS[message.kind]}
          </div>
        )}
        <div
          className="answer"
          dangerouslySetInnerHTML={{ __html: renderRichText(message.text) }}
          onClick={(event) => {
            const target = event.target as HTMLElement;
            const cite = target.closest<HTMLElement>("[data-cite]");
            if (!cite) return;
            const source = sources[Number(cite.dataset.cite) - 1];
            if (source) onOpenSource(source);
          }}
        />
        {sources.length > 0 && (
          <SourceList sources={sources} onOpenSource={onOpenSource} />
        )}
        <div className="msg-time">{formatTime(message.at)}</div>
      </div>
    </div>
  );
}
