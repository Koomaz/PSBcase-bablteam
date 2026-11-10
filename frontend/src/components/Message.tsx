import type { AnswerKind, Source } from "../lib/api";

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
}

function withCitations(
  text: string,
  sources: Source[] | undefined,
  onOpenSource: (source: Source) => void
) {
  return text.split(/(\[\d{1,2}\])/g).map((part, index) => {
    const match = /^\[(\d{1,2})\]$/.exec(part);
    if (!match) {
      return <span key={index}>{part}</span>;
    }
    const source = sources?.[Number(match[1]) - 1];
    if (!source) {
      return <span key={index}>{part}</span>;
    }
    return (
      <button
        key={index}
        type="button"
        className="cite"
        title={source.source_reference}
        onClick={() => onOpenSource(source)}
      >
        {part}
      </button>
    );
  });
}

export function Message({
  message,
  onOpenSource,
  onRetry,
}: {
  message: MessageView;
  onOpenSource: (source: Source) => void;
  onRetry: () => void;
}) {
  if (message.role === "user") {
    return (
      <div className="msg msg-user">
        <div className="bubble bubble-user">{message.text}</div>
      </div>
    );
  }

  if (message.role === "error") {
    return (
      <div className="msg msg-assistant">
        <div className="bubble bubble-error">
          <strong>Не получилось обработать запрос</strong>
          <p>{message.errorMessage}</p>
          <button type="button" className="retry" onClick={onRetry}>
            Повторить
          </button>
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
        <div className="answer">{withCitations(message.text, sources, onOpenSource)}</div>
        {sources.length > 0 && (
          <div className="source-list">
            {sources.map((source, index) => (
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
          </div>
        )}
      </div>
    </div>
  );
}
