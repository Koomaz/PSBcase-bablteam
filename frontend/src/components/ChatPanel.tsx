import { useEffect, useRef, useState } from "react";
import type { Source } from "../lib/api";
import { Message, type MessageView } from "./Message";

const QUESTION_LIMIT = 4000;

export function ChatPanel({
  messages,
  busy,
  startPulse,
  onSend,
  onRetry,
  onOpenSource,
}: {
  messages: MessageView[];
  busy: boolean;
  startPulse: number;
  onSend: (question: string) => void;
  onRetry: () => void;
  onOpenSource: (source: Source) => void;
}) {
  const [input, setInput] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // поле ввода растёт по содержимому до 6 строк (дальше — прокрутка внутри)
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${el.scrollHeight}px`;
  }, [input]);

  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  }, [messages, busy]);

  function submit() {
    const text = input;
    if (!text.trim() || busy) return;
    setInput("");
    onSend(text);
  }

  return (
    <>
      <div className="messages" ref={scrollRef}>
        {messages.length === 0 && (
          <div className="empty" key={startPulse}>
            <h2>Задайте вопрос по юридическим документам</h2>
            <p>
              Ассистент найдёт применимые положения, приведёт ссылки на пункты
              и уточнит детали, если данных недостаточно.
            </p>
            <div className="examples">
              {[
                "Какой срок оплаты предусмотрен договором?",
                "Что будет при просрочке поставки?",
                "На какую дату определяется применимая редакция документа?",
              ].map((example) => (
                <button
                  key={example}
                  type="button"
                  className="example"
                  onClick={() => setInput(example)}
                >
                  {example}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((message) => (
          <Message
            key={message.id}
            message={message}
            onOpenSource={onOpenSource}
            onRetry={onRetry}
          />
        ))}

        {busy && (
          <div className="msg msg-assistant">
            <div className="bubble bubble-typing">
              <span />
              <span />
              <span />
            </div>
          </div>
        )}
      </div>

      <form
        className="composer"
        onSubmit={(event) => {
          event.preventDefault();
          submit();
        }}
      >
        <textarea
          ref={textareaRef}
          value={input}
          rows={1}
          maxLength={QUESTION_LIMIT}
          placeholder="Вопрос по документам базы знаний…"
          onChange={(event) => setInput(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              submit();
            }
          }}
        />
        <button type="submit" className="send" disabled={busy || !input.trim()}>
          {busy ? "Думаю…" : "Отправить"}
        </button>
      </form>
      <div className="composer-hint">
        <span>
          {input.length} / {QUESTION_LIMIT}
        </span>
        <span>Enter — отправить · Shift+Enter — новая строка</span>
      </div>
    </>
  );
}
