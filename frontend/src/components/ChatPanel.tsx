import { useEffect, useRef, useState } from "react";
import { api, type ChatTurn, type Source } from "../lib/api";
import { Message, type MessageView } from "./Message";

const LIMITS = {
  questionChars: 4000,
  historyMessages: 8,
  historyChars: 12000,
};

function buildHistory(messages: MessageView[]): ChatTurn[] {
  const turns = messages
    .filter((m) => m.role === "user" || m.role === "assistant")
    .map((m) => ({ role: m.role as "user" | "assistant", content: m.text }))
    .slice(-LIMITS.historyMessages);
  while (
    turns.length > 1 &&
    turns.reduce((sum, turn) => sum + turn.content.length, 0) > LIMITS.historyChars
  ) {
    turns.shift();
  }
  return turns;
}

export function ChatPanel({
  onOpenSource,
}: {
  onOpenSource: (source: Source) => void;
}) {
  const [messages, setMessages] = useState<MessageView[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const nextId = useRef(1);

  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, busy]);

  async function ask(question: string, isRetry = false) {
    const text = question.trim();
    if (!text || busy) return;

    const userMessage: MessageView = {
      id: nextId.current++,
      role: "user",
      text,
    };
    setMessages((prev) => (isRetry ? prev : [...prev, userMessage]));
    setBusy(true);

    try {
      const result = await api.chat(text, buildHistory(messages));
      setMessages((prev) => [
        ...prev,
        {
          id: nextId.current++,
          role: "assistant",
          kind: result.kind,
          text: result.answer,
          sources: result.sources,
        },
      ]);
    } catch (err) {
      const error = err as Error;
      setMessages((prev) => [
        ...prev,
        {
          id: nextId.current++,
          role: "error",
          text: "",
          errorMessage: error.message,
        },
      ]);
    } finally {
      setBusy(false);
    }
  }

  function retryLast() {
    const lastUser = [...messages].reverse().find((m) => m.role === "user");
    if (!lastUser) return;
    setMessages((prev) => prev.filter((m) => m.role !== "error"));
    void ask(lastUser.text, true);
  }

  function submit() {
    const text = input;
    setInput("");
    void ask(text);
  }

  return (
    <div className="chat">
      <div className="messages" ref={scrollRef}>
        {messages.length === 0 && (
          <div className="empty">
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
            onRetry={retryLast}
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
          value={input}
          rows={1}
          maxLength={LIMITS.questionChars}
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
          {input.length} / {LIMITS.questionChars}
        </span>
        <span>Enter — отправить · Shift+Enter — новая строка</span>
      </div>
    </div>
  );
}
