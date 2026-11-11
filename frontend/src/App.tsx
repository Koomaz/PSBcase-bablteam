import { useEffect, useRef, useState } from "react";
import { ChatPanel } from "./components/ChatPanel";
import { DialogsList } from "./components/DialogsList";
import { SourcePanel } from "./components/SourcePanel";
import { api, type ChatTurn, type Source } from "./lib/api";
import type { MessageView } from "./components/Message";

export interface Dialog {
  id: number;
  title: string;
  date: string;
  messages: MessageView[];
}

type Health = "checking" | "ready" | "down";
type Theme = "light" | "dark";

const HEALTH_LABEL: Record<Health, string> = {
  checking: "проверка сервиса…",
  ready: "сервис готов",
  down: "сервис недоступен",
};

const LIMITS = { historyMessages: 8, historyChars: 12000 };

function today(): string {
  return new Date().toLocaleDateString("ru-RU", { day: "numeric", month: "long" });
}

const STORAGE_KEY = "psb.dialogs";

interface SavedState {
  dialogs: Dialog[];
  activeId: number;
}

function loadSaved(): SavedState {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "null") as SavedState | null;
    if (saved?.dialogs?.length) {
      return { dialogs: saved.dialogs, activeId: saved.activeId ?? saved.dialogs[0].id };
    }
  } catch {
    /* повреждённые данные — начинаем с чистого листа */
  }
  return {
    dialogs: [{ id: 1, title: "Новый диалог", date: today(), messages: [] }],
    activeId: 1,
  };
}

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

export default function App() {
  const [health, setHealth] = useState<Health>("checking");
  const [source, setSource] = useState<Source | null>(null);
  const [theme, setTheme] = useState<Theme>(() => {
    const saved = localStorage.getItem("theme");
    if (saved === "light" || saved === "dark") return saved;
    return window.matchMedia("(prefers-color-scheme: dark)").matches
      ? "dark"
      : "light";
  });
  const [saved] = useState(loadSaved);
  const [dialogs, setDialogs] = useState<Dialog[]>(saved.dialogs);
  const [activeId, setActiveId] = useState(saved.activeId);
  const [busy, setBusy] = useState(false);
  const nextDialogId = useRef(
    Math.max(...saved.dialogs.map((d) => d.id)) + 1
  );
  const nextMessageId = useRef(
    Math.max(0, ...saved.dialogs.flatMap((d) => d.messages.map((m) => m.id))) + 1
  );

  useEffect(() => {
    api
      .health()
      .then(() => setHealth("ready"))
      .catch(() => setHealth("down"));
  }, []);

  useEffect(() => {
    document.documentElement.dataset.colorScheme = theme;
    localStorage.setItem("theme", theme);
  }, [theme]);

  // диалоги переживают перезагрузку страницы
  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ dialogs, activeId }));
  }, [dialogs, activeId]);

  const activeDialog = dialogs.find((d) => d.id === activeId) ?? dialogs[0];

  function patchDialog(dialogId: number, patch: (dialog: Dialog) => Dialog) {
    setDialogs((prev) => prev.map((d) => (d.id === dialogId ? patch(d) : d)));
  }

  function pushMessage(dialogId: number, message: Omit<MessageView, "id">) {
    patchDialog(dialogId, (d) => ({
      ...d,
      messages: [...d.messages, { ...message, id: nextMessageId.current++ }],
    }));
  }

  function newDialog() {
    const id = nextDialogId.current++;
    setDialogs((prev) => [
      { id, title: "Новый диалог", date: today(), messages: [] },
      ...prev,
    ]);
    setActiveId(id);
    setSource(null);
  }

  function clearHistory() {
    if (!window.confirm("Удалить всю историю диалогов? Это действие нельзя отменить.")) {
      return;
    }
    const id = nextDialogId.current++;
    setDialogs([{ id, title: "Новый диалог", date: today(), messages: [] }]);
    setActiveId(id);
    setSource(null);
  }

  async function send(question: string) {
    const text = question.trim();
    if (!text || busy) return;

    const dialogId = activeId;
    const history = buildHistory(activeDialog.messages);
    pushMessage(dialogId, { role: "user", text });
    patchDialog(dialogId, (d) =>
      d.messages.some((m) => m.role === "user") && d.title !== "Новый диалог"
        ? d
        : { ...d, title: text.length > 46 ? text.slice(0, 46) + "…" : text }
    );
    setBusy(true);

    try {
      const result = await api.chat(text, history);
      pushMessage(dialogId, {
        role: "assistant",
        kind: result.kind,
        text: result.answer,
        sources: result.sources,
      });
    } catch (err) {
      pushMessage(dialogId, {
        role: "error",
        text: "",
        errorMessage: (err as Error).message,
      });
    } finally {
      setBusy(false);
    }
  }

  function retry() {
    const messages = activeDialog.messages;
    const lastUser = [...messages].reverse().find((m) => m.role === "user");
    if (!lastUser) return;
    patchDialog(activeId, (d) => ({
      ...d,
      messages: d.messages.filter((m) => m.role !== "error"),
    }));
    void send(lastUser.text);
  }

  return (
    <main className="page">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">ЮА</div>
          <div>
            <h1>Юридический ассистент</h1>
            <p>Поиск по базе документов · ответы с источниками</p>
          </div>
        </div>
        <div className="topbar-actions">
          <button type="button" className="new-dialog" onClick={newDialog}>
            + Новый диалог
          </button>
          <button
            type="button"
            className="theme-toggle"
            title="Сменить тему"
            onClick={() => setTheme((t) => (t === "dark" ? "light" : "dark"))}
          >
            {theme === "dark" ? "☀ Светлая" : "☾ Тёмная"}
          </button>
          <div className="status" title="Состояние сервиса">
            <span className={`dot dot-${health}`} />
            <span>{HEALTH_LABEL[health]}</span>
          </div>
        </div>
      </header>

      <div className="layout">
        <DialogsList
          dialogs={dialogs}
          activeId={activeId}
          onClear={clearHistory}
          onSelect={(id) => {
            setActiveId(id);
            setSource(null);
          }}
        />
        <div className="panel">
          <ChatPanel
            messages={activeDialog.messages}
            busy={busy}
            onSend={send}
            onRetry={retry}
            onOpenSource={setSource}
          />
        </div>
        <SourcePanel source={source} onClose={() => setSource(null)} />
      </div>
    </main>
  );
}
