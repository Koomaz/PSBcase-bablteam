import { useEffect, useRef, useState } from "react";
import { ChatPanel } from "./components/ChatPanel";
import { DialogsList } from "./components/DialogsList";
import { SourcePanel } from "./components/SourcePanel";
import { api, type ChatTurn, type Source } from "./lib/api";
import { installScrollbarHover } from "./lib/scrollbar";
import type { MessageView } from "./components/Message";

export interface Dialog {
  id: number;
  title: string;
  date: string;
  messages: MessageView[];
}

type Theme = "light" | "dark";

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
    .map((m) => ({ role: m.role as "user" | "assistant", content: m.text.slice(0, 4000) }))
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
  const [busyIds, setBusyIds] = useState<number[]>([]);
  const [dialogsOpen, setDialogsOpen] = useState(false);
  const [pulseId, setPulseId] = useState<number | null>(null);
  const [startPulse, setStartPulse] = useState(0);
  const nextDialogId = useRef(
    Math.max(...saved.dialogs.map((d) => d.id)) + 1
  );
  const nextMessageId = useRef(
    Math.max(0, ...saved.dialogs.flatMap((d) => d.messages.map((m) => m.id))) + 1
  );

  useEffect(() => {
    document.documentElement.dataset.colorScheme = theme;
    localStorage.setItem("theme", theme);
  }, [theme]);

  // диалоги переживают перезагрузку страницы
  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ dialogs, activeId }));
  }, [dialogs, activeId]);

  // полоса прокрутки расширяется при наведении на неё
  useEffect(() => installScrollbarHover(), []);

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
    if (activeDialog.messages.length === 0) {
      // уже в новом (пустом) диалоге: просто проигрываем анимацию стартового
      // состояния — как будто новый чат только что создался
      setPulseId(activeDialog.id);
      setStartPulse((n) => n + 1);
      window.setTimeout(() => setPulseId(null), 500);
      return;
    }
    // пустой диалог уже есть — переходим в него, чтобы не копить пустые
    const empty = dialogs.find((d) => d.messages.length === 0);
    if (empty) {
      setActiveId(empty.id);
      setSource(null);
      return;
    }
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

  async function send(question: string, retryMessages?: MessageView[]) {
    const text = question.trim();
    if (!text) return;

    const dialogId = activeId;
    if (busyIds.includes(dialogId)) return;
    const history = buildHistory(retryMessages ?? activeDialog.messages);
    if (!retryMessages) pushMessage(dialogId, { role: "user", text, at: Date.now() });
    patchDialog(dialogId, (d) =>
      d.messages.some((m) => m.role === "user") && d.title !== "Новый диалог"
        ? d
        : { ...d, title: text.length > 46 ? text.slice(0, 46) + "…" : text }
    );
    setBusyIds((prev) => [...prev, dialogId]);

    try {
      const result = await api.chat(text, history);
      pushMessage(dialogId, {
        role: "assistant",
        kind: result.kind,
        text: result.answer,
        sources: result.sources,
        at: Date.now(),
      });
    } catch (err) {
      pushMessage(dialogId, {
        role: "error",
        text: "",
        errorMessage: (err as Error).message,
        at: Date.now(),
      });
    } finally {
      setBusyIds((prev) => prev.filter((id) => id !== dialogId));
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
    const lastUserIndex = messages.findIndex((m) => m.id === lastUser.id);
    void send(lastUser.text, messages.slice(0, lastUserIndex));
  }

  return (
    <main className="page">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">ЮА</div>
          <h1>Юридический ассистент</h1>
        </div>
        <div className="topbar-actions">
          <button type="button" className="new-dialog" onClick={newDialog}>
            + Новый<span className="wide-label"> диалог</span>
          </button>
          <button
            type="button"
            className="theme-toggle"
            title="Сменить тему"
            onClick={() => setTheme((t) => (t === "dark" ? "light" : "dark"))}
          >
            <span className="theme-icon">{theme === "dark" ? "☀" : "☾"}</span>
            <span className="wide-label">
              {theme === "dark" ? " Светлая" : " Тёмная"}
            </span>
          </button>
        </div>
      </header>

      <div className="layout">
        <div className={"dialogs-wrap" + (dialogsOpen ? " open" : "")}>
          <DialogsList
            dialogs={dialogs}
            activeId={activeId}
            pulseId={pulseId}
            onClear={clearHistory}
            onClose={() => setDialogsOpen(false)}
            onSelect={(id) => {
              setActiveId(id);
              setSource(null);
              setDialogsOpen(false);
            }}
          />
        </div>
        <div className="panel">
          <button
            type="button"
            className="dialogs-tab"
            onClick={() => setDialogsOpen(true)}
            aria-label="Диалоги"
          >
            <span className="tab-icon" aria-hidden="true">☰</span>
            <span className="tab-label">Диалоги</span>
          </button>
          <ChatPanel
            key={activeDialog.id}
            startPulse={startPulse}
            busy={busyIds.includes(activeDialog.id)}
            messages={activeDialog.messages}
            onSend={(question) => { void send(question); }}
            onRetry={retry}
            onOpenSource={setSource}
          />
        </div>
        <SourcePanel source={source} onClose={() => setSource(null)} />
      </div>
    </main>
  );
}
