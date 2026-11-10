import { useEffect, useState } from "react";
import { ChatPanel } from "./components/ChatPanel";
import { SourcePanel } from "./components/SourcePanel";
import { api, type Source } from "./lib/api";

type Health = "checking" | "ready" | "down";

const HEALTH_LABEL: Record<Health, string> = {
  checking: "проверка сервиса…",
  ready: "сервис готов",
  down: "сервис недоступен",
};

export default function App() {
  const [health, setHealth] = useState<Health>("checking");
  const [source, setSource] = useState<Source | null>(null);

  useEffect(() => {
    api
      .health()
      .then(() => setHealth("ready"))
      .catch(() => setHealth("down"));
  }, []);

  return (
    <main className="page">
      <div className="page-frame">
        <div className="panel">
          <header className="panel-head">
            <div className="brand">
              <div className="brand-mark">ЮА</div>
              <div>
                <h1>Юридический ассистент</h1>
                <p>Поиск по базе документов · ответы с источниками</p>
              </div>
            </div>
            <div className="status" title="Состояние сервиса">
              <span className={`dot dot-${health}`} />
              <span>{HEALTH_LABEL[health]}</span>
            </div>
          </header>
          <ChatPanel onOpenSource={setSource} />
        </div>
      </div>
      <SourcePanel source={source} onClose={() => setSource(null)} />
    </main>
  );
}
