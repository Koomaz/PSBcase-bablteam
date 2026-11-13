import type { Dialog } from "../App";

/** Дата карточки диалога: сегодня — время последнего сообщения,
 *  вчера — «вчера», раньше — дата. Без сообщений — как есть. */
function formatDialogDate(dialog: Dialog): string {
  const last = [...dialog.messages].reverse().find((m) => m.at);
  if (!last?.at) return dialog.date;
  const moment = new Date(last.at);
  const startOf = (d: Date) => new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime();
  const days = Math.round((startOf(new Date()) - startOf(moment)) / 86400000);
  if (days <= 0) {
    return moment.toLocaleTimeString("ru-RU", { hour: "2-digit", minute: "2-digit" });
  }
  if (days === 1) return "вчера";
  return moment.toLocaleDateString("ru-RU", { day: "numeric", month: "long" });
}

export function DialogsList({
  dialogs,
  activeId,
  pulseId,
  onSelect,
  onClear,
  onClose,
}: {
  dialogs: Dialog[];
  activeId: number;
  pulseId?: number | null;
  onSelect: (id: number) => void;
  onClear: () => void;
  onClose?: () => void;
}) {
  return (
    <aside className="dialogs">
      <div className="dialogs-head">
        <h2>Диалоги</h2>
        <button type="button" className="clear-history" onClick={onClear}>
          Удалить историю
        </button>
        {onClose && (
          <button
            type="button"
            className="panel-close dialogs-close"
            onClick={onClose}
            aria-label="Закрыть"
          >
            ×
          </button>
        )}
      </div>
      <div className="dialog-list">
        {dialogs.map((dialog) => {
          const firstUser = dialog.messages.find((m) => m.role === "user");
          return (
            <button
              key={dialog.id}
              type="button"
              className={
                "dialog-item" +
                (dialog.id === activeId ? " active" : "") +
                (dialog.id === pulseId ? " pop" : "")
              }
              onClick={() => onSelect(dialog.id)}
            >
              <span className="dialog-title">{dialog.title}</span>
              <span className="dialog-snippet">
                {firstUser ? firstUser.text : "Пустой диалог"}
              </span>
              <span className="dialog-date">{formatDialogDate(dialog)}</span>
            </button>
          );
        })}
      </div>
    </aside>
  );
}
