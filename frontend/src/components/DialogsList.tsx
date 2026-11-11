import type { Dialog } from "../App";

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
              <span className="dialog-date">{dialog.date}</span>
            </button>
          );
        })}
      </div>
    </aside>
  );
}
