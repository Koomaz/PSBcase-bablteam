import type { Dialog } from "../App";

export function DialogsList({
  dialogs,
  activeId,
  onSelect,
  onClear,
}: {
  dialogs: Dialog[];
  activeId: number;
  onSelect: (id: number) => void;
  onClear: () => void;
}) {
  return (
    <aside className="dialogs">
      <div className="dialogs-head">
        <h2>Диалоги</h2>
        <button type="button" className="clear-history" onClick={onClear}>
          Удалить историю
        </button>
      </div>
      <div className="dialog-list">
        {dialogs.map((dialog) => {
          const firstUser = dialog.messages.find((m) => m.role === "user");
          return (
            <button
              key={dialog.id}
              type="button"
              className={"dialog-item" + (dialog.id === activeId ? " active" : "")}
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
