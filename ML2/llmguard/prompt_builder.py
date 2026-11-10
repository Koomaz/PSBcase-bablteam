"""Сборка сообщений для чат-модели: статичный system + динамический user."""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Any, Sequence

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "system_prompt.md"

USER_TEMPLATE = """<evidence>
{evidence}
</evidence>

<premise_warnings>
{warnings}
</premise_warnings>

<retrieval_diagnostics>
{diagnostics}
</retrieval_diagnostics>

<user_context>
{user_context}
</user_context>

<question>
{question}
</question>

<answer_structure>
Сначала вывод своими словами (да/нет, сумма, срок, порядок действий), затем основание
с номером пункта и ссылкой [En]. Не начинай ответ с номера пункта и не копируй текст
документа целиком. Если вопрос состоит из нескольких частей — ответь на каждую.
</answer_structure>"""


def _current_date() -> str:
    explicit = os.getenv("EVENT_DATE", "").strip()
    if explicit:
        return explicit
    return date.today().strftime("%d.%m.%Y")


def load_system_prompt(path: str | Path | None = None, today: str | None = None) -> str:
    source = Path(path) if path else PROMPT_PATH
    if source.is_file():
        text = source.read_text(encoding="utf-8").strip()
    else:
        # в ZIP-сабмите .md не переносится: промпт лежит в system_prompt.py
        from system_prompt import SYSTEM_PROMPT as text  # type: ignore

        text = str(text).strip()
    return text.replace("{{CURRENT_DATE}}", today or _current_date())


def _render_item(index: int, item: dict[str, Any]) -> str:
    meta = item.get("meta") or {}
    meta_line = ", ".join(f"{key}={value}" for key, value in meta.items()) or "нет"
    return (
        f"[E{index}] {item.get('source') or 'источник не указан'}\n"
        f"metadata: {meta_line}\n"
        f"text: {item.get('text', '').strip()}"
    )


def render_evidence(items: Sequence[dict[str, Any]]) -> str:
    if not items:
        return "(пусто — в базе не найдено ни одного релевантного фрагмента)"
    blocks = []
    for index, item in enumerate(items, start=1):
        eid = item.get("eid") or f"E{index}"
        block = _render_item(index, item)
        # eid из build_evidence уже совпадает с индексом; подстрахуемся при ручной сборке
        blocks.append(block.replace(f"[E{index}]", f"[{eid}]", 1))
    return "\n\n".join(blocks)


def build_messages(
    question: str,
    items: Sequence[dict[str, Any]],
    warnings: Sequence[str] | None = None,
    diagnostics: str = "",
    system_prompt: str | None = None,
    user_context: str = "",
) -> list[dict[str, str]]:
    system = system_prompt if system_prompt is not None else load_system_prompt()
    warning_text = (
        "\n".join(f"- {warning}" for warning in warnings) if warnings else "(нет)"
    )
    user = USER_TEMPLATE.format(
        evidence=render_evidence(items),
        warnings=warning_text,
        diagnostics=diagnostics or "(нет)",
        user_context=(
            "Ранее указанные пользователем обстоятельства. Используй их для понимания текущего вопроса. "
            "Проверяй утверждения о документах по доказательствам.\n" + user_context if user_context else "(нет)"
        ),
        question=(question or "").strip(),
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_repair_message(issues: Sequence[str]) -> dict[str, str]:
    """Второй проход: показываем модели результаты валидатора, а не переписываем ответ сами."""
    bullets = "\n".join(f"- {issue}" for issue in issues)
    hints: list[str] = []
    joined = " ".join(issues).lower()
    if "расхождение" in joined or "посылк" in joined:
        hints.append(
            "Расхождение с вопросом называется в первом предложении по формуле "
            "«Посылка не подтверждается: <что сказано в вопросе> — в документах <как на самом деле> [E1]», "
            "после чего приводится верное значение. Отказываться в этом случае нельзя."
        )
    if "нет ни одной ссылки" in joined or "необоснованный отказ" in joined:
        hints.append(
            "В доказательствах есть ответ: отвечай по ним и ставь [En] после каждого утверждения. "
            "Отказ допустим только при полностью пустом <evidence>."
        )
    if "обрывается" in joined:
        hints.append("Уложи ответ в 2–4 законченных предложения, последнее должно заканчиваться точкой.")
    hint_block = ("\nПодсказки:\n" + "\n".join(f"- {hint}" for hint in hints)) if hints else ""
    return {
        "role": "user",
        "content": (
            "Твой предыдущий ответ не прошёл проверку. Нарушения:\n"
            f"{bullets}{hint_block}\n\n"
            "Перепиши ответ, устранив нарушения. Требования прежние: только факты из <evidence>, "
            "ссылка [En] после каждого утверждения, ID только из списка, до 180 слов, без markdown. "
            "Не обсуждай проверку и не извиняйся — выдай только исправленный ответ."
        ),
    }
