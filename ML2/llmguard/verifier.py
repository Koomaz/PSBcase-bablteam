"""Второй проход после валидатора: проверка полноты, заземлённости и уместности.

Валидатор проверяет форму (ссылки, числа, объём). Этот модуль задаёт модели
вопросы, которые валидатор проверить не может:

* все ли части вопроса отвечены;
* не упущено ли применимое условие, которое есть в доказательствах;
* нет ли утверждения, которого в доказательствах нет (главный риск — обнуление
  оценки за выдуманный факт);
* нет ли деталей, не нужных для ответа.

Модель-редактор возвращает строгий JSON. Если JSON не получен, проверка
считается пропущенной: сбой редактора не должен ломать прогон.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from .prompt_builder import render_evidence
from .textutil import split_sentences, word_key as _word_key

MAX_EVIDENCE_CHARS = 9000

SYSTEM = (
    "Ты — строгий юридический редактор. Проверяешь черновик ответа по документам. "
    "Не добавляй знаний из памяти и не выдумывай нормы. Отвечай только JSON."
)

PROMPT = """Проверь черновик ответа на юридический вопрос.

<question>
{question}
</question>

<evidence>
{evidence}
</evidence>

<draft>
{draft}
</draft>

Проверь четыре вещи.

1. ПОЛНОТА. Представь, что ответ читает юрист, который будет действовать по нему.
   Все ли условия, которые меняют его действие, расходы или риск, отражены? Включай
   прямо связанные условия того же обязательства/раздела, даже если вопрос называет
   только один элемент. Ориентиры:
   - вопрос об арендной плате: размер, срок внесения и отдельная оплата коммунальных
     платежей/электроэнергии; обеспечительный платёж — только если вопрос о первом
     платеже или депозите; индексация — только если вопрос об изменении платы или
     о расхождении суммы;
   - вопрос о приёмке по качеству: срок проверки качества и приёмка по количеству
     в момент передачи, если они установлены тем же пунктом;
   - вопрос о том, кто запускает согласование: инициатор и применимая редакция;
     сроки согласования и СЭД — лишнее;
   - вопрос об оплате поставки: срок оплаты, предоплата и порядок её согласования.
   Если вопрос построен на ложной посылке/расхождении, не требуй соседних условий, не относящихся
   к опровержению: достаточно верного факта и правила, которое он нарушает.
   Если вопрос — продолжение диалога, не требуй повторять ранее сообщённые факты (номер
   договора, гарантийный срок), если новый вопрос о другом.
   Не требуй того, чего в доказательствах нет.

2. ЗАЗЕМЛЁННОСТЬ. Каждое утверждение должно прямо следовать из доказательств.
   Перечисли утверждения, которых в доказательствах нет. Особое внимание — выводам
   «обязан удалить», «прекращается», «запрещено», суммам, срокам, статьям и пунктам,
   которых нет в тексте. Если вывод сформулирован как толкование, а не как текст нормы,
   отметь его как недоказанный. Не считай недоказанным вывод, который прямо следует
   из текста (например, «4 рабочих дня превышают предел в 3 дня»).

3. УМЕСТНОСТЬ. Перечисли детали, не нужные для ответа: процедуры и сроки других
   процессов, реквизиты, адреса, площади, сведения о других договорах, общие
   рассуждения. Не считай недостатком краткость. Если деталь помогает понять ответ
   (например, применимая редакция), не удаляй её.

   Включай в ответ только элементы того же обязательства: размер, срок, порядок, состав,
   начальное событие. Посторонние последствия и процедуры (претензии, расторжение, неустойка,
   согласование, архив, реквизиты) — лишние, если вопрос о них не спрашивает. Если сомневаешься,
   считай деталь лишней, кроме случая, когда она входит в обязательный слот вопроса.
   Сравнение редакций для датированного вопроса — не лишнее, если оно объясняет, почему
   новая редакция не применяется к событию.

   Типичные примеры лишнего:
   - адрес и площадь помещения, когда спрашивают о ночном доступе;
   - срок ответа на претензию, когда спрашивают, что приложить к претензии;
   - сроки согласования и СЭД, когда спрашивают, кто запускает согласование;
   - сведения о поставщике и предмете договора, когда спрашивают о гарантийном сроке;
   - порядок выдачи из архива, когда спрашивают о передаче оригинала в юридический отдел;
   - повтор гарантийного срока из истории, когда спрашивают о сроке проверки качества;
   - правило индексации, когда спрашивают только о размере и сроке платежа.

4. ССЫЛКИ. Соответствует ли каждая ссылка [En] тому утверждению, рядом с которым она
   стоит? Если ссылка подтверждает не это утверждение — укажи.

Правила:
- Для каждого тезиса проверь субъект, действие, модальность (вправе/обязан), отрицание,
  начальное событие срока, величину, предел, исключения и применимость редакции.
- Непустой evidence не означает наличие ответа. Различай достаточное, частичное,
  только тематически близкое и противоречивое основание. Обоснованный отказ допустим.
- Утверждение без ссылки тоже проверяй. Отказ не должен утверждать отсутствие нормы
  во всей базе, если проверены только фрагменты. Обстоятельства пользователя — условия
  задачи, а не доказательства содержания документа.
- Не предлагай добавлять внешние знания, законы или нормы, которых нет в evidence.
- Не требуй дословного совпадения с evidence.
- В drop и unsupported копируй ТОЧНУЮ фразу из <draft> (одно предложение или его часть),
  а не пересказ. Не возвращай весь черновик.
- Если всё в порядке, верни пустые списки и verdict "ok".

Верни строго один JSON-объект без markdown:
{{"verdict": "ok" или "revise", "answerability": "sufficient|partial|related_only|contradictory", "claims": [{{"claim": "точный тезис из draft", "evidence_ids": ["E1"], "status": "supported|unsupported|wrong_reference"}}], "drop": [], "unsupported": [], "missing": [], "wrong_refs": [], "reason": "одно предложение"}}
"""


@dataclass
class Review:
    ok: bool = True
    missing: list[str] = field(default_factory=list)
    drop: list[str] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)
    wrong_refs: list[str] = field(default_factory=list)
    reason: str = ""
    skipped: bool = False
    answerability: str = "unknown"

    @property
    def noise(self) -> list[str]:
        """Совместимый псевдоним: старые вызовы и отчёты используют noise."""
        return self.drop

    @property
    def issues(self) -> list[str]:
        return self.missing + self.drop + self.unsupported + self.wrong_refs


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return []
    out: list[str] = []
    for item in value:
        text = re.sub(r"\s+", " ", str(item)).strip(" -•\t")
        if text and text.lower() not in ("нет", "нет.", "none", "null", "-"):
            out.append(text[:300])
    return out[:8]


def _parse_json(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip(), flags=re.S).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def review_answer(
    client: Any,
    question: str,
    items: Sequence[dict[str, Any]],
    draft: str,
    *,
    max_tokens: int = 800,
) -> Review:
    if not draft.strip() or not items:
        return Review(skipped=True)

    evidence = render_evidence(items)
    prompt = PROMPT.format(
        question=(question or "").strip(),
        evidence=evidence[:MAX_EVIDENCE_CHARS],
        draft=draft.strip(),
    )
    try:
        raw = client.complete(
            [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=max_tokens,
        )
        data = _parse_json(raw)
        if not isinstance(data, dict) or data.get("verdict") not in ("ok", "revise"):
            raise ValueError("invalid_review_schema")
        if "answerability" in data and data["answerability"] not in ("sufficient", "partial", "related_only", "contradictory"):
            raise ValueError("invalid_answerability")
        for key in ("missing", "drop", "noise", "unsupported", "wrong_refs"):
            if key in data and (not isinstance(data[key], list) or not all(isinstance(x, str) for x in data[key])):
                raise ValueError("invalid_review_list")
        if "claims" in data:
            if not isinstance(data["claims"], list):
                raise ValueError("invalid_claims")
            for claim in data["claims"]:
                if not isinstance(claim, dict) or not isinstance(claim.get("claim"), str):
                    raise ValueError("invalid_claim")
                if claim.get("status") not in ("supported", "unsupported", "wrong_reference"):
                    raise ValueError("invalid_claim_status")
                ids = claim.get("evidence_ids")
                if not isinstance(ids, list) or any(eid not in {i["eid"] for i in items} for eid in ids):
                    raise ValueError("invalid_claim_citation")
                if claim["status"] != "supported":
                    key = "wrong_refs" if claim["status"] == "wrong_reference" else "unsupported"
                    data.setdefault(key, []).append(claim["claim"])
    except Exception:  # редактор не должен ронять прогон
        return Review(skipped=True)

    review = Review(
        missing=_as_list(data.get("missing")),
        drop=_as_list(data.get("drop")) or _as_list(data.get("noise")),
        unsupported=_as_list(data.get("unsupported")),
        wrong_refs=_as_list(data.get("wrong_refs")),
        reason=str(data.get("reason") or "")[:400],
        answerability=data.get("answerability", "unknown"),
    )
    verdict = str(data.get("verdict") or "").strip().lower()
    review.ok = not review.issues and verdict in ("", "ok", "accept", "pass")
    return review


def _sentence_contains(sentence: str, phrase: str) -> bool:
    sentence_key = _word_key(sentence)
    phrase_key = _word_key(phrase)
    if not phrase_key or not sentence_key:
        return False
    return phrase_key in sentence_key or (
        len(sentence_key) > 12 and sentence_key in phrase_key
    )


def prune_reviewed_answer(draft: str, review: Review) -> str:
    """Удаляет из черновика ровно те предложения, которые редактор назвал лишними.

    Редактор не переписывает ответ целиком: это главный источник регрессий.
    Если после удаления не осталось ни одного содержательного предложения,
    возвращается исходный черновик — лучше шумный ответ, чем пустой.
    """
    text = (draft or "").strip()
    if not text or review.skipped:
        return text
    phrases = [
        phrase for phrase in review.drop + review.unsupported + review.wrong_refs
        if len(_word_key(phrase)) >= 10
    ]
    if not phrases:
        return text

    sentences = split_sentences(text)
    kept = [
        sentence for sentence in sentences
        if not any(_sentence_contains(sentence, phrase) for phrase in phrases)
    ]
    if not kept or not any(len(_word_key(sentence)) >= 12 for sentence in kept):
        # A negative factual verdict must never restore the rejected draft.
        return "" if review.unsupported or review.wrong_refs else text
    return " ".join(kept)


def build_missing_repair_message(review: Review, draft: str) -> dict[str, str]:
    """Адресный repair: добавляет только пропущенные обязательные элементы."""
    return {
        "role": "user",
        "content": (
            "В ответе не хватает обязательных элементов:\n"
            + "\n".join(f"- {item}" for item in review.missing)
            + "\n\nДобавь их, только если они прямо подтверждены <evidence>. "
            "Не меняй остальные предложения и не добавляй новых тем. "
            "Если нужного факта в <evidence> нет — назови пробел в данных. "
            "Верни только исправленный ответ, до 180 слов.\n\nТекущий ответ:\n"
            + draft.strip()
        ),
    }


def build_review_repair_message(review: Review) -> dict[str, str]:
    blocks: list[str] = []
    if review.missing:
        blocks.append(
            "Полнота — добавь только то, что есть в <evidence>:\n"
            + "\n".join(f"- {item}" for item in review.missing)
        )
    if review.unsupported:
        blocks.append(
            "Нет опоры в доказательствах — удали или переформулируй строго по тексту:\n"
            + "\n".join(f"- {item}" for item in review.unsupported)
        )
    if review.noise:
        blocks.append(
            "Лишнее — удали, чтобы ответ не размывался:\n"
            + "\n".join(f"- {item}" for item in review.noise)
        )
    if review.wrong_refs:
        blocks.append(
            "Ссылки — исправь, чтобы [En] стояла после того утверждения, которое она подтверждает:\n"
            + "\n".join(f"- {item}" for item in review.wrong_refs)
        )
    return {
        "role": "user",
        "content": (
            "Твой ответ проверил редактор. Замечания:\n\n"
            + "\n\n".join(blocks)
            + "\n\nПерепиши ответ целиком, устранив все замечания. Сохрани проверенные факты и ссылки. "
            "Если для «полноты» нужного факта нет в <evidence>, не добавляй его — назови пробел в данных. "
            "Не упоминай редактора и проверку, не извиняйся. Только исправленный ответ, до 180 слов."
        ),
    }

MINIMALITY_SYSTEM = (
    "Ты — строгий редактор, который сокращает юридический ответ до необходимого минимума. "
    "Не добавляй ничего и не переписывай ответ. Отвечай только JSON."
)

MINIMALITY_PROMPT = """Сократи ответ до минимума, необходимого для правильного ответа на вопрос.

<question>
{question}
</question>

<draft>
{draft}
</draft>

Оставь:
1) прямой ответ на вопрос;
2) условия того же обязательства, без которых ответ неверен: размер, срок, порядок,
   состав, начальное событие;
3) применимую редакцию или дату, если она влияет на вывод;
4) указание пробела в данных, если ответа нет.

Удали:
- сроки и процедуры других процессов (согласование, СЭД, архив, претензии, расторжение);
- реквизиты, адреса, площади, сведения о других договорах;
- повторы одной и той же мысли и мета-выводы («то есть», «таким образом»);
- правила для других ситуаций, сроков договора или категорий работников.

Не удаляй:
- отдельные составляющие платежа (коммунальные платежи при вопросе об арендной плате);
- противопоставленный вид приёмки из того же пункта (количество при вопросе о качестве);
- сравнение редакций для датированного вопроса, если оно объясняет, почему новая
  редакция не применяется.

Если удалять нечего, верни пустой список.
Верни строго JSON без markdown:
{{"drop": ["точная фраза из draft"]}}
"""


def review_minimality(
    client: Any,
    question: str,
    draft: str,
    *,
    max_tokens: int = 350,
) -> Review:
    """Второе мнение: только удаление лишнего, без переписывания и добавлений."""
    if len(re.findall(r"\w+", draft or "")) < 25:
        return Review(skipped=True)
    try:
        raw = client.complete(
            [
                {"role": "system", "content": MINIMALITY_SYSTEM},
                {"role": "user", "content": MINIMALITY_PROMPT.format(question=question, draft=draft.strip())},
            ],
            temperature=0.0,
            max_tokens=max_tokens,
        )
        data = _parse_json(raw)
    except Exception:
        return Review(skipped=True)
    drop = _as_list(data.get("drop"))
    return Review(drop=drop, ok=not drop)

