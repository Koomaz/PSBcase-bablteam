from __future__ import annotations

import re

from .config import MAX_CHARS, OVERLAP
from .metadata import ARTICLE_HEADING, first_date, normalize, stable_id
from .schemas import Chunk, ParsedDocument

SECTION = re.compile(r"^(\d+)\.\s+(.+)")
CLAUSE = re.compile(r"^[«\"“]?(\d+(?:\.\d+)+(?:-\d+)?)\.?\s+")
ENUM = re.compile(r"^(?:\(([а-яa-z]|\d+(?:\.\d+)*(?:-\d+)?)\)|([а-яa-z]|\d+(?:\.\d+)*(?:-\d+)?)\))\s+", re.I)
APPENDIX = re.compile(r"^Приложение\s*(?:№\s*)?(\d+)?(?:[.\s]|$)", re.I)
NOTE = re.compile(
    r"^(?:ГАРАНТ:|См\.\s|Информация об изменениях:|"
    r"(?:Пункт|Подпункт|Часть|Абзац|Статья)\s+[\d.-]+\s+(?:измен[её]н|дополнен|введ[её]н|изложен|в редакции)|"
    r"Федеральн\w+ законами? от |О проверке конституционности |Положения статьи .+ применяются )",
    re.I,
)
DIRECTIVE = re.compile(
    r"^\s*(?:внимание\s+для\s+(?:ии|ai)|(?:системный промпт|system prompt)\s*:)"
    r"|(?:игнорируй(?:те)?|ignore)\s+(?:все\s+)?(?:остальные|предыдущие|prior|previous)"
    r"|не упоминай(?:те)? эту инструкцию",
    re.I,
)


def _windows(text: str, start: int, end: int):
    while end - start > MAX_CHARS:
        lower = start + MAX_CHARS * 2 // 3
        limit = start + MAX_CHARS
        boundary = max(text.rfind("\n", lower, limit), text.rfind(". ", lower, limit) + 1)
        if boundary <= lower:
            boundary = text.rfind(" ", lower, limit)
        if boundary <= lower:
            boundary = limit
        yield start, boundary
        overlap_start = boundary - OVERLAP
        space = text.find(" ", overlap_start, boundary)
        start = space + 1 if space >= 0 else overlap_start
    if end > start:
        yield start, end


def chunk_document(doc: ParsedDocument) -> tuple[list[Chunk], list[dict]]:
    text, base = doc.text, doc.metadata
    kind = base["doc_type"]
    section = clause = subclause = part = appendix = None
    amendment_target = pending_note = clause_effective = None
    clause_parent = None
    article = base["article"]
    chunks, issues = [], []
    span = None
    in_article = kind != "law"
    in_body = kind not in {"memo", "letter", "addendum"}
    law_parts = base["law_code"] in {"152ФЗ", "КоАП"}

    def flush() -> None:
        nonlocal span
        if span is None:
            return
        start, end = span
        span = None
        reference = [base["doc_title"]]
        if article and f"ст. {article}" not in base["doc_title"]:
            reference.append(f"ст. {article}")
        if appendix:
            reference.append(appendix)
        if part:
            reference.append(f"ч. {part}")
        if section and not (clause or part):
            reference.append(f"раздел {section}")
        if clause:
            reference.append(f"п. {clause}")
        if subclause:
            reference.append(f"подп. {subclause}")
        if amendment_target:
            reference.append(f"изменение п. {amendment_target} основного договора")
        source = ", ".join(reference)
        context = [source]
        if section and (clause or part):
            context.append(section)
        if base["subject"]:
            context.append(base["subject"])
        if base["counterparties"]:
            context.append("Стороны: " + "; ".join(base["counterparties"]))
        for left, right in _windows(text, start, end):
            raw = text[left:right]
            application = re.search(r"\b(?:применяется|применяются)\b[^\n]*?\b(после|начиная с)\s+(.+)", raw, re.I)
            meta = {
                **base, "section": section, "article": article, "part": part,
                "clause": clause, "subclause": subclause, "appendix": appendix,
                "amends_clause": amendment_target, "provision_effective_date": clause_effective,
                "amendment_application_date": first_date(application[2]) if amendment_target and application else None,
                "amendment_application_relation": application[1].lower() if amendment_target and application else None,
                "page_start": None, "page_end": None, "char_start": left, "char_end": right,
                "is_table": any(line.count("|") >= 2 for line in raw.splitlines()),
                "is_footnote": bool(re.match(r"^(?:\*+\)|Примечание\b)", raw)),
                "structure_path": [
                    {"level": level, "label": label}
                    for level, label in (("appendix", appendix), ("section", section),
                                         ("article", article), ("part", part),
                                         ("clause", clause), ("subclause", subclause)) if label
                ],
            }
            contextual = "\n".join((*context, raw))
            chunks.append(Chunk(
                stable_id("chunk_", f'{base["doc_id"]}:{left}:{right}'), base["doc_id"],
                raw, contextual, source, normalize(contextual), meta,
            ))

    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        left, right = offset + len(line) - len(line.lstrip()), offset + len(line.rstrip())
        offset += len(line)
        if not stripped:
            if kind == "law":
                flush()
            continue
        if DIRECTIVE.search(stripped):
            flush()
            issues.append({"file_path": base["file_path"], "issue": "quarantined_prompt_directive",
                           "char_start": left, "char_end": right})
            continue
        if kind == "law":
            heading = ARTICLE_HEADING.match(stripped)
            if heading and not NOTE.match(stripped):
                flush()
                article, in_article = heading[1], True
                clause = subclause = part = pending_note = clause_effective = None
                if base["law_code"] == "МРОТ":
                    span = (left, right)
                continue
            if not in_article:
                continue
            if NOTE.match(stripped):
                flush()
                target = re.match(r"^(Пункт|Подпункт|Часть)\s+(\d+(?:\.\d+)*(?:-\d+)?)\s", stripped, re.I)
                if not target:
                    target = re.match(r"^Статья\s+[\d.]+\s+дополнена\s+(пунктом|частью)\s+([\d.-]+)\s", stripped, re.I)
                effective = first_date(stripped)
                if target and effective and (not base["snapshot_date"] or effective <= base["snapshot_date"]):
                    pending_note = (target[1].lower(), target[2], effective)
                continue
        else:
            if kind in {"memo", "letter"} and stripped == base["subject"]:
                in_body = True
                continue
            if kind == "addendum" and stripped.lower().startswith("изложить пункт"):
                in_body = True
            if not in_body:
                continue
            if re.match(r"^(?:Уважаемые коллеги!|Руководитель службы комплаенс|Начальник юридического отдела\s+\S\.)", stripped):
                continue
            if kind == "addendum" and re.match(r"^(?:Поставщик|Покупатель):", stripped):
                flush()
                continue
            attachment = APPENDIX.match(stripped)
            if attachment:
                flush()
                appendix = "Приложение" + (f" {attachment[1]}" if attachment[1] else "")
                section = clause = subclause = None
            if kind == "addendum" and (target := re.match(r"^Изложить пункт\s+(\d+(?:\.\d+)*)", stripped, re.I)):
                flush()
                amendment_target = target[1]
            if kind in {"regulation", "contract"} and (heading := SECTION.match(stripped)):
                flush()
                section, clause, subclause = stripped, None, None
                continue
            if kind in {"regulation", "contract"} and section is None and appendix is None:
                continue

        number = CLAUSE.match(stripped)
        numbered = SECTION.match(stripped)
        enumerated = ENUM.match(stripped)
        if (number or (numbered and kind == "law")) and not (
            kind == "addendum" and number and number[1] == amendment_target
        ):
            flush()
            label = number[1] if number else numbered[1]
            if kind != "addendum":
                if law_parts:
                    part, clause = label, None
                elif kind == "contract" and label.count(".") > 1:
                    clause, subclause = label.rsplit(".", 1)[0], label
                else:
                    clause = label
            if not (kind == "contract" and label.count(".") > 1):
                subclause = None
            clause_parent = None
            clause_effective = None
            if pending_note and pending_note[1] == label and pending_note[0] != "подпункт":
                clause_effective = pending_note[2]
                pending_note = None
        elif enumerated:
            flush()
            if subclause is None:
                clause_parent = clause_effective
            subclause = enumerated[1] or enumerated[2]
            clause_effective = clause_parent
            if pending_note and pending_note[:2] == ("подпункт", subclause):
                clause_effective = max(clause_effective or "", pending_note[2])
                pending_note = None
        elif kind in {"memo", "letter"}:
            flush()
        if span is None:
            span = (left, right)
        else:
            span = (span[0], right)
    flush()
    if not chunks:
        issues.append({"file_path": base["file_path"], "issue": "no_searchable_provisions"})
    return chunks, issues
