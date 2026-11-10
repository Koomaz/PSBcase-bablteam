from __future__ import annotations

import hashlib
import re
import unicodedata
from datetime import date
from calendar import monthrange

MONTHS = (
    "январ феврал март апрел ма июн июл август сентябр октябр ноябр декабр"
).split()
MONTH = r"(?:" + "|".join(MONTHS) + r")[а-яё]*"
RUS_DATE = rf"\b(\d{{1,2}})\s+({MONTH})\s+(\d{{4}})(?!\d)"
NUM_DATE = r"\b(\d{1,2})[./](\d{1,2})[./](\d{4})(?!\d)"
ISO_DATE = r"\b(\d{4})-(\d{2})-(\d{2})(?!\d)"
DATE_PATTERN = re.compile(f"{ISO_DATE}|{NUM_DATE}|{RUS_DATE}", re.I)
DOC_NUM = r"[A-ZА-ЯЁ]{1,12}\s*-\s*\d+(?:\s*[/–—-]\s*\d{4})?|\d+\s*[-–—]?\s*[Фф][Зз]"
ARTICLE_HEADING = re.compile(r"^Статья\s+(\d+(?:\.\d+)*)\.\s+(.+)", re.I)


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold().replace("ё", "е")
    value = re.sub(r"[‐‑‒–—−]", "-", value)
    return " ".join(value.split())


def canonical_number(value: str) -> str:
    value = normalize(value).replace(" ", "")
    value = re.sub(r"(?<=\d)-(?=\d{4}$)", "/", value)
    value = re.sub(r"^(\d+)фз$", r"\1-фз", value)
    # Latinate spellings commonly used for Cyrillic contract prefixes.
    head, separator, tail = value.partition("-")
    return {"a": "а", "p": "п", "pt": "пт", "k": "к"}.get(head, head) + separator + tail


def stable_id(prefix: str, value: str) -> str:
    return prefix + hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]


def dates_in(text: str) -> list[tuple[int, date]]:
    found = []
    for match in DATE_PATTERN.finditer(text):
        iso_year, iso_month, iso_day, day, month, year, rus_day, rus_month, rus_year = match.groups()
        try:
            if iso_year:
                result = date(int(iso_year), int(iso_month), int(iso_day))
            elif year:
                result = date(int(year), int(month), int(day))
            else:
                number = next(i for i, stem in enumerate(MONTHS, 1) if rus_month.lower().startswith(stem))
                result = date(int(rus_year), number, int(rus_day))
        except ValueError:
            continue
        found.append((match.start(), result))
    return found


def first_date(text: str) -> str | None:
    found = dates_in(text)
    return found[0][1].isoformat() if found else None


def query_period(text: str) -> tuple[str, str] | None:
    full = [when for _, when in dates_in(text)]
    if full:
        return min(full).isoformat(), max(full).isoformat()
    for match in re.finditer(rf"\b({MONTH})\s+(19\d{{2}}|20\d{{2}})\b", text, re.I):
        word, year = match.groups()
        month = next(i for i, stem in enumerate(MONTHS, 1) if word.lower().startswith(stem))
        return date(int(year), month, 1).isoformat(), date(
            int(year), month, monthrange(int(year), month)[1]
        ).isoformat()
    years = re.findall(r"\b(?:в|за|на|до|после|с)\s+(19\d{2}|20\d{2})(?:\s*г(?:од[ауе]?)?\.?\b|\b)|\b(19\d{2}|20\d{2})\s*год[ауе]?\b", text, re.I)
    if years:
        bounds = [first or second for first, second in years]
        return min(bounds) + "-01-01", max(bounds) + "-12-31"
    return None


def extract_metadata(path: str, text: str, sha256: str) -> dict:
    name = path.rsplit("/", 1)[-1]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    header = lines[:8]
    title = header[0]
    kind, number, doc_date, edition, start, end = "other", None, None, None, None, None
    parent_number, article, law_code, export, source_url, subject = (None,) * 6
    counterparties = []
    if title.startswith("Источник:") or name.startswith("законы__"):
        kind = "law"
        source_url = next((line[4:].strip() for line in header if line.startswith("URL:")), None)
        export = next((first_date(line) for line in header if line.startswith("Дата выгрузки:")), None)
        heading = next((ARTICLE_HEADING.match(line) for line in lines if ARTICLE_HEADING.match(line)), None)
        code = re.match(r"законы__(.*?)_ст(\d+(?:\.\d+)*)_", name, re.I)
        if code:
            law_code, article = code.groups()
            title_code = {
                "ТК": "ТК РФ", "НК": "НК РФ", "КоАП": "КоАП РФ",
                "152ФЗ": "Федеральный закон № 152-ФЗ", "МРОТ": "Федеральный закон о МРОТ",
            }.get(law_code, law_code)
            title = f"{title_code}, ст. {article}"
        if heading:
            article = heading[1]
            subject = heading[2] or None
        if re.fullmatch(r"\d+ФЗ", law_code or "", re.I):
            number = law_code[:-2] + "-ФЗ"
        elif law_code in {"МРОТ", "ТК"}:
            explicit_number = re.search(r"(?:N|№)\s*(\d+-ФЗ)", header[0], re.I)
            number = explicit_number[1] if explicit_number else None
        if re.search(r"(?:Трудовой|Налоговый) кодекс|Федеральный закон от", header[0], re.I):
            doc_date = first_date(header[0])
        prefix = lines[:next((i for i, line in enumerate(lines) if ARTICLE_HEADING.match(line)), len(header))]
        effective_dates = [
            found for line in prefix
            if article and re.match(rf"Статья\s+{re.escape(article)}\s+изменена\s+с\b", line, re.I)
            if (found := first_date(line)) and export and found <= export
        ]
        start = max(effective_dates, default=None)
    else:
        title = next((line for line in header if re.match(
            r"^(?:ДОГОВОР|ДОПОЛНИТЕЛЬНОЕ СОГЛАШЕНИЕ|РЕГЛАМЕНТ|СЛУЖЕБНАЯ ЗАПИСКА)\b", line
        )), title)
        if title.startswith("ДОГОВОР"):
            kind, doc_date = "contract", first_date(header[1]) if len(header) > 1 else None
            term = next((line for line in lines if re.search(r"Договор вступает в силу с даты подписания", line, re.I)), "")
            if term:
                start = doc_date
                if match := re.search(r"действует до\s+(.+)", term, re.I):
                    end_date = first_date(match[1])
                    if end_date:
                        end = date.fromordinal(date.fromisoformat(end_date).toordinal() + 1).isoformat()
        elif title.startswith("ДОПОЛНИТЕЛЬНОЕ"):
            kind = "addendum"
            parent = re.search(rf"к договору[^\n]*?№\s*({DOC_NUM})", "\n".join(header), re.I)
            parent_number = parent[1] if parent else None
            doc_date = first_date(header[2]) if len(header) > 2 else None
            if re.search(r"вступает в силу с даты подписания", text, re.I):
                start = doc_date
            if parent_number:
                title += f" к договору № {parent_number}"
        elif title.startswith("РЕГЛАМЕНТ"):
            kind = "regulation"
            approved = next((line for line in header if line.lower().startswith(("утверждён", "утвержден"))), "")
            doc_date = first_date(approved)
            version = next((line for line in header if line.lower().startswith("редакция от")), "")
            edition = first_date(version)
            active = re.search(r"Действует с\s+(.+)", version, re.I)
            start = first_date(active[1]) if active else None
            end = next((first_date(line) for line in header if re.match(r"РЕДАКЦИЯ УТРАТИЛА СИЛУ с", line, re.I)), None)
        elif title.startswith("СЛУЖЕБНАЯ"):
            kind, doc_date = "memo", first_date(title)
            subject = next((line for line in header if line.startswith("О ")), None)
        elif any(line.startswith("Исх.") for line in header):
            kind = "letter"
            registration = next(line for line in header if line.startswith("Исх."))
            doc_date = first_date(registration)
            title = "Письмо службы комплаенс " + re.sub(r"^Исх\.\s*", "", registration)
            subject = next((line for line in header if line.startswith("О ")), None)
            start = next((first_date(line) for line in lines if re.match(rf"С {RUS_DATE}", line, re.I)), None)
        found_number = re.search(rf"№\s*({DOC_NUM}|\d+)", title, re.I)
        number = found_number[1] if found_number else None
        if kind in {"contract", "addendum"}:
            counterparties = list(dict.fromkeys(re.findall(r'ООО\s+[«"]([^»"]+)[»"]', "\n".join(header), re.I)))

    if kind in {"contract", "addendum"} and doc_date:
        title += f" от {date.fromisoformat(doc_date):%d.%m.%Y}"
    if edition:
        title += f" (ред. от {date.fromisoformat(edition):%d.%m.%Y})"
    return {
        "doc_id": stable_id("doc_", path), "file_name": name, "file_path": path,
        "sha256": sha256, "doc_title": title, "doc_type": kind, "doc_number": number,
        "doc_date": doc_date, "edition_date": edition, "effective_date": start,
        "validity_period": {"from": start, "to_exclusive": end} if start or end else None,
        "edition_group": normalize(title.split(" (ред.")[0]) if kind == "regulation" else None,
        "version_status": "current_snapshot" if kind == "law" and "действующая" in name else (
            "superseded" if kind == "regulation" and end else "current" if kind == "regulation" and start else None
        ),
        "parent_doc_id": None, "parent_doc_number": parent_number, "related_doc_ids": [],
        "counterparties": counterparties, "article": article, "law_code": law_code,
        "snapshot_date": export, "source_url": source_url, "subject": subject,
    }


def link_documents(documents: list) -> None:
    contracts = {}
    for doc in documents:
        meta = doc.metadata
        if meta["doc_type"] == "contract" and meta["doc_number"]:
            contracts.setdefault(canonical_number(meta["doc_number"]), []).append(meta)
    for doc in documents:
        meta = doc.metadata
        parent = meta["parent_doc_number"]
        matches = contracts.get(canonical_number(parent), []) if parent else []
        if len(matches) == 1:
            meta["parent_doc_id"] = matches[0]["doc_id"]
            matches[0]["related_doc_ids"].append(meta["doc_id"])
