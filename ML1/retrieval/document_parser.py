from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

from .metadata import extract_metadata, link_documents
from .schemas import ParsedDocument


def load_documents(source: str | Path) -> tuple[list[ParsedDocument], list[dict]]:
    """Read a KB directory or its ZIP without extracting archive members."""
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"Knowledge base not found: {source}")
    documents, issues, paths = [], [], set()

    def read_one(relative: str, raw: bytes) -> None:
        path = PurePosixPath(relative.replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts or ":" in path.parts[0]:
            raise ValueError("Unsafe path in knowledge base ZIP")
        parts = path.parts[1:] if path.parts[0].casefold() == "knowledge-base" else path.parts
        key = "/".join(("knowledge-base", *parts))
        if key in paths:
            raise ValueError(f"Duplicate knowledge base member: {key}")
        paths.add(key)
        if path.suffix.lower() not in {".txt", ".md"}:
            issues.append({"file_path": key, "issue": "unsupported_format"})
            return
        try:
            content = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            issues.append({"file_path": key, "issue": "invalid_utf8"})
            return
        if not content.strip() or "\x00" in content:
            issues.append({"file_path": key, "issue": "empty_or_binary"})
            return
        meta = extract_metadata(key, content, hashlib.sha256(raw).hexdigest())
        documents.append(ParsedDocument(meta, content))

    if source.is_dir():
        for path in sorted(source.rglob("*")):
            if path.is_file() and not path.is_symlink() and not path.name.startswith("."):
                read_one(path.relative_to(source).as_posix(), path.read_bytes())
    else:
        with ZipFile(source) as archive:
            for item in sorted(archive.infolist(), key=lambda entry: entry.filename):
                if not item.is_dir() and not any(
                    part.startswith(".") or part == "__MACOSX" for part in PurePosixPath(item.filename).parts
                ):
                    read_one(item.filename, archive.read(item))

    if not documents:
        raise ValueError(f"No readable UTF-8 TXT/MD documents in {source}")
    documents.sort(key=lambda doc: doc.metadata["file_path"])
    link_documents(documents)
    return documents, issues
