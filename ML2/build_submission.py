"""Сборка плоского ZIP-сабмита по техническому регламенту ПСБ.

Регламент: в архив кладутся только корневые .py-файлы и requirements.txt,
вложенные каталоги не переносятся. Поэтому пакет llmguard/ и prompts/*.md
разворачиваются в корень, а системный промпт зашивается в system_prompt.py.

    python3 build_submission.py            # dist/ + dist/psb_legal_assistant.zip
    python3 build_submission.py --check    # плюс прогон тестов перед сборкой
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIST = ROOT / "dist"
ZIP_NAME = "psb_legal_assistant.zip"

MODULES = ("evidence.py", "prompt_builder.py", "validator.py", "client.py", "csvio.py")
PROMPT = ROOT / "prompts" / "system_prompt.md"
RETRIEVAL_RUNTIME = ROOT.parent / "ML1" / "retrieval_runtime.py"

REQUIREMENTS = """# Внешние зависимости не нужны: только стандартная библиотека Python 3.10+.
# retrieval_runtime.py (ML1) также не требует сторонних пакетов.
"""


def _prompt_module() -> str:
    text = PROMPT.read_text(encoding="utf-8").strip()
    escaped = text.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
    return (
        '"""Системный промпт юридического ассистента (встроен для ZIP-сабмита).\n\n'
        "Правится в ML2/prompts/system_prompt.md, отсюда только генерируется.\n"
        '"""\n\n'
        f'SYSTEM_PROMPT = """{escaped}"""\n'
    )


def _rewrite_imports(source: str) -> str:
    return source.replace("from llmguard.", "from ")


def build() -> Path:
    if DIST.exists():
        shutil.rmtree(DIST)
    DIST.mkdir()

    for name in MODULES:
        src = ROOT / "llmguard" / name
        (DIST / name).write_text(_rewrite_imports(src.read_text(encoding="utf-8")), encoding="utf-8")

    (DIST / "system_prompt.py").write_text(_prompt_module(), encoding="utf-8")

    solution = _rewrite_imports((ROOT / "solution.py").read_text(encoding="utf-8"))
    (DIST / "solution.py").write_text(solution, encoding="utf-8")

    if not RETRIEVAL_RUNTIME.is_file():
        raise SystemExit(f"Нет {RETRIEVAL_RUNTIME}. Сначала сгенерируйте его в ML1.")
    shutil.copy2(RETRIEVAL_RUNTIME, DIST / "retrieval_runtime.py")

    (DIST / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")

    archive = DIST / ZIP_NAME
    files = sorted(p for p in DIST.iterdir() if p.is_file() and p.name != ZIP_NAME)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(path, arcname=path.name)  # строго корень архива
    return archive


def check_archive(archive: Path) -> None:
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
    assert all("/" not in name for name in names), f"вложенные пути в архиве: {names}"
    assert "solution.py" in names, "нет solution.py в корне архива"
    assert "requirements.txt" in names, "нет requirements.txt"
    assert not any(name.endswith(".env") for name in names), "в архив попал .env"
    assert not any("__pycache__" in name for name in names), "в архив попал кэш"
    print(f"Архив корректен: {len(names)} файлов в корне — {', '.join(names)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Сборка ZIP-сабмита (ML2)")
    parser.add_argument("--check", action="store_true", help="прогнать тесты перед сборкой")
    args = parser.parse_args(argv)

    if args.check:
        result = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests"],
            cwd=ROOT,
        )
        if result.returncode != 0:
            print("Тесты упали — сборка отменена.", file=sys.stderr)
            return result.returncode

    archive = build()
    check_archive(archive)
    size = archive.stat().st_size / 1024
    print(f"Готово: {archive} ({size:.0f} КБ)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
