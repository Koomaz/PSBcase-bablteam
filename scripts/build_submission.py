"""Build the organizer's flat ZIP using only the explicitly listed files."""

import argparse
import runpy
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]
ML2 = ROOT / "ML2"
FILES = {
    "solution.py": ROOT / "solution.py",
    "psb_backend.py": ROOT / "psb_backend.py",
    "retrieval_runtime.py": ROOT / "ML1" / "retrieval_runtime.py",
    "requirements.txt": ROOT / "requirements.txt",
    "ml2_solution.py": ML2 / "solution.py",
    **{f"{module}.py": ML2 / "llmguard" / f"{module}.py"
       for module in ("client", "csvio", "evidence", "prompt_builder", "validator")},
}
MAX_ZIP_BYTES = 128_000_000


def build(target: Path) -> None:
    ml2_builder = runpy.run_path(str(ML2 / "build_submission.py"))
    files = {name: path.read_bytes() for name, path in FILES.items()}
    files["ml2_solution.py"] = ml2_builder["_rewrite_imports"](files["ml2_solution.py"].decode("utf-8")).encode("utf-8")
    files["system_prompt.py"] = ml2_builder["_prompt_module"]().encode("utf-8")
    target.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=target.parent, delete=False) as stream:
        temp = Path(stream.name)
    try:
        with ZipFile(temp, "w") as archive:
            for name, content in files.items():
                if name.endswith(".py"):
                    compile(content, name, "exec")
                elif content.splitlines() != [b"httpx==0.28.1"]:
                    raise ValueError("Recheck requirements.txt against the organizer's allowed versions")
                info = ZipInfo(name)
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                archive.writestr(info, content, compress_type=ZIP_DEFLATED, compresslevel=9)
        if temp.stat().st_size > MAX_ZIP_BYTES:
            raise ValueError("Submission exceeds 128 MB")
        with ZipFile(temp) as archive:
            if archive.namelist() != list(files) or archive.testzip() is not None:
                raise ValueError("Invalid submission ZIP")
        temp.replace(target)
    finally:
        temp.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist" / "submission.zip")
    args = parser.parse_args()
    build(args.output)
    print(f"{args.output} ({args.output.stat().st_size} bytes)")
