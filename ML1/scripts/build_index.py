from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retrieval.artifacts import build_indexes, save_indexes
from retrieval.config import Settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Build retrieval indexes from a directory or KB ZIP.")
    parser.add_argument("--source", type=Path, default=Path("knowledge-base"))
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--offline", action="store_true")
    mode.add_argument("--require-dense", action="store_true")
    args = parser.parse_args()
    data = build_indexes(args.source, Settings.from_env(), dense=not args.offline)
    if args.require_dense and data.dense is None:
        parser.exit(1, f"Dense index required: {data.manifest['known_limitations'][0]}\n")
    save_indexes(data, args.output)
    print(json.dumps({key: data.manifest[key] for key in (
        "document_count", "chunk_count", "embedding_model", "known_limitations", "issues"
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
