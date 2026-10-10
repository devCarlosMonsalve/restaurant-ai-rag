from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Iterable, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def fingerprint_rows(rows: Iterable[Sequence[object]]) -> tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    for row in rows:
        payload = json.dumps(
            list(row),
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        digest.update(len(payload).to_bytes(8, byteorder="big"))
        digest.update(payload)
        count += 1
    return digest.hexdigest(), count


def build_run_metadata(
    *,
    evaluator: str,
    cases_path: Path,
    configuration: dict[str, object],
    corpus_name: str,
    corpus_fingerprint: str,
    corpus_row_count: int,
) -> dict[str, object]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    case_bytes = cases_path.read_bytes()
    return {
        "evaluator": evaluator,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "worktree_dirty": dirty,
        "dataset": {
            "name": cases_path.name,
            "sha256": hashlib.sha256(case_bytes).hexdigest(),
        },
        "corpus": {
            "name": corpus_name,
            "row_count": corpus_row_count,
            "sha256": corpus_fingerprint,
            "fingerprint_includes": "all fields selected by the evaluator, including embeddings",
        },
        "configuration": configuration,
    }


def write_evaluation_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
