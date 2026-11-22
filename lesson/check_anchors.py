"""Check that file:line anchors in the lesson text still resolve.

The lessons point at `vllm/...:123` style anchors to describe the source. When
the repo moves, those line numbers drift. Run this after switching branches or
rebasing:

    python lesson/check_anchors.py

Exit code is 1 if any referenced file is missing or the line is out of range.
With `--show`, print the source line each anchor currently lands on so the
drift can be reviewed by hand.
"""

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LESSON_ROOT = Path(__file__).resolve().parent

TITLE_RE = re.compile(r'title="([^"]+?):(\d+)(?:-\d+)?(?:"|\b)')
INLINE_RE = re.compile(r"`([A-Za-z0-9_./-]+\.(?:py|md|in|toml|cc|cpp|cu)):(\d+)`")
KNOWN_PREFIXES = ("vllm/", "docs/", "tests/", "csrc/", "requirements/")


def iter_refs(text: str):
    for match in TITLE_RE.finditer(text):
        path, line = match.group(1), int(match.group(2))
        if path.startswith(KNOWN_PREFIXES):
            yield path, line
    for match in INLINE_RE.finditer(text):
        path, line = match.group(1), int(match.group(2))
        if path.startswith(KNOWN_PREFIXES):
            yield path, line


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--show", action="store_true", help="print the line each anchor points at"
    )
    args = parser.parse_args()

    failures: list[str] = []
    checked = 0

    for readme in sorted(LESSON_ROOT.glob("*/README.md")):
        rel = readme.relative_to(REPO_ROOT)
        for path, line in iter_refs(readme.read_text(encoding="utf-8")):
            checked += 1
            target = REPO_ROOT / path
            if not target.exists():
                failures.append(f"{rel}: missing file {path}:{line}")
                continue
            lines = target.read_text(encoding="utf-8", errors="ignore").splitlines()
            if line > len(lines):
                failures.append(
                    f"{rel}: {path}:{line} out of range (file has {len(lines)} lines)"
                )
                continue
            if args.show:
                print(f"{path}:{line}: {lines[line - 1].strip()}")

    print(f"checked {checked} anchors in {LESSON_ROOT}")
    for failure in failures:
        print(f"FAIL {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
