#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2021 Nilesh Verma
"""
Bump the project version in every place it is recorded.

The version lives in three files, and tests/test_package.py asserts they agree:
    pyproject.toml            version = "X.Y.Z"
    DeepImageSearch/__init__  __version__ = "X.Y.Z"
    CITATION.cff              version: X.Y.Z  (plus date-released)

Usage:
    python scripts/bump_version.py 3.0.3
    python scripts/bump_version.py 3.0.3 --date 2026-09-01
    python scripts/bump_version.py 3.0.3 --dry-run

Then commit, tag, and push — the release workflow does the rest:
    git commit -am "Bump version to 3.0.3"
    git tag v3.0.3 && git push && git push --tags
"""

from __future__ import annotations

import argparse
import datetime
import pathlib
import re
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

SEMVER = re.compile(r"^\d+\.\d+\.\d+$")


def replace_once(text: str, pattern: str, replacement: str, path: pathlib.Path) -> str:
    """Substitute exactly one match, or fail loudly — a silent miss ships a bad release."""
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise SystemExit(f"error: no version line matching {pattern!r} in {path}")
    return updated


def bump(version: str, released: str, root: pathlib.Path = REPO_ROOT) -> dict[str, str]:
    """Return {relative path: new content} for every file that records the version."""
    edits = {
        "pyproject.toml": (r'^version = "[^"]+"', f'version = "{version}"'),
        "DeepImageSearch/__init__.py": (r'^__version__ = "[^"]+"', f'__version__ = "{version}"'),
        "CITATION.cff": (r"^version: .+$", f"version: {version}"),
    }

    result = {}
    for relative, (pattern, replacement) in edits.items():
        path = root / relative
        text = path.read_text(encoding="utf-8")
        result[relative] = replace_once(text, pattern, replacement, path)

    # CITATION.cff also carries the release date.
    result["CITATION.cff"] = replace_once(
        result["CITATION.cff"],
        r"^date-released: .+$",
        f'date-released: "{released}"',
        root / "CITATION.cff",
    )
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("version", help="new version, e.g. 3.0.3")
    parser.add_argument("--date", help="release date for CITATION.cff (default: today)")
    parser.add_argument("--dry-run", action="store_true", help="print what would change, write nothing")
    args = parser.parse_args(argv)

    if not SEMVER.match(args.version):
        parser.error(f"version must look like X.Y.Z, got {args.version!r}")

    released = args.date or datetime.date.today().isoformat()
    updated = bump(args.version, released)

    for relative, content in updated.items():
        path = REPO_ROOT / relative
        if args.dry_run:
            print(f"would update {relative}")
            continue
        path.write_text(content, encoding="utf-8")
        print(f"updated {relative}")

    if args.dry_run:
        print(f"\ndry run — nothing written (version {args.version}, date {released})")
        return 0

    print(
        f"\nVersion set to {args.version}. Next:\n"
        f"  1. Add a {args.version} section to CHANGELOG.md\n"
        f"  2. git commit -am 'Bump version to {args.version}'\n"
        f"  3. git tag v{args.version} && git push && git push --tags\n"
        f"The release workflow builds, verifies, and publishes to PyPI."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
