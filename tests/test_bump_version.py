# SPDX-License-Identifier: MIT
# Copyright (c) 2021 Nilesh Verma
"""
Tests for scripts/bump_version.py.

A missed substitution here ships a release tagged with one version and packaged
as another, which PyPI will not let you take back — so the script fails loudly
rather than silently skipping a file, and that behaviour is pinned below.
"""

import importlib.util
import pathlib
import re

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "bump_version.py"


def load_script():
    spec = importlib.util.spec_from_file_location("bump_version", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bump_version = load_script()


@pytest.fixture
def fake_repo(tmp_path):
    (tmp_path / "DeepImageSearch").mkdir()
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "DeepImageSearch"\nversion = "3.0.2"\nrequires-python = ">=3.10"\n',
        encoding="utf-8",
    )
    (tmp_path / "DeepImageSearch" / "__init__.py").write_text(
        '"""docstring"""\n\n__version__ = "3.0.2"\n\nimport os\n', encoding="utf-8"
    )
    (tmp_path / "CITATION.cff").write_text(
        'cff-version: 1.2.0\ntitle: "DeepImageSearch"\nlicense: MIT\n'
        'version: 3.0.2\ndate-released: "2026-04-01"\n',
        encoding="utf-8",
    )
    return tmp_path


class TestBump:
    def test_updates_all_three_files(self, fake_repo):
        updated = bump_version.bump("3.1.0", "2026-09-01", root=fake_repo)
        assert set(updated) == {"pyproject.toml", "DeepImageSearch/__init__.py", "CITATION.cff"}
        assert 'version = "3.1.0"' in updated["pyproject.toml"]
        assert '__version__ = "3.1.0"' in updated["DeepImageSearch/__init__.py"]
        assert "version: 3.1.0" in updated["CITATION.cff"]

    def test_updates_the_release_date(self, fake_repo):
        updated = bump_version.bump("3.1.0", "2026-09-01", root=fake_repo)
        assert 'date-released: "2026-09-01"' in updated["CITATION.cff"]

    def test_leaves_other_content_alone(self, fake_repo):
        updated = bump_version.bump("3.1.0", "2026-09-01", root=fake_repo)
        assert 'requires-python = ">=3.10"' in updated["pyproject.toml"]
        assert "import os" in updated["DeepImageSearch/__init__.py"]
        assert "cff-version: 1.2.0" in updated["CITATION.cff"]

    def test_does_not_touch_the_cff_schema_version(self, fake_repo):
        # `cff-version:` also matches a careless `version:` pattern.
        updated = bump_version.bump("3.1.0", "2026-09-01", root=fake_repo)
        assert "cff-version: 1.2.0" in updated["CITATION.cff"]
        assert "cff-version: 3.1.0" not in updated["CITATION.cff"]

    def test_writes_nothing_itself(self, fake_repo):
        bump_version.bump("3.1.0", "2026-09-01", root=fake_repo)
        assert 'version = "3.0.2"' in (fake_repo / "pyproject.toml").read_text(encoding="utf-8")

    def test_a_missing_version_line_is_fatal(self, fake_repo):
        (fake_repo / "pyproject.toml").write_text("[project]\nname = 'x'\n", encoding="utf-8")
        with pytest.raises(SystemExit, match="no version line"):
            bump_version.bump("3.1.0", "2026-09-01", root=fake_repo)

    def test_a_missing_file_is_fatal(self, fake_repo):
        (fake_repo / "CITATION.cff").unlink()
        with pytest.raises(FileNotFoundError):
            bump_version.bump("3.1.0", "2026-09-01", root=fake_repo)


class TestCli:
    @pytest.mark.parametrize("bad", ["3.1", "v3.1.0", "3.1.0-rc1", "latest"])
    def test_rejects_malformed_versions(self, bad):
        with pytest.raises(SystemExit):
            bump_version.main([bad])

    def test_dry_run_writes_nothing(self, capsys, monkeypatch, fake_repo):
        monkeypatch.setattr(bump_version, "REPO_ROOT", fake_repo)
        assert bump_version.main(["3.1.0", "--dry-run"]) == 0
        assert "dry run" in capsys.readouterr().out
        assert 'version = "3.0.2"' in (fake_repo / "pyproject.toml").read_text(encoding="utf-8")

    def test_writes_every_file_and_prints_next_steps(self, capsys, monkeypatch, fake_repo):
        monkeypatch.setattr(bump_version, "REPO_ROOT", fake_repo)
        assert bump_version.main(["3.1.0", "--date", "2026-09-01"]) == 0

        assert 'version = "3.1.0"' in (fake_repo / "pyproject.toml").read_text(encoding="utf-8")
        assert '__version__ = "3.1.0"' in (fake_repo / "DeepImageSearch" / "__init__.py").read_text(encoding="utf-8")
        assert "version: 3.1.0" in (fake_repo / "CITATION.cff").read_text(encoding="utf-8")
        assert "git tag v3.1.0" in capsys.readouterr().out

    def test_defaults_the_date_to_today(self, monkeypatch, fake_repo):
        import datetime

        monkeypatch.setattr(bump_version, "REPO_ROOT", fake_repo)
        bump_version.main(["3.1.0"])
        today = datetime.date.today().isoformat()
        assert f'date-released: "{today}"' in (fake_repo / "CITATION.cff").read_text(encoding="utf-8")


def test_script_agrees_with_the_live_repo_layout():
    """Guards against the real files drifting away from the script's patterns."""
    import DeepImageSearch

    updated = bump_version.bump("9.9.9", "2099-01-01", root=REPO_ROOT)
    assert 'version = "9.9.9"' in updated["pyproject.toml"]
    assert '__version__ = "9.9.9"' in updated["DeepImageSearch/__init__.py"]
    assert "version: 9.9.9" in updated["CITATION.cff"]
    # ...and the current version is still the one the package reports.
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert re.search(r'^version = "([^"]+)"', pyproject, re.MULTILINE).group(1) == DeepImageSearch.__version__
