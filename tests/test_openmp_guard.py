# SPDX-License-Identifier: MIT
# Copyright (c) 2021 Nilesh Verma
"""Unit tests for the macOS OpenMP duplicate-runtime mitigation."""

import os
import sys
import types

import pytest

from DeepImageSearch import _openmp


class FakeFaiss:
    def __init__(self):
        self.threads = None

    def omp_set_num_threads(self, n):
        self.threads = n


@pytest.fixture
def fake_faiss():
    return FakeFaiss()


class TestDetection:
    def test_non_macos_platforms_are_never_affected(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        assert _openmp.has_duplicate_libomp() is False

    def test_reports_true_when_two_distinct_runtimes_exist(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setattr(_openmp, "_vendored_libomp_paths", lambda: ["/a/libomp.dylib", "/b/libomp.dylib"])
        assert _openmp.has_duplicate_libomp() is True

    def test_reports_false_when_both_resolve_to_one_file(self, monkeypatch):
        # This is what the documented symlink fix produces.
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setattr(_openmp, "_vendored_libomp_paths", lambda: ["/a/libomp.dylib", "/a/libomp.dylib"])
        assert _openmp.has_duplicate_libomp() is False

    def test_reports_false_when_only_one_package_vendors_one(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setattr(_openmp, "_vendored_libomp_paths", lambda: ["/a/libomp.dylib"])
        assert _openmp.has_duplicate_libomp() is False

    def test_missing_packages_are_tolerated(self, monkeypatch):
        monkeypatch.setattr(_openmp, "find_spec", lambda name: None)
        assert _openmp._vendored_libomp_paths() == []

    def test_unimportable_packages_are_tolerated(self, monkeypatch):
        def explode(name):
            raise ValueError("no spec")

        monkeypatch.setattr(_openmp, "find_spec", explode)
        assert _openmp._vendored_libomp_paths() == []

    def test_namespace_packages_without_an_origin_are_skipped(self, monkeypatch):
        monkeypatch.setattr(_openmp, "find_spec", lambda name: types.SimpleNamespace(origin=None))
        assert _openmp._vendored_libomp_paths() == []


class TestEnvironmentConfiguration:
    def test_sets_the_duplicate_override_on_macos(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.delenv("KMP_DUPLICATE_LIB_OK", raising=False)
        _openmp.configure_environment()
        assert os.environ["KMP_DUPLICATE_LIB_OK"] == "TRUE"

    def test_leaves_other_platforms_alone(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.delenv("KMP_DUPLICATE_LIB_OK", raising=False)
        _openmp.configure_environment()
        assert "KMP_DUPLICATE_LIB_OK" not in os.environ

    def test_an_explicit_user_setting_wins(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setenv("KMP_DUPLICATE_LIB_OK", "FALSE")
        _openmp.configure_environment()
        assert os.environ["KMP_DUPLICATE_LIB_OK"] == "FALSE"


class TestFaissConfiguration:
    def test_pins_to_one_thread_when_duplicated(self, monkeypatch, fake_faiss):
        monkeypatch.delenv("DEEPIMAGESEARCH_FAISS_THREADS", raising=False)
        monkeypatch.setattr(_openmp, "has_duplicate_libomp", lambda: True)
        _openmp.configure_faiss(fake_faiss)
        assert fake_faiss.threads == 1

    def test_leaves_a_healthy_environment_untouched(self, monkeypatch, fake_faiss):
        monkeypatch.delenv("DEEPIMAGESEARCH_FAISS_THREADS", raising=False)
        monkeypatch.setattr(_openmp, "has_duplicate_libomp", lambda: False)
        _openmp.configure_faiss(fake_faiss)
        assert fake_faiss.threads is None

    def test_env_override_sets_an_explicit_thread_count(self, monkeypatch, fake_faiss):
        monkeypatch.setenv("DEEPIMAGESEARCH_FAISS_THREADS", "4")
        monkeypatch.setattr(_openmp, "has_duplicate_libomp", lambda: True)
        _openmp.configure_faiss(fake_faiss)
        assert fake_faiss.threads == 4

    def test_env_override_of_zero_restores_faiss_defaults(self, monkeypatch, fake_faiss):
        monkeypatch.setenv("DEEPIMAGESEARCH_FAISS_THREADS", "0")
        monkeypatch.setattr(_openmp, "has_duplicate_libomp", lambda: True)
        _openmp.configure_faiss(fake_faiss)
        assert fake_faiss.threads is None  # never touched, so faiss keeps its own default

    def test_warning_is_a_single_line(self, monkeypatch, fake_faiss, caplog):
        monkeypatch.delenv("DEEPIMAGESEARCH_FAISS_THREADS", raising=False)
        monkeypatch.setattr(_openmp, "has_duplicate_libomp", lambda: True)
        with caplog.at_level("WARNING", logger=_openmp.__name__):
            _openmp.configure_faiss(fake_faiss)

        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 1
        message = warnings[0].getMessage()
        assert "libomp" in message
        assert "DEEPIMAGESEARCH_FAISS_THREADS" in message
        # Importing the package must not dump a paragraph into stderr.
        assert message.count("\n") == 0

    def test_full_instructions_are_available_at_info(self, monkeypatch, fake_faiss, caplog):
        monkeypatch.delenv("DEEPIMAGESEARCH_FAISS_THREADS", raising=False)
        monkeypatch.setattr(_openmp, "has_duplicate_libomp", lambda: True)
        with caplog.at_level("INFO", logger=_openmp.__name__):
            _openmp.configure_faiss(fake_faiss)
        assert "ln -s" in caplog.text  # the symlink recipe


def test_hint_names_both_packages():
    hint = _openmp.duplicate_libomp_hint()
    assert "torch" in hint and "faiss" in hint
