# SPDX-License-Identifier: MIT
# Copyright (c) 2021 Nilesh Verma
"""Package-level guarantees: public API surface, version consistency, licensing."""

import os
import pathlib
import re
import sys

import pytest

import DeepImageSearch

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_version_is_a_semver_string():
    assert re.fullmatch(r"\d+\.\d+\.\d+", DeepImageSearch.__version__)


def test_version_matches_pyproject():
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    declared = re.search(r'^version = "([^"]+)"', pyproject, re.MULTILINE).group(1)
    assert DeepImageSearch.__version__ == declared


def test_version_matches_citation_file():
    citation = (REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    declared = re.search(r"^version: (.+)$", citation, re.MULTILINE).group(1).strip()
    assert DeepImageSearch.__version__ == declared


@pytest.mark.parametrize("name", DeepImageSearch.__all__)
def test_every_exported_name_is_importable(name):
    assert getattr(DeepImageSearch, name, None) is not None


def test_documented_entry_points_are_exported():
    # The README's quick start depends on exactly these names.
    for name in ["SearchEngine", "Load_Data", "Search_Setup"]:
        assert name in DeepImageSearch.__all__


def test_subpackages_import_cleanly():
    import DeepImageSearch.core  # noqa: F401
    import DeepImageSearch.data  # noqa: F401
    import DeepImageSearch.metadatastore as metadatastore
    import DeepImageSearch.vectorstores as vectorstores

    assert "FAISSStore" in vectorstores.__all__
    assert "JsonMetadataStore" in metadatastore.__all__


def test_optional_backends_are_not_hard_requirements():
    # Chroma/Qdrant/Postgres live behind extras; importing the package without
    # them installed must still work.
    import DeepImageSearch.metadatastore as metadatastore
    import DeepImageSearch.vectorstores as vectorstores

    assert "BaseVectorStore" in vectorstores.__all__
    assert "BaseMetadataStore" in metadatastore.__all__


@pytest.mark.skipif(sys.platform != "darwin", reason="OpenMP duplicate-runtime clash is macOS-specific")
def test_macos_openmp_workaround_is_applied_on_import():
    # torch and faiss-cpu vendor separate libomp copies; without this the first
    # FAISS search aborts the process with "OMP: Error #15".
    assert os.environ.get("KMP_DUPLICATE_LIB_OK") == "TRUE"


def test_faiss_search_survives_torch_being_loaded():
    """End-to-end guard for the OpenMP clash: this aborts the process if it regresses."""
    import faiss
    import numpy as np
    import torch  # noqa: F401  — must be loaded for the clash to be possible

    index = faiss.IndexFlatIP(8)
    index.add(np.ones((2, 8), dtype=np.float32))
    scores, indices = index.search(np.ones((1, 8), dtype=np.float32), 1)
    assert indices[0][0] in (0, 1)


@pytest.mark.skipif(sys.platform != "darwin", reason="OpenMP duplicate-runtime clash is macOS-specific")
def test_faiss_is_pinned_to_one_thread_when_runtimes_are_duplicated():
    """
    KMP_DUPLICATE_LIB_OK alone only converts the abort into a segfault inside
    FAISS's parallel routines, so the thread pin has to be in place too.
    """
    import faiss

    from DeepImageSearch._openmp import has_duplicate_libomp

    if has_duplicate_libomp():
        assert faiss.omp_get_max_threads() == 1


def test_ivf_training_survives_torch_being_loaded():
    """IVF k-means is the OpenMP-heaviest path — it segfaults if the guard regresses."""
    import numpy as np
    import torch  # noqa: F401

    from DeepImageSearch.vectorstores.faiss_store import FAISSStore

    store = FAISSStore(dimension=8, index_type="ivf")
    rng = np.random.default_rng(0)
    vectors = rng.random((4000, 8)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    store.add([str(i) for i in range(4000)], vectors)

    assert store.index.is_trained
    assert store.count() == 4000
    assert store.search(vectors[0], k=1)[0]["id"] == "0"


def test_every_source_file_declares_its_license():
    missing = [
        str(path.relative_to(REPO_ROOT))
        for path in sorted((REPO_ROOT / "DeepImageSearch").rglob("*.py"))
        if "SPDX-License-Identifier: MIT" not in path.read_text(encoding="utf-8")
    ]
    assert missing == []
