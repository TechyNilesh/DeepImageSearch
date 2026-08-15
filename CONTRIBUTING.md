# Contributing to DeepImageSearch

Thanks for your interest in improving DeepImageSearch. Bug reports, documentation
fixes, new backends, and test coverage are all welcome.

## Development setup

```bash
git clone https://github.com/TechyNilesh/DeepImageSearch.git
cd DeepImageSearch
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Add the extras you need for the backend you are working on, e.g.
`pip install -e ".[dev,chroma,qdrant]"`.

## Running the checks

```bash
pytest                                  # full suite, no model downloads, ~1s
pytest --cov=DeepImageSearch            # with coverage
ruff check DeepImageSearch tests        # lint
```

CI runs the same three commands on Python 3.10–3.13 across Linux, macOS, and
Windows. A pull request should be green on all of them.

**macOS note:** `torch` and `faiss-cpu` each vendor their own copy of
`libomp.dylib`. Loading both is unsupported and fails two ways: the process
aborts with `OMP: Error #15`, or — once that abort is suppressed — FAISS's
OpenMP-parallel routines segfault instead (IVF k-means training is the usual
casualty). `DeepImageSearch/_openmp.py` handles both: it sets
`KMP_DUPLICATE_LIB_OK` at import time and pins FAISS to a single thread when it
detects two distinct runtimes.

Two caveats. It only takes effect if `DeepImageSearch` is imported *before*
`torch` or `faiss`. And single-threaded FAISS is slower on large indexes — the
module's docstring documents how to remove the duplicate runtime for real, after
which `DEEPIMAGESEARCH_FAISS_THREADS=0` restores full multithreading.

## Writing tests

Tests must not download model weights — CI runs twelve jobs and cannot afford
it. Use the `embedding` fixture from `tests/conftest.py`, a deterministic
stand-in for CLIP that supports both text and image queries, or monkeypatch the
backend as `tests/test_embeddings.py` does. Anything genuinely requiring weights
belongs behind an explicit opt-in marker.

## Adding a backend

The pluggable layers are all defined by abstract base classes:

| Extension point | Base class | Existing implementations |
|---|---|---|
| Vector store | `DeepImageSearch/vectorstores/base.py` | FAISS, ChromaDB, Qdrant |
| Metadata store | `DeepImageSearch/metadatastore/base.py` | JSON, PostgreSQL |
| Embedding | `DeepImageSearch/core/embeddings.py` | CLIP, timm, custom callable |

To add one:

1. Subclass the relevant base and implement every abstract method.
2. Import it in the package's `__init__.py` inside a `try/except ImportError`
   so the dependency stays optional, and append it to `__all__`.
3. Declare the dependency as a new extra in `pyproject.toml`.
4. Add tests. The interface-conformance tests at the bottom of
   `tests/test_vectorstores.py` and `tests/test_metadata_store.py` show the
   pattern; skip the suite when the optional dependency is absent.
5. Document it in `Documents/` and add a demo to `Demo/` if it changes usage.

## Pull requests

- Branch from `main` and keep each PR to one logical change.
- Every new source file needs the `# SPDX-License-Identifier: MIT` header.
- Update `CHANGELOG.md` under an "Unreleased" heading.
- Version numbers live in `pyproject.toml`, `DeepImageSearch/__init__.py`, and
  `CITATION.cff`; `tests/test_package.py` asserts all three agree.

## Reporting bugs

Open an issue at https://github.com/TechyNilesh/DeepImageSearch/issues with your
OS, Python version, DeepImageSearch version, the backend in use, and a minimal
reproduction.

By contributing you agree that your contributions are licensed under the MIT
License.
