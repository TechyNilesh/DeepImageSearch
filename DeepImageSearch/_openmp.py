# SPDX-License-Identifier: MIT
# Copyright (c) 2021 Nilesh Verma
"""
macOS OpenMP duplicate-runtime mitigation.

The `torch` and `faiss-cpu` wheels each vendor their own copy of libomp.dylib
(torch/lib/libomp.dylib and faiss/.dylibs/libomp.dylib). Loading both into one
process is unsupported, and it fails in two different ways:

1. Without KMP_DUPLICATE_LIB_OK the process aborts on the first FAISS call
   ("OMP: Error #15").
2. With KMP_DUPLICATE_LIB_OK=TRUE the abort is suppressed, but FAISS's
   OpenMP-parallel routines — k-means training for IVF indexes above all —
   segfault instead.

So the env var alone is not enough. When the duplicate is present we also pin
FAISS to a single OpenMP thread, which keeps the parallel routines off the
conflicting runtime and makes them correct rather than fatal. This costs FAISS
throughput on large indexes, so it is applied only when both copies are
actually installed, and only on macOS.

The real fix is to have one libomp in the environment; see
`duplicate_libomp_hint()` for how. Set DEEPIMAGESEARCH_FAISS_THREADS to override
the thread pinning once you have done that.
"""

import logging
import os
import sys
from importlib.util import find_spec

logger = logging.getLogger(__name__)

_THREADS_ENV = "DEEPIMAGESEARCH_FAISS_THREADS"


def _vendored_libomp_paths():
    """Locate each package's vendored libomp without importing the packages."""
    paths = []
    for module, relative in (("torch", "lib/libomp.dylib"), ("faiss", ".dylibs/libomp.dylib")):
        try:
            spec = find_spec(module)
        except (ImportError, ValueError):
            continue
        if spec is None or not spec.origin:
            continue
        candidate = os.path.join(os.path.dirname(spec.origin), *relative.split("/"))
        if os.path.exists(candidate):
            paths.append(os.path.realpath(candidate))
    return paths


def has_duplicate_libomp() -> bool:
    """True when torch and faiss ship distinct OpenMP runtimes on this machine."""
    if sys.platform != "darwin":
        return False
    return len(set(_vendored_libomp_paths())) > 1


def duplicate_libomp_hint() -> str:
    """Human-readable instructions for removing the duplicate runtime."""
    return (
        "torch and faiss-cpu each vendor their own libomp.dylib. DeepImageSearch "
        "works around this by allowing the duplicate and pinning FAISS to one "
        "thread. To remove the conflict (and restore FAISS multithreading), point "
        "one wheel at the other's runtime:\n"
        "  cd \"$(python -c 'import faiss,os;print(os.path.dirname(faiss.__file__))')/.dylibs\"\n"
        "  mv libomp.dylib libomp.dylib.bak\n"
        "  ln -s \"$(python -c 'import torch,os;print(os.path.dirname(torch.__file__))')/lib/libomp.dylib\" .\n"
        f"Then set {_THREADS_ENV}=0 to let FAISS use all cores again."
    )


def configure_environment() -> None:
    """
    Allow the duplicate runtime to load. Must run before torch or faiss is
    imported, so this is called at the top of the package's __init__.
    """
    if sys.platform != "darwin":
        return
    # setdefault: an explicit value in the environment always wins.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def configure_faiss(faiss_module) -> None:
    """
    Pin FAISS to a single OpenMP thread when the duplicate runtime is present.

    Called right after `import faiss`. Without this, IVF training segfaults on
    macOS whenever torch is also loaded.
    """
    override = os.environ.get(_THREADS_ENV)
    if override is not None:
        threads = int(override)
        if threads > 0:
            faiss_module.omp_set_num_threads(threads)
        return

    if not has_duplicate_libomp():
        return

    try:
        faiss_module.omp_set_num_threads(1)
    except AttributeError:  # pragma: no cover — very old faiss builds
        return

    # One concise line at WARNING; the full instructions are a level down, so
    # importing the package does not dump a paragraph into every user's stderr.
    logger.warning(
        "FAISS pinned to 1 thread: torch and faiss-cpu ship duplicate libomp "
        "runtimes on macOS, and leaving FAISS multithreaded segfaults. Set "
        "%s=0 once the duplicate is removed (logger '%s' at INFO explains how).",
        _THREADS_ENV, __name__,
    )
    logger.info("%s", duplicate_libomp_hint())
