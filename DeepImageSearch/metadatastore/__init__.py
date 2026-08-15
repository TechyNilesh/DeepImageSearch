# SPDX-License-Identifier: MIT
# Copyright (c) 2021 Nilesh Verma
from DeepImageSearch.metadatastore.base import ImageRecord, BaseMetadataStore
from DeepImageSearch.metadatastore.json_store import JsonMetadataStore

__all__ = ["ImageRecord", "BaseMetadataStore", "JsonMetadataStore"]

try:
    from DeepImageSearch.metadatastore.postgres_store import PostgresMetadataStore
    __all__.append("PostgresMetadataStore")
except ImportError:
    pass
