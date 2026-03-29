"""
JSON file-based metadata store.

Stores image records as a JSON array in `image_records.json`.
Designed so the record structure maps 1:1 to a database table.
"""

import json
import logging
import os
from typing import Dict, List, Optional

from DeepImageSearch.metadatastore.base import BaseMetadataStore, ImageRecord

logger = logging.getLogger(__name__)

RECORDS_FILENAME = "image_records.json"


class JsonMetadataStore(BaseMetadataStore):
    """
    Local JSON file backend for image records.

    Records are held in memory and persisted to `image_records.json`
    on save(). The file format is a JSON array of record objects.
    """

    def __init__(self):
        self._records: Dict[str, ImageRecord] = {}  # keyed by image_id

    def add(self, records: List[ImageRecord]) -> None:
        for record in records:
            self._records[record.image_id] = record

    def get(self, image_id: str) -> Optional[ImageRecord]:
        return self._records.get(image_id)

    def get_by_index(self, image_index: int) -> Optional[ImageRecord]:
        for record in self._records.values():
            if record.image_index == image_index:
                return record
        return None

    def list_all(self) -> List[ImageRecord]:
        return sorted(self._records.values(), key=lambda r: r.image_index)

    def count(self) -> int:
        return len(self._records)

    def delete(self, image_ids: List[str]) -> None:
        for image_id in image_ids:
            self._records.pop(image_id, None)

    def next_index(self) -> int:
        if not self._records:
            return 0
        return max(r.image_index for r in self._records.values()) + 1

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        filepath = os.path.join(path, RECORDS_FILENAME)
        records_list = [r.to_dict() for r in self.list_all()]
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(records_list, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(records_list)} image records to {filepath}")

    def load(self, path: str) -> None:
        filepath = os.path.join(path, RECORDS_FILENAME)
        if not os.path.exists(filepath):
            logger.debug(f"No image records file found at {filepath}, starting fresh")
            return

        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)

        self._records = {}
        for item in data:
            record = ImageRecord.from_dict(item)
            self._records[record.image_id] = record

        logger.info(f"Loaded {len(self._records)} image records from {filepath}")
