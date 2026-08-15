# SPDX-License-Identifier: MIT
# Copyright (c) 2021 Nilesh Verma
"""Tests for DeepImageSearch.data.loader.Load_Data."""

import csv

import pytest

from DeepImageSearch.data.loader import VALID_IMAGE_EXTENSIONS, Load_Data
from tests.conftest import make_image


class TestFromFolder:
    def test_recursive_finds_nested_images(self, image_dir):
        paths = Load_Data().from_folder([str(image_dir)])
        assert len(paths) == 4
        assert any("yellow.png" in p for p in paths)

    def test_non_recursive_skips_nested(self, image_dir):
        paths = Load_Data().from_folder([str(image_dir)], recursive=False)
        assert len(paths) == 3
        assert not any("yellow.png" in p for p in paths)

    def test_ignores_non_image_extensions(self, image_dir):
        (image_dir / "notes.txt").write_text("not an image")
        paths = Load_Data().from_folder([str(image_dir)])
        assert not any(p.endswith(".txt") for p in paths)

    def test_skips_corrupt_image_when_validating(self, image_dir):
        (image_dir / "broken.png").write_bytes(b"definitely not a png")
        paths = Load_Data().from_folder([str(image_dir)], validate=True)
        assert not any("broken.png" in p for p in paths)

    def test_keeps_corrupt_image_when_not_validating(self, image_dir):
        (image_dir / "broken.png").write_bytes(b"definitely not a png")
        paths = Load_Data().from_folder([str(image_dir)], validate=False)
        assert any("broken.png" in p for p in paths)

    def test_missing_folder_is_skipped_not_fatal(self, image_dir):
        paths = Load_Data().from_folder([str(image_dir), "/no/such/folder"])
        assert len(paths) == 4

    def test_file_passed_as_folder_is_skipped(self, tmp_path, image_dir):
        a_file = make_image(tmp_path / "loose.png")
        paths = Load_Data().from_folder([a_file])
        assert paths == []

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            Load_Data().from_folder([])

    def test_non_list_raises(self, image_dir):
        with pytest.raises(TypeError, match="must be a list"):
            Load_Data().from_folder(str(image_dir))

    def test_all_documented_extensions_are_lowercase(self):
        assert all(ext == ext.lower() and ext.startswith(".") for ext in VALID_IMAGE_EXTENSIONS)

    def test_uppercase_extension_is_matched(self, tmp_path):
        folder = tmp_path / "upper"
        folder.mkdir()
        make_image(folder / "SHOUT.PNG")
        assert len(Load_Data().from_folder([str(folder)])) == 1


class TestFromCsv:
    def _write_csv(self, path, rows, column="image"):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[column, "label"])
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        return str(path)

    def test_reads_existing_paths(self, tmp_path, image_paths):
        csv_path = self._write_csv(
            tmp_path / "data.csv",
            [{"image": p, "label": "x"} for p in image_paths],
        )
        assert Load_Data().from_csv(csv_path, "image") == image_paths

    def test_skips_missing_and_blank_paths(self, tmp_path, image_paths):
        rows = [{"image": image_paths[0], "label": "x"},
                {"image": "/no/such/image.png", "label": "y"},
                {"image": "   ", "label": "z"}]
        csv_path = self._write_csv(tmp_path / "data.csv", rows)
        assert Load_Data().from_csv(csv_path, "image") == [image_paths[0]]

    def test_strips_surrounding_whitespace(self, tmp_path, image_paths):
        csv_path = self._write_csv(tmp_path / "data.csv", [{"image": f"  {image_paths[0]}  ", "label": "x"}])
        assert Load_Data().from_csv(csv_path, "image") == [image_paths[0]]

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            Load_Data().from_csv("/no/such/file.csv", "image")

    def test_unknown_column_raises(self, tmp_path, image_paths):
        csv_path = self._write_csv(tmp_path / "data.csv", [{"image": image_paths[0], "label": "x"}])
        with pytest.raises(ValueError, match="not found"):
            Load_Data().from_csv(csv_path, "picture")


class TestFromList:
    def test_validates_and_returns_paths(self, image_paths):
        assert Load_Data().from_list(image_paths) == image_paths

    def test_skips_missing_files(self, image_paths):
        result = Load_Data().from_list(image_paths + ["/no/such/image.png"])
        assert result == image_paths

    def test_skips_corrupt_when_validating(self, tmp_path, image_paths):
        broken = tmp_path / "broken.png"
        broken.write_bytes(b"not a png")
        assert str(broken) not in Load_Data().from_list(image_paths + [str(broken)])

    def test_keeps_corrupt_when_not_validating(self, tmp_path, image_paths):
        broken = tmp_path / "broken.png"
        broken.write_bytes(b"not a png")
        result = Load_Data().from_list(image_paths + [str(broken)], validate=False)
        assert str(broken) in result

    def test_empty_list_returns_empty(self):
        assert Load_Data().from_list([]) == []
