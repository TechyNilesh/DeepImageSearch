import csv
import os
import logging
from typing import List
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)

VALID_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp", ".avif"})


class Load_Data:
    """Load image paths from folders, CSV files, or lists."""

    def from_folder(self, folder_list: List[str], recursive: bool = True, validate: bool = True) -> List[str]:
        """
        Collect image paths from one or more folders.

        Parameters
        ----------
        folder_list : list[str]
            Directories to scan for images.
        recursive : bool
            Walk subdirectories (default True).
        validate : bool
            Verify each file is a valid image (default True).

        Returns
        -------
        list[str]
            Valid image file paths.
        """
        if not folder_list:
            raise ValueError("folder_list cannot be empty")
        if not isinstance(folder_list, list):
            raise TypeError("folder_list must be a list")

        image_paths: List[str] = []

        for folder in folder_list:
            folder_path = Path(folder)
            if not folder_path.exists():
                logger.warning(f"Folder does not exist: {folder}")
                continue
            if not folder_path.is_dir():
                logger.warning(f"Path is not a directory: {folder}")
                continue

            pattern = "**/*" if recursive else "*"
            for file_path in folder_path.glob(pattern):
                if file_path.suffix.lower() not in VALID_IMAGE_EXTENSIONS:
                    continue
                if not file_path.is_file():
                    continue

                if validate:
                    try:
                        with Image.open(file_path) as img:
                            img.verify()
                    except Exception as e:
                        logger.warning(f"Skipping invalid image {file_path}: {e}")
                        continue

                image_paths.append(str(file_path))

        logger.info(f"Found {len(image_paths)} valid images")
        return image_paths

    def from_csv(self, csv_file_path: str, images_column_name: str) -> List[str]:
        """
        Load image paths from a CSV column.

        Parameters
        ----------
        csv_file_path : str
            Path to the CSV file.
        images_column_name : str
            Column containing image paths.

        Returns
        -------
        list[str]
            Valid image paths from the CSV.
        """
        if not os.path.exists(csv_file_path):
            raise FileNotFoundError(f"CSV file not found: {csv_file_path}")

        with open(csv_file_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if images_column_name not in reader.fieldnames:
                raise ValueError(f"Column '{images_column_name}' not found. Available: {reader.fieldnames}")

            valid_paths = []
            for row in reader:
                path = row[images_column_name]
                if not path or not path.strip():
                    continue
                path = path.strip()
                if os.path.exists(path) and os.path.isfile(path):
                    valid_paths.append(path)
                else:
                    logger.warning(f"Image path does not exist: {path}")

        logger.info(f"Loaded {len(valid_paths)} valid image paths from CSV")
        return valid_paths

    def from_list(self, image_paths: List[str], validate: bool = True) -> List[str]:
        """
        Validate and return a list of image paths.

        Parameters
        ----------
        image_paths : list[str]
            List of image file paths.
        validate : bool
            Verify each file is a valid image (default True).

        Returns
        -------
        list[str]
            Valid image paths.
        """
        valid = []
        for path in image_paths:
            if not os.path.isfile(path):
                logger.warning(f"File not found: {path}")
                continue
            if validate:
                try:
                    with Image.open(path) as img:
                        img.verify()
                    valid.append(path)
                except Exception as e:
                    logger.warning(f"Invalid image {path}: {e}")
            else:
                valid.append(path)
        logger.info(f"Validated {len(valid)} image paths")
        return valid
