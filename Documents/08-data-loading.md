# Data Loading

The `Load_Data` class provides methods for collecting image file paths from various sources.

## Quick Usage

With `SearchEngine`, you can skip `Load_Data` entirely and pass a folder path directly:

```python
engine.index("./photos")
```

Use `Load_Data` when you need more control over which images are loaded.

## From Folders

```python
from DeepImageSearch import Load_Data

loader = Load_Data()

# Single folder
images = loader.from_folder(["./photos"])

# Multiple folders
images = loader.from_folder(["./photos", "./screenshots", "./downloads"])

# Non-recursive (top-level only)
images = loader.from_folder(["./photos"], recursive=False)

# Skip validation (faster, but may include corrupt files)
images = loader.from_folder(["./photos"], validate=False)
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `folder_list` | `List[str]` | required | Directories to scan. |
| `recursive` | `bool` | `True` | Walk subdirectories. |
| `validate` | `bool` | `True` | Verify each file is a valid image. |

**Returns:** `List[str]` -- valid image file paths.

### Supported Extensions

`.png`, `.jpg`, `.jpeg`, `.gif`, `.bmp`, `.tiff`, `.webp`, `.avif`

## From CSV

```python
images = loader.from_csv("dataset.csv", "image_path")
```

The CSV should have a column containing image file paths:

```csv
image_path,label
/data/photos/001.jpg,cat
/data/photos/002.jpg,dog
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `csv_file_path` | `str` | required | Path to the CSV file. |
| `images_column_name` | `str` | required | Column containing image paths. |

**Returns:** `List[str]` -- valid image paths from the CSV.

Paths that don't exist on disk are skipped with a warning.

## From a List

Validate an existing list of paths:

```python
paths = ["/data/img1.jpg", "/data/img2.jpg", "/data/missing.jpg"]
valid = loader.from_list(paths)
# ["/data/img1.jpg", "/data/img2.jpg"]  (missing.jpg skipped)
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `image_paths` | `List[str]` | required | List of image file paths. |
| `validate` | `bool` | `True` | Verify each file is a valid image. |

**Returns:** `List[str]` -- valid image paths.
