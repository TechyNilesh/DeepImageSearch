# Deep Image Search - AI-Based Image Search Engine

Deep Image Search is an AI-based image search engine that incorporates ViT (Vision Transformer) for feature extraction and utilizes a tree-based vectorized search technique.

## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Import required modules

```python
from DeepImageSearch import Index,LoadData,SearchImage
from PIL import Image
```

### 2. Load Data

Load data from single/multiple folders or a CSV file using the `LoadData` class.

#### Example:

```python
data_loader = LoadData()

# Load data from a single folder
image_paths = data_loader.from_folder(["path/to/folder"])

# Load data from multiple folders
image_paths = data_loader.from_folder(["path/to/folder1", "path/to/folder2"])

# Load data from a CSV file
image_paths = data_loader.from_csv("path/to/csv_file.csv", "images_column_name")
```

### 3. Feature Extraction

Extract features using a transformer-based feature extraction model with the `FeatureExtractor` class.

#### Example:

```python
feature_extractor = FeatureExtractor(model_name='vit_base_patch16_224', pretrained=True)

# Extract features from a single image
img = Image.open("path/to/image.jpg")
features = feature_extractor.extract(img)

# Extract features from a list of image paths
image_features = feature_extractor.get_feature(image_paths)
```

### 4. Indexing

Index the extracted image features using the `Index` class.

#### Example:

```python
indexer = Index(image_list=image_paths)
indexer.start()
```

### 5. Search for similar images

Search for similar images using the `SearchImage` class.

#### Example:

```python
searcher = SearchImage()

# Display a plot of the input image and its most similar images
searcher.plot_similar_images("path/to/query_image.jpg", number_of_images=6)

# Retrieve a dictionary of the most similar images to the input image
similar_images = searcher.get_similar_images("path/to/query_image.jpg", number_of_images=10)
```

## Classes and Methods

### LoadData

A class for loading data from single/multiple folders or a CSV file.

#### Methods

- `from_folder(self, folder_list: list)`: Method to load data from a single folder path or a list of folder paths
- `from_csv(self, csv_file_path: str, images_column_name: str)`: Method to load data from a CSV file with a column containing image paths

### FeatureExtractor

A class for extracting features using a transformer-based feature extraction model.

#### Methods

- `extract(self, img)`: Method to extract features from an image using the pre-trained model
- `get_feature(self, image_data: list)`: Method to extract features from a list of image paths

### Index

A class for indexing the extracted image features.

#### Methods

- `start_feature_extraction(self)`: Extracts features from images and saves the data as a pickle file
- `start_indexing(self, image_data)`: Indexes the extracted image features using Annoy
- `start(self)`: Starts the feature extraction and indexing process

### SearchImage

A class for searching for similar images using the extracted features.

#### Methods

- `search_by_vector(self, v, n: int)`: Search for similar images using a given feature vector
- `get_query_vector(self, image_path: str)`: Extract the feature vector for a given image
- `plot_similar_images(self, image_path: str, number_of_images: int = 6)`: Display a plot of the input image and its most similar images
- `get_similar_images(self, image_path: str, number_of_images: int = 10)`: Retrieve a dictionary of the most similar images to the input image