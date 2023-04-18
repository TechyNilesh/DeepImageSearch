# Deep Image Search - AI Based Image Search Engine

**DeepImageSearch** is a Python library for fast and accurate image search. It offers seamless integration with Python, GPU support, and advanced capabilities for identifying complex image patterns using the Vision Transformer models.

## Features

- **500+ Pre-trained Models:** With DeepImageSearch, you can import more than 500 pre-trained image and transformer models based on the `timm` library. 
    - **Listing Models:** To list all the available models, you can use the following code snippet:

        ```python
        import timm
        timm.list_models(pretrained=True)
        ```

- **Facebook FAISS Integration:** DeepImageSearch also integrates with Facebook's FAISS library, which allows for efficient similarity search and clustering of dense vectors. This enhances the performance of the search and provides better results.

## Installation

```bash
pip install DeepImageSearch --upgrade
```

## Usage

```python
from DeepImageSearch import Load_Data, Search_Setup
```

### Load Data

Load data from single/multiple folders or a CSV file.

```python
dl = Load_Data()

image_list = dl.from_folder(["folder1", "folder2"])

# or

image_list = dl.from_csv("image_data.csv", "image_paths")
```

### Initialize Search Setup

Initialize the search setup with the list of images, model name, and other configurations.

```python
st = Search_Setup(image_list, model_name="vgg19", pretrained=True, image_count=None)
```

### Index Images

Index images for searching.

```python
st.run_index()
```

### Add New Images to Index

Add new images to the existing index.

```python
new_image_paths = ["new_image1.jpg", "new_image2.jpg"]

st.add_images_to_index(new_image_paths)
```

### Plot Similar Images

Display similar images in a grid.

```python
st.plot_similar_images("query_image.jpg", number_of_images=6)
```

### Get Similar Images

Get a list of similar images.

```python
similar_images = search_setup.get_similar_images("query_image.jpg", number_of_images=10)
```

### Get Image Metadata File

Get the metadata file containing image paths and features.

```python
metadata = st.get_image_metadata_file()
```

## Classes and Methods

### Load_Data

A class for loading data from single/multiple folders or a CSV file.

- `from_folder(folder_list: list) -> list`: Loads image paths from a list of folder paths. The method iterates through all files in each folder and appends the path of the image files with extensions like .png, .jpg, .jpeg, .gif, and .bmp. The method returns a list of image paths.
- `from_csv(csv_file_path: str, images_column_name: str) -> list`: Load images from a CSV file with the specified column name and return a list of image paths.

### Search_Setup

A class to setup the search functionality.

- `Search_Setup(image_list: list, model_name='vgg19', pretrained=True, image_count: int = None)`: Initialize the search setup with the given image list, model name, and pretrained flag. Optionally, limit the number of images to use.
    - `image_list`: A list of images paths.
    - `model_name`: Name of the pre-trained model to be used for feature extraction. Default is 'vgg19'.
    - `pretrained`: Boolean value indicating whether to use the pre-trained weights for the model. Default is True.
    - `image_count`: Number of images to be considered for feature extraction. If not specified, all images will be used.
- `run_index() -> info`: Extracts features from the dataset and indexes them. If the metadata and features are already present, the method prompts the user to confirm whether to re-extract the features or not. The method also loads the metadata and feature vectors from the saved pickle file.
- `add_images_to_index(new_image_paths: list) -> info`: Adds new images to the existing index. The method loads the existing metadata and index and appends the feature vectors of the new images. The method saves the updated index to disk.
- `plot_similar_images(image_path: str, number_of_images: int = 6) -> plot`: Display similar images in a grid for the given image path and number of images.
- `get_similar_images(image_path: str, number_of_images: int = 10) -> dict`: Get a dictionary of similar images for the given image path and number of images.
- `get_image_metadata_file() -> pd.DataFrame`: Get the metadata file containing image paths and features as a pandas DataFrame.