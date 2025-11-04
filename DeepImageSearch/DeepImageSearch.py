import DeepImageSearch.config as config
import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import numpy as np
from torchvision import transforms
import torch
from torch.autograd import Variable
import timm
from PIL import ImageOps
import math
import faiss
import logging
from typing import List, Dict, Optional, Union
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Load_Data:
    """A class for loading data from single/multiple folders or a CSV file"""

    def __init__(self):
        """
        Initializes an instance of LoadData class
        """
        pass

    def from_folder(self, folder_list: List[str]) -> List[str]:
        """
        Adds images from the specified folders to the image_list.

        Parameters:
        -----------
        folder_list : list
            A list of paths to the folders containing images to be added to the image_list.

        Returns:
        --------
        list
            List of valid image paths found in the folders.
        """
        if not folder_list:
            raise ValueError("folder_list cannot be empty")

        if not isinstance(folder_list, list):
            raise TypeError("folder_list must be a list")

        self.folder_list = folder_list
        image_path = []
        valid_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')

        for folder in self.folder_list:
            if not os.path.exists(folder):
                logger.warning(f"Folder does not exist: {folder}")
                continue

            if not os.path.isdir(folder):
                logger.warning(f"Path is not a directory: {folder}")
                continue

            for root, dirs, files in os.walk(folder):
                for file in files:
                    if file.lower().endswith(valid_extensions):
                        full_path = os.path.join(root, file)
                        # Validate that the file can be opened as an image
                        try:
                            with Image.open(full_path) as img:
                                img.verify()
                            image_path.append(full_path)
                        except Exception as e:
                            logger.warning(f"Skipping invalid image file {full_path}: {e}")
                            continue

        if not image_path:
            logger.warning("No valid images found in the specified folders")
        else:
            logger.info(f"Found {len(image_path)} valid images")

        return image_path

    def from_csv(self, csv_file_path: str, images_column_name: str) -> List[str]:
        """
        Adds images from the specified column of a CSV file to the image_list.

        Parameters:
        -----------
        csv_file_path : str
            The path to the CSV file.
        images_column_name : str
            The name of the column containing the paths to the images to be added to the image_list.

        Returns:
        --------
        list
            List of image paths from the CSV file.
        """
        if not os.path.exists(csv_file_path):
            raise FileNotFoundError(f"CSV file not found: {csv_file_path}")

        self.csv_file_path = csv_file_path
        self.images_column_name = images_column_name

        try:
            df = pd.read_csv(self.csv_file_path)
        except Exception as e:
            raise ValueError(f"Error reading CSV file: {e}")

        if images_column_name not in df.columns:
            raise ValueError(f"Column '{images_column_name}' not found in CSV. Available columns: {df.columns.tolist()}")

        image_paths = df[self.images_column_name].dropna().tolist()

        # Validate image paths
        valid_paths = []
        for path in image_paths:
            if os.path.exists(path) and os.path.isfile(path):
                valid_paths.append(path)
            else:
                logger.warning(f"Image path does not exist: {path}")

        logger.info(f"Loaded {len(valid_paths)} valid image paths from CSV")
        return valid_paths

class Search_Setup:
    """ A class for setting up and running image similarity search."""

    def __init__(
        self,
        image_list: List[str],
        model_name: str = 'vgg19',
        pretrained: bool = True,
        image_count: Optional[int] = None,
        image_size: int = 224,
        metadata_dir: str = 'metadata-files',
        use_gpu: bool = False,
        index_type: str = 'flat'
    ):
        """
        Parameters:
        -----------
        image_list : list
            A list of images to be indexed and searched.
        model_name : str, optional (default='vgg19')
            The name of the pre-trained model to use for feature extraction.
        pretrained : bool, optional (default=True)
            Whether to use the pre-trained weights for the chosen model.
        image_count : int, optional (default=None)
            The number of images to be indexed and searched. If None, all images in the image_list will be used.
        image_size : int, optional (default=224)
            The size to which images will be resized for feature extraction.
        metadata_dir : str, optional (default='metadata-files')
            Directory to store metadata and index files.
        use_gpu : bool, optional (default=False)
            Whether to use GPU for feature extraction.
        index_type : str, optional (default='flat')
            Type of FAISS index to use. Options: 'flat', 'ivf', 'hnsw'
        """
        # Validate inputs
        if not image_list:
            raise ValueError("image_list cannot be empty")

        if not isinstance(image_list, list):
            raise TypeError("image_list must be a list")

        if image_count is not None and (not isinstance(image_count, int) or image_count <= 0):
            raise ValueError("image_count must be a positive integer or None")

        if not isinstance(image_size, int) or image_size <= 0:
            raise ValueError("image_size must be a positive integer")

        if index_type not in ['flat', 'ivf', 'hnsw']:
            raise ValueError("index_type must be one of: 'flat', 'ivf', 'hnsw'")

        self.model_name = model_name
        self.pretrained = pretrained
        self.image_data = pd.DataFrame()
        self.d = None
        self.image_size = image_size
        self.metadata_dir = metadata_dir
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.index_type = index_type

        if image_count is None:
            self.image_list = image_list
        else:
            self.image_list = image_list[:image_count]

        logger.info(f"Initialized with {len(self.image_list)} images")

        # Create metadata directory
        model_metadata_dir = os.path.join(self.metadata_dir, self.model_name)
        if not os.path.exists(model_metadata_dir):
            try:
                os.makedirs(model_metadata_dir)
                logger.info(f"Created metadata directory: {model_metadata_dir}")
            except Exception as e:
                raise RuntimeError(f"Error creating metadata directory {model_metadata_dir}: {e}")

        # Load the pre-trained model and remove the last layer
        logger.info(f"Loading model: {model_name}")
        try:
            base_model = timm.create_model(self.model_name, pretrained=self.pretrained)
            self.model = torch.nn.Sequential(*list(base_model.children())[:-1])
            self.model.eval()

            if self.use_gpu:
                self.model = self.model.cuda()
                logger.info(f"Model loaded successfully on GPU: {model_name}")
            else:
                logger.info(f"Model loaded successfully on CPU: {model_name}")
        except Exception as e:
            raise RuntimeError(f"Error loading model {model_name}: {e}")

    def _extract(self, img: Image.Image) -> np.ndarray:
        """
        Extract features from a single image.

        Parameters:
        -----------
        img : PIL.Image
            Input image

        Returns:
        --------
        np.ndarray
            Normalized feature vector
        """
        # Resize and convert the image
        img = img.resize((self.image_size, self.image_size))
        img = img.convert('RGB')

        # Preprocess the image
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        x = preprocess(img)
        x = Variable(torch.unsqueeze(x, dim=0).float(), requires_grad=False)

        if self.use_gpu:
            x = x.cuda()

        # Extract features
        with torch.no_grad():
            feature = self.model(x)

        if self.use_gpu:
            feature = feature.cpu()

        feature = feature.data.numpy().flatten()
        return feature / np.linalg.norm(feature)

    def _get_feature(self, image_data: List[str]) -> List[Optional[np.ndarray]]:
        """
        Extract features from a list of images.

        Parameters:
        -----------
        image_data : list
            List of image paths

        Returns:
        --------
        list
            List of feature vectors (or None for failed extractions)
        """
        self.image_data = image_data
        features = []
        failed_images = []

        for img_path in tqdm(self.image_data, desc="Extracting features"):
            try:
                with Image.open(img_path) as img:
                    feature = self._extract(img=img)
                    features.append(feature)
            except FileNotFoundError:
                logger.error(f"File not found: {img_path}")
                features.append(None)
                failed_images.append((img_path, "File not found"))
            except IOError as e:
                logger.error(f"Error opening image {img_path}: {e}")
                features.append(None)
                failed_images.append((img_path, f"IO Error: {e}"))
            except Exception as e:
                logger.error(f"Unexpected error processing {img_path}: {e}")
                features.append(None)
                failed_images.append((img_path, f"Error: {e}"))

        if failed_images:
            logger.warning(f"Failed to process {len(failed_images)} images")

        return features

    def _start_feature_extraction(self) -> pd.DataFrame:
        """
        Extract features from all images and save to disk.

        Returns:
        --------
        pd.DataFrame
            DataFrame containing image paths and features
        """
        image_data = pd.DataFrame()
        image_data['images_paths'] = self.image_list
        f_data = self._get_feature(self.image_list)
        image_data['features'] = f_data

        # Remove rows with None features
        original_count = len(image_data)
        image_data = image_data.dropna().reset_index(drop=True)
        removed_count = original_count - len(image_data)

        if removed_count > 0:
            logger.warning(f"Removed {removed_count} images due to feature extraction failures")

        if len(image_data) == 0:
            raise RuntimeError("No valid features extracted from any images")

        # Save metadata
        pkl_path = config.image_data_with_features_pkl(self.model_name, self.metadata_dir)
        image_data.to_pickle(pkl_path)
        logger.info(f"Image metadata saved: {pkl_path}")

        return image_data

    def _create_faiss_index(self, dimension: int) -> faiss.Index:
        """
        Create a FAISS index based on the specified index_type.

        Parameters:
        -----------
        dimension : int
            Dimension of feature vectors

        Returns:
        --------
        faiss.Index
            FAISS index object
        """
        if self.index_type == 'flat':
            index = faiss.IndexFlatL2(dimension)
            logger.info("Using IndexFlatL2 (exact search)")
        elif self.index_type == 'ivf':
            # For IVF, use sqrt(n) clusters as a rule of thumb
            n_list = min(100, int(np.sqrt(len(self.image_list))))
            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, n_list)
            logger.info(f"Using IndexIVFFlat with {n_list} clusters (approximate search)")
        elif self.index_type == 'hnsw':
            index = faiss.IndexHNSWFlat(dimension, 32)
            logger.info("Using IndexHNSWFlat (approximate search)")
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

        return index

    def _start_indexing(self, image_data: pd.DataFrame, batch_size: int = 1000):
        """
        Create and save FAISS index from extracted features.

        Parameters:
        -----------
        image_data : pd.DataFrame
            DataFrame containing image features
        batch_size : int, optional (default=1000)
            Number of features to process at once to avoid memory issues
        """
        self.image_data = image_data
        d = len(image_data['features'][0])
        self.d = d

        index = self._create_faiss_index(d)

        # For IVF index, need to train first
        if self.index_type == 'ivf':
            logger.info("Training IVF index...")
            features_matrix = np.vstack(image_data['features'].values).astype(np.float32)
            index.train(features_matrix)

        # Add features in batches to avoid memory issues
        total_features = len(image_data)
        for start_idx in tqdm(range(0, total_features, batch_size), desc="Indexing features"):
            end_idx = min(start_idx + batch_size, total_features)
            batch_features = image_data['features'].iloc[start_idx:end_idx]
            features_matrix = np.vstack(batch_features.values).astype(np.float32)
            index.add(features_matrix)

        # Save index
        idx_path = config.image_features_vectors_idx(self.model_name, self.metadata_dir)
        faiss.write_index(index, idx_path)
        logger.info(f"Index saved: {idx_path}")

    def run_index(self, force_reindex: bool = False):
        """
        Indexes the images in the image_list and creates an index file for fast similarity search.

        Parameters:
        -----------
        force_reindex : bool, optional (default=False)
            If True, force re-extraction of features even if metadata exists.
            If False, will skip if metadata already exists.
        """
        model_metadata_dir = os.path.join(self.metadata_dir, self.model_name)

        # Check if metadata already exists
        metadata_exists = (
            os.path.exists(config.image_data_with_features_pkl(self.model_name, self.metadata_dir)) and
            os.path.exists(config.image_features_vectors_idx(self.model_name, self.metadata_dir))
        )

        if metadata_exists and not force_reindex:
            logger.info("Metadata and index files already exist. Use force_reindex=True to re-extract.")
            logger.info(f"Loading existing metadata from: {model_metadata_dir}")
        else:
            if force_reindex:
                logger.info("Force reindex enabled. Re-extracting features...")
            data = self._start_feature_extraction()
            self._start_indexing(data)

        # Load metadata
        self.image_data = pd.read_pickle(config.image_data_with_features_pkl(self.model_name, self.metadata_dir))
        self.f = len(self.image_data['features'][0])
        logger.info(f"Loaded {len(self.image_data)} indexed images")

    def add_images_to_index(self, new_image_paths: List[str], batch_size: int = 100):
        """
        Adds new images to the existing index.

        Parameters:
        -----------
        new_image_paths : list
            A list of paths to the new images to be added to the index.
        batch_size : int, optional (default=100)
            Number of images to process in each batch for efficiency.
        """
        if not new_image_paths:
            logger.warning("No new images provided")
            return

        # Validate new image paths
        valid_paths = []
        for path in new_image_paths:
            if os.path.exists(path) and os.path.isfile(path):
                valid_paths.append(path)
            else:
                logger.warning(f"Skipping invalid path: {path}")

        if not valid_paths:
            logger.warning("No valid images to add")
            return

        logger.info(f"Adding {len(valid_paths)} new images to index")

        # Load existing metadata and index
        self.image_data = pd.read_pickle(config.image_data_with_features_pkl(self.model_name, self.metadata_dir))
        index = faiss.read_index(config.image_features_vectors_idx(self.model_name, self.metadata_dir))

        # Process new images in batches
        new_metadata_list = []
        new_features_list = []

        for i, new_image_path in enumerate(tqdm(valid_paths, desc="Processing new images")):
            try:
                with Image.open(new_image_path) as img:
                    feature = self._extract(img)
                new_metadata_list.append({"images_paths": new_image_path, "features": feature})
                new_features_list.append(feature)
            except Exception as e:
                logger.error(f"Error processing {new_image_path}: {e}")
                continue

            # Add to index in batches
            if len(new_features_list) >= batch_size or i == len(valid_paths) - 1:
                if new_features_list:
                    features_array = np.array(new_features_list, dtype=np.float32)
                    index.add(features_array)
                    new_features_list = []

        # Update metadata
        if new_metadata_list:
            new_metadata_df = pd.DataFrame(new_metadata_list)
            self.image_data = pd.concat([self.image_data, new_metadata_df], axis=0, ignore_index=True)

            # Save the updated metadata and index
            self.image_data.to_pickle(config.image_data_with_features_pkl(self.model_name, self.metadata_dir))
            faiss.write_index(index, config.image_features_vectors_idx(self.model_name, self.metadata_dir))

            logger.info(f"Successfully added {len(new_metadata_list)} new images to the index")
        else:
            logger.warning("No new images were successfully processed")

    def _search_by_vector(self, v: np.ndarray, n: int) -> Dict[int, str]:
        """
        Search for similar images using a feature vector.

        Parameters:
        -----------
        v : np.ndarray
            Feature vector to search for
        n : int
            Number of similar images to return

        Returns:
        --------
        dict
            Dictionary mapping indices to image paths
        """
        self.v = v
        self.n = n

        index = faiss.read_index(config.image_features_vectors_idx(self.model_name, self.metadata_dir))

        # For IVF index, set number of probes for search
        if self.index_type == 'ivf':
            index.nprobe = 10

        D, I = index.search(np.array([self.v], dtype=np.float32), self.n)
        return dict(zip(I[0], self.image_data.iloc[I[0]]['images_paths'].to_list()))

    def _get_query_vector(self, image_path: str) -> np.ndarray:
        """
        Extract feature vector from a query image.

        Parameters:
        -----------
        image_path : str
            Path to the query image

        Returns:
        --------
        np.ndarray
            Feature vector
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Query image not found: {image_path}")

        self.image_path = image_path

        try:
            with Image.open(self.image_path) as img:
                query_vector = self._extract(img)
        except Exception as e:
            raise RuntimeError(f"Error extracting features from query image: {e}")

        return query_vector

    def plot_similar_images(self, image_path: str, number_of_images: int = 6):
        """
        Plots a given image and its most similar images according to the indexed image features.

        Parameters:
        -----------
        image_path : str
            The path to the query image to be plotted.
        number_of_images : int, optional (default=6)
            The number of most similar images to the query image to be plotted.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Query image not found: {image_path}")

        if number_of_images <= 0:
            raise ValueError("number_of_images must be positive")

        input_img = Image.open(image_path)
        input_img_resized = ImageOps.fit(input_img, (224, 224), Image.LANCZOS)
        plt.figure(figsize=(5, 5))
        plt.axis('off')
        plt.title('Input Image', fontsize=18)
        plt.imshow(input_img_resized)
        plt.show()

        query_vector = self._get_query_vector(image_path)
        img_list = list(self._search_by_vector(query_vector, number_of_images).values())

        grid_size = math.ceil(math.sqrt(number_of_images))
        axes = []
        fig = plt.figure(figsize=(20, 15))
        for a in range(number_of_images):
            if a >= len(img_list):
                break
            axes.append(fig.add_subplot(grid_size, grid_size, a + 1))
            plt.axis('off')
            try:
                img = Image.open(img_list[a])
                img_resized = ImageOps.fit(img, (224, 224), Image.LANCZOS)
                plt.imshow(img_resized)
            except Exception as e:
                logger.error(f"Error displaying image {img_list[a]}: {e}")

        fig.tight_layout()
        fig.subplots_adjust(top=0.93)
        fig.suptitle('Similar Result Found', fontsize=22)
        plt.show(fig)

    def get_similar_images(self, image_path: str, number_of_images: int = 10) -> Dict[int, str]:
        """
        Returns the most similar images to a given query image according to the indexed image features.

        Parameters:
        -----------
        image_path : str
            The path to the query image.
        number_of_images : int, optional (default=10)
            The number of most similar images to the query image to be returned.

        Returns:
        --------
        dict
            Dictionary mapping indices to similar image paths
        """
        if number_of_images <= 0:
            raise ValueError("number_of_images must be positive")

        self.image_path = image_path
        self.number_of_images = number_of_images
        query_vector = self._get_query_vector(self.image_path)
        img_dict = self._search_by_vector(query_vector, self.number_of_images)
        return img_dict

    def get_image_metadata_file(self) -> pd.DataFrame:
        """
        Returns the metadata file containing information about the indexed images.

        Returns:
        --------
        DataFrame
            The Pandas DataFrame of the metadata file.
        """
        self.image_data = pd.read_pickle(config.image_data_with_features_pkl(self.model_name, self.metadata_dir))
        return self.image_data
