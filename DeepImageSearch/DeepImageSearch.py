import DeepImageSearch.config as config
import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import numpy as np
from annoy import AnnoyIndex
from torchvision import transforms
import torch
from torchvision.models import vgg16
from torch.autograd import Variable
import timm
from PIL import ImageOps
import math

class LoadData:
    """A class for loading data from single/multiple folders or a CSV file"""

    def __init__(self):
        """
        Initializes an instance of LoadData class
        """
        pass
    
    def from_folder(self, folder_list: list):
        """
        Method to load data from a single folder path or a list of folder paths

        Args:
        - folder_list (list): A list of folder paths

        Returns:
        - image_path (list): A list of image paths
        """
        self.folder_list = folder_list
        image_path = []
        for folder in self.folder_list:
            for path in os.listdir(folder):
                image_path.append(os.path.join(folder, path))
        return image_path

    def from_csv(self, csv_file_path: str, images_column_name: str):
        """
        Method to load data from a CSV file with a column containing image paths

        Args:
        - csv_file_path (str): The path of the CSV file
        - images_column_name (str): The name of the column containing the image paths

        Returns:
        - image_path (list): A list of image paths
        """
        self.csv_file_path = csv_file_path
        self.images_column_name = images_column_name
        return pd.read_csv(self.csv_file_path)[self.images_column_name].to_list()

class FeatureExtractor:
    """A class for extracting features using a transformer-based feature extraction model"""

    def __init__(self, model_name='vit_base_patch16_224', pretrained=True):
        """
        Initializes an instance of FeatureExtractor class

        Args:
        - model_name (str): The name of the pre-trained model to use for feature extraction
        - pretrained (bool): Whether to use the pre-trained weights or not
        """
        self.model_name = model_name
        self.pretrained = pretrained
        
        # Load the pre-trained model and remove the last layer
        base_model = timm.create_model(self.model_name, pretrained=self.pretrained)
        self.model = torch.nn.Sequential(*list(base_model.children())[:-1])
        self.model.eval()

    def extract(self, img):
        """
        Method to extract features from an image using the pre-trained model

        Args:
        - img (PIL Image): The input image

        Returns:
        - feature (numpy array): The extracted features
        """
        # Resize and convert the image
        img = img.resize((224, 224))
        img = img.convert('RGB')
        
        # Preprocess the image
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        x = preprocess(img)
        x = Variable(torch.unsqueeze(x, dim=0).float(), requires_grad=False)
        
        # Extract features
        feature = self.model(x)
        feature = feature.data.numpy().flatten()
        return feature / np.linalg.norm(feature)

    def get_feature(self, image_data: list):
        """
        Method to extract features from a list of image paths

        Args:
        - image_data (list): A list of image paths

        Returns:
        - features (list): A list of extracted features
        """
        self.image_data = image_data
        features = []
        for img_path in tqdm(self.image_data):  # Iterate through images
            # Extract features from the image
            try:
                feature = self.extract(img=Image.open(img_path))
                features.append(feature)
            except:
                # If there is an error, append None to the feature list
                features.append(None)
                continue
        return features
    
class Index:
    def __init__(self, image_list: list):
        """
        Initializes the Index class with a list of image paths.

        Parameters
        ----------
        image_list : list
            A list of image paths.
        """
        self.image_list = image_list
        if 'meta-data-files' not in os.listdir():
            os.makedirs("meta-data-files")
        self.FE = FeatureExtractor()

    def start_feature_extraction(self):
        """
        Extracts features from images and saves the data as a pickle file.

        Returns
        -------
        image_data : DataFrame
            A DataFrame containing image paths and their extracted features.
        """
        image_data = pd.DataFrame()
        image_data['images_paths'] = self.image_list
        f_data = self.FE.get_feature(self.image_list)
        image_data['features']  = f_data
        image_data = image_data.dropna().reset_index(drop=True)
        image_data.to_pickle(config.image_data_with_features_pkl)
        print("Image Meta Information Saved: [meta-data-files/image_data_features.pkl]")
        return image_data

    def start_indexing(self, image_data):
        """
        Indexes the extracted image features using Annoy.

        Parameters
        ----------
        image_data : DataFrame
            A DataFrame containing image paths and their extracted features.
        """
        self.image_data = image_data
        f = len(image_data['features'][0]) # Length of item vector that will be indexed
        t = AnnoyIndex(f, 'euclidean')
        for i, v in tqdm(zip(self.image_data.index, image_data['features'])):
            t.add_item(i, v)
        t.build(100) # 100 trees
        print("Saved the Indexed File:"+"[meta-data-files/image_features_vectors.ann]")
        t.save(config.image_features_vectors_ann)

    def start(self):
        """
        Starts the feature extraction and indexing process.
        """
        if len(os.listdir("meta-data-files/")) == 0:
            data = self.start_feature_extraction()
            self.start_indexing(data)
        else:
            print("Metadata and Features are already present, Do you want Extract Again? Enter yes or no")
            flag  = str(input())
            if flag.lower() == 'yes':
                data = self.start_feature_extraction()
                self.start_indexing(data)
            else:
                print("Meta data already Present, Please Apply Search!")
                print(os.listdir("meta-data-files/"))

class SearchImage:
    def __init__(self):
        """
        Initialize the `SearchImage` class.

        Attributes:
            image_data (pandas.DataFrame): A DataFrame containing the image paths and features.
            f (int): The length of the feature vectors.
        """
        self.image_data = pd.read_pickle(config.image_data_with_features_pkl)
        self.f = len(self.image_data['features'][0])

    def search_by_vector(self, v, n: int):
        """
        Search for similar images using a given feature vector.

        Args:
            v (numpy.ndarray): The feature vector to search by.
            n (int): The number of similar images to retrieve.

        Returns:
            dict: A dictionary mapping the indices of the most similar images to their corresponding image paths.
        """
        self.v = v
        self.n = n
        u = AnnoyIndex(self.f, 'euclidean')
        u.load(config.image_features_vectors_ann)
        index_list = u.get_nns_by_vector(self.v, self.n)
        return dict(zip(index_list, self.image_data.iloc[index_list]['images_paths'].to_list()))

    def get_query_vector(self, image_path: str):
        """
        Extract the feature vector for a given image.

        Args:
            image_path (str): The path to the image file.

        Returns:
            numpy.ndarray: The feature vector for the image.
        """
        self.image_path = image_path
        img = Image.open(self.image_path)
        fe = FeatureExtractor()
        query_vector = fe.extract(img)
        return query_vector

    def plot_similar_images(self, image_path: str, number_of_images: int = 6):
        """
        Display a plot of the input image and its most similar images.

        Args:
            image_path (str): The path to the input image file.
            number_of_images (int): The number of similar images to display.
        """
        input_img = Image.open(image_path)
        input_img_resized = ImageOps.fit(input_img, (224, 224), Image.LANCZOS)
        plt.figure(figsize=(5, 5))
        plt.axis('off')
        plt.title('Input Image', fontsize=18)
        plt.imshow(input_img_resized)
        plt.show()

        query_vector = self.get_query_vector(image_path)
        img_list = list(self.search_by_vector(query_vector, number_of_images).values())

        grid_size = math.ceil(math.sqrt(number_of_images))
        axes = []
        fig = plt.figure(figsize=(20, 15))
        for a in range(number_of_images):
            axes.append(fig.add_subplot(grid_size, grid_size, a + 1))
            plt.axis('off')
            img = Image.open(img_list[a])
            img_resized = ImageOps.fit(img, (224, 224), Image.LANCZOS)
            plt.imshow(img_resized)
        fig.tight_layout()
        fig.subplots_adjust(top=0.93)
        fig.suptitle('Similar Result Found', fontsize=22)
        plt.show(fig)

    def get_similar_images(self,image_path:str,number_of_images:int=10):
        """
        Retrieve a dictionary of the most similar images to the input image.

        Args:
            image_path (str): The path to the input image file.
            number_of_images (int): The number of similar images to retrieve.

        Returns:
            dict: A dictionary mapping the indices of the most similar images to their corresponding image paths.
        """
        self.image_path = image_path
        self.number_of_images = number_of_images
        query_vector = self.get_query_vector(self.image_path)
        img_dict = self.search_by_vector(query_vector,self.number_of_images)
        return img_dict