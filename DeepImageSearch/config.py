import os

def image_data_with_features_pkl(model_name: str, metadata_dir: str = 'metadata-files') -> str:
    """
    Get the path to the image data features pickle file.

    Parameters:
    -----------
    model_name : str
        Name of the model
    metadata_dir : str, optional (default='metadata-files')
        Base directory for metadata files

    Returns:
    --------
    str
        Path to the pickle file
    """
    image_data_with_features_pkl = os.path.join(metadata_dir, model_name, 'image_data_features.pkl')
    return image_data_with_features_pkl

def image_features_vectors_idx(model_name: str, metadata_dir: str = 'metadata-files') -> str:
    """
    Get the path to the image features vectors index file.

    Parameters:
    -----------
    model_name : str
        Name of the model
    metadata_dir : str, optional (default='metadata-files')
        Base directory for metadata files

    Returns:
    --------
    str
        Path to the FAISS index file
    """
    image_features_vectors_idx = os.path.join(metadata_dir, model_name, 'image_features_vectors.idx')
    return image_features_vectors_idx
