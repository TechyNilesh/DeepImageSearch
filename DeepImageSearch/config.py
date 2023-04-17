import os 

def image_data_with_features_pkl(model_name):
    image_data_with_features_pkl = os.path.join('metadata-files/',f'{model_name}/','image_data_features.pkl')
    return image_data_with_features_pkl

def image_features_vectors_idx(model_name):
    image_features_vectors_idx = os.path.join('metadata-files/',f'{model_name}/','image_features_vectors.idx')
    return image_features_vectors_idx
