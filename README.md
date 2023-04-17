# Deep Image Search - AI-Based Image Search Engine
<p align="center"><img src="https://raw.githubusercontent.com/TechyNilesh/DeepImageSearch/786e96c48561d67be47dccbab2bc8debced414a3/images/deep%20image%20search%20logo%20New.png" alt="Deep+Image+Search+logo" height="218" width="350"></p>

**Deep Image Search** is a Python library that allows you to search for similar images in a dataset using deep learning techniques. It uses pre-trained models from the `timm` and Facebook `faiss` for indexing and searching images. It supports more than 500+ pre-trained computer vision models for image feature extraction and is compatible with both CPU and GPU versions of `faiss`.

![Generic badge](https://img.shields.io/badge/AI-Advance-green.svg) ![Generic badge](https://img.shields.io/badge/Python-v3-blue.svg) ![Generic badge](https://img.shields.io/badge/pip-v3-red.svg)
![Generic badge](https://img.shields.io/badge/ViT-Vision_Transformer-g.svg)   ![Generic badge](https://img.shields.io/badge/TorchVision-v0.15-orange.svg) ![Generic badge](https://img.shields.io/badge/FAISS-latest-green.svg) [![Downloads](https://static.pepy.tech/personalized-badge/deepimagesearch?period=total&units=none&left_color=grey&right_color=green&left_text=Downloads)](https://pepy.tech/project/deepimagesearch)

## Developed By

### [Nilesh Verma](https://nileshverma.com "Nilesh Verma")

## Features
- You can now load more than 500+ pre-trained state-of-the-art computer vision models available on [timm](https://timm.fast.ai/).
- Faster Search using [FAISS (Facebook AI Similarity Search)](https://github.com/facebookresearch/faiss).
- Highly Accurate Output Results.
- Best for implementing on Python-based web applications or APIs.
- Ideal for college students and freshers for project creation.
- Applications include image-based e-commerce recommendations, social media, and other image-based platforms that want to implement image recommendations and search.

## Installation

This library is compatible with both *windows* and *Linux system* you can just use **PIP command** to install this library on your system:

```shell
pip install DeepImageSearch --upgrade
```

## How To Use?

We have provided the **Demo** folder under the *GitHub repository*, you can find the example in both **.py** and **.ipynb**  file. Following are the ideal flow of the code:

```python
from DeepImageSearch import Load_Data, Search_Setup

# Load images from a folder
image_list = Load_Data().from_folder(['folder_path'])

# Set up the search engine
st = Search_Setup(image_list=image_list, model_name='vgg19', pretrained=True, image_count=100)

# Index the images
st.run_index()

# Get metadata
metadata = st.get_image_metadata_file()

# Add images to the index
st.add_images_to_index(['image_path_1', 'image_path_2'])

# Get similar images
st.get_similar_images(image_path='image_path', number_of_images=10)

# Plot similar images
st.plot_similar_images(image_path='image_path', number_of_images=9)

# Update metadata
metadata = st.get_image_metadata_file()
```

This code demonstrates how to load images, set up the search engine, index the images, add new images to the index, and retrieve similar images.

## Documentation

This project aims to provide a powerful image search engine using deep learning techniques. To get started, please follow the link: [Read Full Documents](https://github.com/TechyNilesh/DeepImageSearch/blob/main/Documents/Document.md)

## Screenshot

<p align="center"><img src="https://raw.githubusercontent.com/TechyNilesh/DeepImageSearch/786e96c48561d67be47dccbab2bc8debced414a3/images/Deep-Image-Search-Demo-Screenshot.png?" alt="Brain+Machine" height="auto" width="auto"></p>

## Citaion

If you use DeepImageSerach in your Research/Product, please cite the following GitHub Repository:

```latex
@misc{TechyNilesh/DeepImageSearch,
  author = {VERMA, NILESH},
  title = {Deep Image Search - AI-Based Image Search Engine},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/TechyNilesh/DeepImageSearch}},
}
```

### Please do STAR the repository, if it helped you in anyway.

**More cool features will be added in future. Feel free to give suggestions, report bugs and contribute.**
