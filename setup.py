from setuptools import setup
import pathlib


# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setup(
  long_description_content_type="text/markdown",
  name = 'DeepImageSearch',         
  packages = ['DeepImageSearch'],
  version = '2.1',
  license='MIT',        
  description = 'Deep Image Search is an AI-based image search engine that incorporates ViT (Vision Transformer) for feature extraction and utilizes a tree-based vectorized search technique.',
  long_description=README,
  author = 'Nilesh Verma',                   
  author_email = 'me@nileshverma.com',     
  url = 'https://github.com/TechyNilesh/DeepImageSearch',
  download_url = 'https://github.com/TechyNilesh/DeepImageSearch/archive/refs/tags/v_20.tar.gz',    
  keywords = ['Deep Image Search Engine', 'AI Image search', 'Image Search Python'],   
  install_requires=[        
          'annoy',
          'matplotlib',
          'pandas',
          'numpy',
          'torch',
          'tqdm',
          'Pillow',
          'timm',
          'torchvision'
      ],
  classifiers=[
    'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Developers', 
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10'
  ],
)
