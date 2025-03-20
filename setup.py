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
  version = '2.5',
  license='MIT',        
  description = 'DeepImageSearch is a Python library for fast and accurate image search. It offers seamless integration with Python, GPU support, and advanced capabilities for identifying complex image patterns using the Vision Transformer models.',
  long_description=README,
  author = 'Nilesh Verma',                   
  author_email = 'me@nileshverma.com',     
  url = 'https://github.com/TechyNilesh/DeepImageSearch',
  download_url = 'https://github.com/TechyNilesh/DeepImageSearch/archive/refs/tags/v_25.tar.gz',    
  keywords = ['Deep Image Search Engine', 'AI Image search', 'Image Search Python'],   
  install_requires=[        
    'faiss_cpu>=1.7.3,<1.8.0',
    'torch>=2.0.0,<2.1.0',
    'torchvision>=0.15.1,<0.16.0',
    'matplotlib>=3.5.2,<3.6.0',
    'pandas>=1.4.3,<1.5.0',
    'numpy>=1.24.2,<1.25.0',
    'tqdm>=4.65.0,<5.0.0',
    'Pillow>=9.5.0,<10.0.0',
    'timm>=0.6.13,<0.7.0'
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
