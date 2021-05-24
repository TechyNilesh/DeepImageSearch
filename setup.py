from distutils.core import setup
from os import path

# The directory containing this file
here = path.abspath(path.dirname(__file__))

setup(from distutils.core import setup
import pathlib


# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setup(
  name = 'DeepImageSearch',         
  packages = ['DeepImageSearch'],
  version = '0.7',
  license='MIT',        
  description = 'Deep Image Search is an AI-based image search engine that includes deep transfor learning features Extraction and tree-based vectorized search.',
  long_description=README,
  long_description_content_type="text/markdown",
  author = 'Nilesh Verma',                   
  author_email = 'me@nileshverma.com',     
  url = 'https://github.com/TechyNilesh/DeepImageSearch',
  download_url = 'https://github.com/TechyNilesh/DeepImageSearch/archive/refs/tags/v_07.tar.gz',    
  keywords = ['Deep Image Search Engine', 'AI Image search', 'Image Search Python'],   
  install_requires=[        
          'annoy',
          'matplotlib',
          'pandas',
          'numpy',
          'tensorflow',
          'tqdm',
          'Pillow'
      ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers', 
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
  ],
)
  name = 'DeepImageSearch',         
  packages = ['DeepImageSearch'],
  version = '0.6',
  license='MIT',        
  description = 'Deep Image Search is an AI-based image search engine that includes deep transfor learning features Extraction and tree-based vectorized search.',
  long_description=open('README.rst', encoding="utf8").read(),
  author = 'Nilesh Verma',                   
  author_email = 'me@nileshverma.com',     
  url = 'https://github.com/TechyNilesh/DeepImageSearch',
  download_url = 'https://github.com/TechyNilesh/DeepImageSearch/archive/refs/tags/v_06.tar.gz',    
  keywords = ['Deep Image Search Engine', 'AI Image search', 'Image Search Python'],   
  install_requires=[        
          'annoy',
          'matplotlib',
          'pandas',
          'numpy',
          'tensorflow',
          'tqdm',
          'Pillow'
      ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers', 
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
  ],
)
