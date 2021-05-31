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
  version = '1.4',
  license='MIT',        
  description = 'Deep Image Search is an AI-based image search engine that includes deep transfor learning features Extraction and tree-based vectorized search.',
  long_description=README,
  author = 'Nilesh Verma',                   
  author_email = 'me@nileshverma.com',     
  url = 'https://github.com/TechyNilesh/DeepImageSearch',
  download_url = 'https://github.com/TechyNilesh/DeepImageSearch/archive/refs/tags/v_14.tar.gz',    
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
