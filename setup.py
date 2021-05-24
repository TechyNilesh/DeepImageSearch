from distutils.core import setup
setup(
  name = 'DeepImageSearch',         
  packages = ['DeepImageSearch'],
  version = '0.1',
  license='MIT',        
  description = 'Deep Image Search is an AI-based image search engine that includes deep transfor learning features Extraction and tree-based vectorized search.',
  author = 'Nilesh Verma',                   
  author_email = 'me@nileshverma.com',      # Type in your E-Mail
  url = 'https://github.com/user/reponame',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/user/reponame/archive/v_01.tar.gz',    # I explain this later on
  keywords = ['Deep Image Search Engine', 'AI Image search', 'Image Search Python'],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
          'annoy',
          'matplotlib',
          'pandas',
          'numpy',
          'tensorflow',
          'tqdm'
          'Pillow'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',
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