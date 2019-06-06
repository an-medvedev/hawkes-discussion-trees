from distutils.core import setup

setup(
  name = 'hawkes-discussion-trees',
  packages = [],
  version = '0.0.1',
  license='MIT',
  description = 'Hawkes model for discussion trees. Infer the parameters, generate the branching tree from the model.',
  author = 'Alexey Medvedev',
  author_email = 'an_medvedev@yahoo.com',
  url = 'https://github.com/an-medvedev/hawkes-discussion-trees',
  download_url = 'https://github.com/an-medvedev/hawkes-discussion-trees',
  keywords = ['discussion tree', 'Hawkes process', 'statistical inference'],
  install_requires=[
          'networkx',
          'numpy',
          'scipy',
          'warnings'
      ],
  classifiers=[  # Optional
    # How mature is this project? Common values are
    #   3 - Alpha
    #   4 - Beta
    #   5 - Production/Stable
    'Development Status :: 3 - Alpha',

    # Indicate who your project is intended for
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering :: Information Analysis'

    # Pick your license as you wish
    'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',

    # Specify the Python versions you support here. In particular, ensure
    # that you indicate whether you support Python 2, Python 3 or both.
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)