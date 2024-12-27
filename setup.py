from setuptools import setup, find_packages

setup(
    name='nullingexplorer',
    version='0.1.0',
    description='A framework for nulling interferometry exploration',
    author='Ruiting Ma',
    author_email='ruitingma@foxmail.com',
    url='https://github.com/Mary-Tim/Exoplanet-Nulling-Explorer',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'yaml',
        'tensordict',
        'h5py',
        'matplotlib',
        'iminuit',
        'torchquad',
        'uproot',
        'tqdm',
        'spectres',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
)