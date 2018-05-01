#!/usr/bin/env python
from setuptools import setup

setup(
    name='torchnets',
    version='0.1a1',
    description='Implementations of useful neural nets in pytorch',
    url='https://github.com/jeffkinnison/pyrameter',
    author='Jeff Kinnison',
    author_email='jkinniso@nd.edu',
    packages=['torchnets',
              'torchnets.inception',
              'torchnets.lenet',
              'torchnets.loaders',
              'torchnets.resnets',
              'torchnets.unet',
              'torchnets.vgg',
              'torchnets.utils',],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Users',
        'License :: MIT',
        'Topic :: Machine Learning :: Neural Networks',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS',
    ],
    keywords='machine_learning neural_networks',
    install_requires=[
        'torch>=0.3.0',
        'torchvision>=0.2.0'
    ],
)
