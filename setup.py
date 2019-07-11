# -*- coding: utf-8 -*-
from setuptools import setup, find_packages
import sys

requires = ['numpy',
            'matplotlib',
            'h5py',
            'gym',
            'kaggle'
           ]

if '--cuda' in sys.argv:
      idx = sys.argv.index('--cuda')
      sys.argv.pop(idx)
      cuda = sys.argv.pop(idx)
      requires.append('cupy-cuda{}'.format(cuda))            

setup(name='qualia2',
      version='0.0.1',
      description='Qualia2.0 is a deep learning framework deeply integrated with automatic differentiation and dynamic graphing with CUDA acceleration. ',
      author='Kashu',
      author_email='echo_six0566 {at} yahoo.co.jp',
      url='https://github.com/Kashu7100/Qualia2.0',
      license='MIT',
      packages=find_packages(),
      install_requires=requires,)
