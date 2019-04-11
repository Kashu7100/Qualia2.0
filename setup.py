# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

setup(name='qualia2',
      version='0.0.1',
      description='Qualia2.0 is a deep learning framework deeply integrated with automatic differentiation and dynamic graphing with CUDA acceleration. ',
      author='Kashu',
      author_email='echo_six0566@yahoo.co.jp',
      url='https://github.com/Kashu7100/Qualia2.0',
      packages=find_packages(),
      install_requires=[
            'numpy',
            'matplotlib',
            'h5py'
      ],)
