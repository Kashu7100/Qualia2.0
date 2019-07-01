<p align="center">
  <img src="/assets/qualia.png" alt="Qualia Logo"/>
</p>

Qualia is a deep learning framework deeply integrated with automatic differentiation and dynamic graphing with CUDA acceleration. 

## Introduction
David J. Chalmers, an Australian philosopher and cognitive scientist, onece argued that if a system reproduces the functional organization of the brain, it will also reproduce the qualia associated with the brain in the paper "[Absent Qualia, Fading Qualia, Dancing Qualia](http://consc.net/papers/qualia.html)." This library "Qualia" named after the series of arguments in philosophy of mind associated with the qualia, hoping for the creation of a system with subjective consciousness. 

## Overview

[![Build](https://img.shields.io/badge/build-passing-success.svg)](https://github.com/Kashu7100/Qualia2.0/)
[![Version](https://img.shields.io/badge/package-v0.0.1-informational.svg)](https://github.com/Kashu7100/Qualia2.0/)
[![Size](https://img.shields.io/github/repo-size/Kashu7100/Qualia2.0.svg)](https://github.com/Kashu7100/Qualia2.0/)
[![License: MIT](https://img.shields.io/github/license/Kashu7100/Qualia2.0.svg)](/LICENSE)

The main components of Qualia2.0 is listed below:

| Component | Description |
| ---- | --- |
| **qualia2.autograd** | supports a dynamic automatic differentiation |
| **qualia2.functions** | pre-defined functions capable of automatic differentiation |
| **qualia2.nn** | a neural networks library deeply integrated with autograd with CUDA acceleration |
| **qualia2.applications** | provides implemented deep leaning models for handy testing |
| [**qualia2.data**](/qualia2/data) | provides datasets for handy testing |
| **qualia2.environment** | provides environments for handy testing of reinforcement learning |
| **qualia2.util** | utility functions for convenience |

## Requirements

* [NVIDIA CUDA GPU](https://developer.nvidia.com/cuda-gpus): Compute Capability of the GPU must be at least 3.0.
* [CUDA Toolkit](https://developer.nvidia.com/cuda-zone): Supported Versions: 8.0, 9.0, 9.1, 9.2, 10.0, and 10.1. (*Note: Qualia2.0 is also available for CPU use*)
* [Python 3.6](https://www.python.org/)

## Installation
Upgrade of setuptools and pip is recommended before the installation:
```bash
$ pip install -U setuptools pip
```
CUDA Toolkit version can be found by:
```bash
$ nvcc --version
```
Depending on the CUDA version you have installed on your host, choose the best option from following.
```bash
(For CUDA 8.0)
$ python setup.py install --cuda 80
(For CUDA 9.0)
$ python setup.py install --cuda 90
(For CUDA 9.1)
$ python setup.py install --cuda 91
(For CUDA 9.2)
$ python setup.py install --cuda 92
(For CUDA 10.0)
$ python setup.py install --cuda 100
(For CUDA 10.1)
% python setup.py install --cuda 101
(For without CUDA)
$ python setup.py install
```
See [more](https://github.com/Kashu7100/Qualia2.0/wiki/Installation-Guide) in wiki.

## Tutorial
Detailed tutorial of Qualia2.0 can be found [here](/tutorial).

| Component | Description |
| ---- | --- |
| [Automatic Differentiation](/tutorial/#automatic_differentiation) | usage of automatic differentiation with simple example |
| [Validation of Automatic Differentiation](/tutorial/#valid_automatic_differentiation) | numerical method to validate automatic differentiation |
| [Qualia Tensor](/tutorial/#qualia_tensor) | Tensor class for automatic differentiation in Qualia |
| [Network Definition](/tutorial/#network_definition) | create a custom neural network model with Qualia |
| [Model Summary](/tutorial/#model_summary) | get the summary of the neural network model |
| [Saving/Loading Weights](/tutorial/#save_load) | save and load the trained weights |
| [Setting up Optimizer](/tutorial/#optim_setup) | preparing optimizers to train a neural network |
| [Learning Qualia with Examples](/tutorial/#ex) | examples that cover essentials of Qualia |

## Examples
Examples can be found [here](/examples).

## Citation
Please cite **Qualia** if you use the contents in this repository for your research or in a scientific publication.
```
Y. Kashu, Qualia2.0 - Automatic Differentiation and Dynamic Graphing with CUDA for Deep Learning Application, (2019), GitHub repository, https://github.com/Kashu7100/Qualia2.0
```
BibTex
```bibtex
@misc{qualia,
  author = {Kashu Yamazaki},
  title = {{Q}ualia2.0},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  keywords = {Python, Automatic Differentiation, Dynamic Graphing, CUDA, Deep Learning}
  howpublished  = {\url{https://github.com/Kashu7100/Qualia2.0}},
}
```

## License
Source codes in the repository follows [MIT](http://www.opensource.org/licenses/MIT) license.
