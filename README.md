<p align="center">
  <img src="/assets/qualia.png" alt="Qualia Logo"/>
</p>

Qualia is a deep learning framework deeply integrated with automatic differentiation and dynamic graphing with CUDA acceleration. Thanks to the define-by-run API, the code written with Qualia enjoys high modularity.

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
| **qualia2.autograd** | provides a Tensor object for a dynamic automatic differentiation |
| **qualia2.functions** | pre-defined functions capable of automatic differentiation |
| **qualia2.nn** | a neural networks library deeply integrated with autograd with CUDA acceleration |
| [**qualia2.data**](/qualia2/data) | datasets for handy testing |
| **qualia2.rl** | reinforcement learning models and utilities |
| **qualia2.util** | utility functions for convenience |
| [**qualia2.vision**](/qualia2/vision) | pretrained model architectures for computer vision |

## Requirements

* [NVIDIA CUDA GPU](https://developer.nvidia.com/cuda-gpus): Compute Capability of the GPU must be at least 3.0.
* [CUDA Toolkit](https://developer.nvidia.com/cuda-zone): Supported Versions: 8.0, 9.0, 9.1, 9.2, 10.0, and 10.1. 
* [Python 3.6](https://www.python.org/)

>   *Note: Qualia2.0 is also available for CPU use*

## Installation
Upgrade of setuptools and pip is recommended before the installation:
```bash
$ pip install -U setuptools pip
```
CUDA Toolkit version can be found by:
```bash
$ nvcc --version
```
Clone Github repo and cd to Qualia2.0 to install:
```bash
$ git clone https://github.com/Kashu7100/Qualia2.0.git
$ cd Qualia2.0
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
See [more](https://kashu7100.github.io/Qualia2.0/install.html) in docs.

## Docs
Online document is available [here](https://kashu7100.github.io/Qualia2.0). 

## Examples
More examples can be found [here](/examples).

### [Supervised learning](/examples/supervised_learning)

[<img src="/assets/spiral_boundary.png" height="200"/>](/examples/supervised_learning/spiral)<img src="/assets/openpose_hand.gif" height="200"/>[<img src="/assets/baseball.gif" height="220"/>](/examples/supervised_learning/openpose)

### [Unsupervised learning](examples/unsupervised_learning)

[<img src="/assets/lorenz_compare.png" height="145"/>](examples/unsupervised_learning/lorenz_system)[<img src="/assets/gan_mnist.gif" height="200"/>](examples/unsupervised_learning/mnist)

### [Reinforcement learning](/examples/reinforcement_learning)

[<img src="/assets/cartpole_dqn.gif" height="180"/>](/examples/reinforcement_learning/inverted_pendulum)[<img src="/assets/mountaincar_duelingnet.gif" height="180"/>](examples/reinforcement_learning/mountain_car)[<img src="/assets/roboschool_walker2d_td3.gif" height="180"/>](examples/reinforcement_learning/roboschool_walker2d)[<img src="/assets/bipedal_walker_td3.gif" height="180"/>](examples/reinforcement_learning/bipedal_walker)[<img src="/assets/lunar_lander_cont_td3.gif" height="180"/>](examples/reinforcement_learning/lunar_lander)


## Citation
Please cite **Qualia** if you use the contents in this repository for your research or in a scientific publication.
```
Y. Kashu, Qualia2.0 - Automatic Differentiation and Dynamic Graphing with CUDA for Deep Learning Application, (2019), GitHub repository, https://github.com/Kashu7100/Qualia2.0
```
BibTex
```bibtex
@misc{qualia,
  author = {Kashu Yamazaki},
  title = {{Q}ualia2.0 - Automatic Differentiation and Dynamic Graphing with CUDA for Deep Learning Application},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  keywords = {Python, Automatic Differentiation, Dynamic Graphing, CUDA, Deep Learning}
  howpublished  = {\url{https://github.com/Kashu7100/Qualia2.0}},
}
```

## License
Source codes in the repository follows [MIT](http://www.opensource.org/licenses/MIT) license.

## References
References are listed in [wiki](https://github.com/Kashu7100/Qualia2.0/wiki/References-(Editing...))
