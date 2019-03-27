<p align="center">
  <img src="https://kashu.ml/wp-content/uploads/2018/08/qualia-1-700x379.png?raw=true" alt="Qualia Logo"/>
</p>

------

[![Build](https://img.shields.io/badge/build-passing-success.svg)](https://github.com/Kashu7100/Qualia2.0/blob/master/)
[![License: MIT](https://img.shields.io/badge/license-MIT-informational.svg)](/LICENSE.md)
[![Tweet](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/intent/tweet?text=Qualia2)

Qualia is a deep learning framework deeply integrated with automatic differentiation designed for maximum flexibility. Qualia features in automatic differentiation and dynamic graphing with CUDA acceleration. Qualia was designed from scratch to have similar interface as PyTorch or Chainer.

## Introduction
David J. Chalmers, an Australian philosopher and cognitive scientist, onece argued that if a system reproduces the functional organization of the brain, it will also reproduce the qualia associated with the brain in the paper "[Absent Qualia, Fading Qualia, Dancing Qualia](http://consc.net/papers/qualia.html)." This library "Qualia" named after the series of arguments in philosophy of mind associated with the qualia, hoping for the creation of a system with subjective consciousness. 

## Overview
Qualia2.0 is a library that consists of the following components:

| Component | Description |
| ---- | --- |
| **qualia2.autograd** | supports a dynamic automatic differentiation |
| **qualia2.applications** | provides implemented models for handy testing |
| **qualia2.nn** | a neural networks library deeply integrated with autograd with CUDA acceleration |
| [**qualia2.data**](/qualia2/data) | provides datasets for handy testing |
| **qualia2.environment** | provides environments for handy testing of reinforcement learning |
| **qualia2.config** | select whether to use GPU |
| **qualia2.functions** | pre-defined functions capable of automatic differentiation |
| **qualia2.util** | utility functions for convenience |

## Requirements

* [NVIDIA CUDA GPU](https://developer.nvidia.com/cuda-gpus): Compute Capability of the GPU must be at least 3.0.
* [CUDA Toolkit](https://developer.nvidia.com/cuda-zone): Supported Versions: 8.0, 9.0, 9.1, 9.2 and 10.0.

    (*Note: you can still use Qualia2.0 without GPU*)

* [Python 3.6](https://www.python.org/)

## Installation
### Ubuntu

```bash
UNDER DEV
```
### Windows

```bash
UNDER DEV
```

in case pip from source did not work, use wheel from following links:
[ atari-py](https://github.com/Kojoley/atari-py/releases)
[ pybox2d](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pybox2d)

## Tutorial
Detailed tutorial of Qualia2.0 can be found [here](/tutorial).
- [Automatic Differentiation](/tutorial/#automatic_differentiation)
- [Validation of Automatic Differentiation](/tutorial/#valid_automatic_differentiation)
- [Network Definition](/tutorial/#network_definition)
- [Model Summary](/tutorial/#model_summary)
- [Saving/Loading a Trained Weights](/tutorial/#save_load)
- [Setting up Optimizer](/tutorial/#optim_setup)
- [Example with Spiral Dataset - Decision Boundary](/tutorial/#ex1)
- [Example with MNIST Dataset - PCA](/tutorial/#ex2)
- [Example with CartPole - DQN](/tutorial/#ex3)
- [Example with MountainCar - Dueling Network](/tutorial/#ex4)

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
