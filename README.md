<p align="center">
  <img src="https://kashu.ml/wp-content/uploads/2018/08/qualia-1-700x379.png?raw=true" alt="Qualia Logo"/>
</p>

Qualia is a deep learning framework deeply integrated with autograd designed for maximum flexibility. Qualia features in automatic differentiation and dynamic graphing with CUDA acceleration. The optimized code offers faster and stable processing than the [previous version](https://github.com/Kashu7100/Qualia).

## Introduction

David J. Chalmers, an Australian philosopher and cognitive scientist, onece argued that if a system reproduces the functional organization of the brain, it will also reproduce the qualia associated with the brain in the paper "[Absent Qualia, Fading Qualia, Dancing Qualia](http://consc.net/papers/qualia.html)." This library "Qualia" named after the series of arguments in philosophy of mind associated with the qualia, hoping for the creation of a system with subjective consciousness. 

## Overview
Qualia2.0 is a library that consists of the following components:

| Component | Description |
| ---- | --- |
| **qualia2.autograd** | supports a dynamic automatic differentiation |
| [**qualia2.applications**](qualia2/applications) | provides implemented models for handy testing |
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

* [Python 3.6.0+](https://www.python.org/)
* [Anaconda](https://www.anaconda.com/distribution/)

## Installation

```bash
UNDER DEV
```

## Tutorial
Detailed tutorial of Qualia2.0 can be found [here](/tutorial).
- [Automatic Differentiation](/tutorial/#automatic_differentiation)
- [Validation of Automatic Differentiation](/tutorial/#valid_automatic_differentiation)
- [Network Definition](/tutorial/#network_definition)
- [Model Summary](/tutorial/#model_summary)
- [Saving/Loading a Trained Weights](/tutorial/#save_load)
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
