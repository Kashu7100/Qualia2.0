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
| **qualia2.nn** | a neural networks library deeply integrated with autograd with CUDA acceleration |
| [**qualia2.data**]('/qualia2/data') | provides datasets for handy testing |
| **qualia2.config** | select whether to use GPU |
| **qualia2.functions** | predefined functions are presented |
| **qualia2.util** | utility functions for convenience |

## Requirements

* [NVIDIA CUDA GPU](https://developer.nvidia.com/cuda-gpus): Compute Capability of the GPU must be at least 3.0.
* [CUDA Toolkit](https://developer.nvidia.com/cuda-zone): Supported Versions: 8.0, 9.0, 9.1, 9.2 and 10.0.

    (*Note: you can still use Qualia2.0 without GPU*)

* [Python 3.6.0+](https://www.python.org/)

## Installation

```bash
UNDER DEV
```

## Tutorial
Detailed tutorial of Qualia2.0 can be found [here](/tutorial).

## Examples
Examples can be found [here](/examples).

## Citation

Please cite **Qualia** if you use in your research.
```
Kashu Y., Qualia2.0 (2019). Automatic Differentiation and Dynamic Graphing with CUDA for Deep Learning Application, <https://github.com/Kashu7100/Qualia2.0>
```
BibTex
```bibtex
@software{qualia,
  title = {{Q}ualia2.0},
  author = {Kashu Yamazaki},
  year = {2019},
  keywords = {Python, Automatic Differentiation, Dynamic Graphing, CUDA, Deep Learning}
  publisher = {GitHub},
  url = {\url{https://github.com/Kashu7100/Qualia2.0}},
}
```

## License

Source codes in the repository follows [MIT](http://www.opensource.org/licenses/MIT) license.
