# -*- coding: utf-8 -*- 
from .core import * 
from .autograd import *
from .functions import * 
from .util import *
from .util import _mul
from .nn import * 
from .data import *
from .vision import *
from .rl import *

pi = np.pi
e = np.e

def copy(tensor):
    return Tensor(np.copy(tensor.data), tensor.requires_grad, tensor.dtype)

def numel(obj):
    return _mul(*obj.shape)

def array(obj, dtype='float64'):
    return Tensor(np.array(obj), dtype=dtype)

def arange(*args, dtype='float64'):
    return Tensor(np.arange(*args), dtype=dtype)

def empty(shape, dtype='float64'):
    return Tensor(np.empty(shape), dtype=dtype)

def empty_like(obj, dtype='float64'):
    return Tensor(np.empty(obj.shape), dtype=dtype)

def zeros(shape, dtype='int64'):
    return Tensor(np.zeros(shape), dtype=dtype)

def zeros_like(obj, dtype='int64'):
    return Tensor(np.zeros(obj.shape), dtype=dtype)

def ones(shape, dtype='int64'):
    return Tensor(np.ones(shape), dtype=dtype)

def ones_like(obj, dtype='int64'):
    return Tensor(np.ones(obj.shape), dtype=dtype)

def rand(*args, dtype='float64'):
    return Tensor(np.random.rand(*args), dtype=dtype)

def rand_like(obj, dtype='float64'):
    return Tensor(np.random.rand(*obj.shape), dtype=dtype)

def randn(*args, dtype='float64'):
    return Tensor(np.random.randn(*args), dtype=dtype)

def randn_like(obj, dtype='float64'):
    return Tensor(np.random.randn(*obj.shape), dtype=dtype)

def uniform(*args, dtype='float64'):
    return Tensor(np.random.uniform(*args), dtype=dtype)
