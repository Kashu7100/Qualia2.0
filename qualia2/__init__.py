# -*- coding: utf-8 -*- 
from .core import * 
from .autograd import *
from .functions import * 
from .util import *
from .util import _mul
from .nn import * 
from .data import *
from .applications import *
from .rl import *

pi = np.pi
e = np.e

def numel(obj):
    return _mul(*obj.shape)

def array(obj, dtype='float64'):
    return Tensor(np.array(obj, dtype=dtype))

def arange(*args, dtype='float64'):
    return Tensor(np.arange(*args, dtype=dtype))

def zeros(shape, dtype='float64'):
    return Tensor(np.zeros(shape, dtype=dtype))

def zeros_like(obj, dtype='float64'):
    return Tensor(np.zeros(obj.shape, dtype=dtype))

def ones(shape, dtype='float64'):
    return Tensor(np.ones(shape, dtype=dtype))

def ones_like(obj, dtype='float64'):
    return Tensor(np.ones(obj.shape, dtype=dtype))

def rand(*args, dtype='float64'):
    return Tensor(np.random.rand(*args).astype(dtype))

def rand_like(obj, dtype='float64'):
    return Tensor(np.random.rand(*obj.shape).astype(dtype))

def randn(*args, dtype='float64'):
    return Tensor(np.random.randn(*args).astype(dtype))

def randn_like(obj, dtype='float64'):
    return Tensor(np.random.randn(*obj.shape).astype(dtype))

def uniform(*args, dtype='float64'):
    return Tensor(np.random.uniform(*args).astype(dtype))
