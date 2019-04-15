# -*- coding: utf-8 -*- 
from .config import * 
from .core import * 
from .autograd import *
from .functions import * 
from .util import *
from .nn import * 
from .data import *
from .applications import *
from .environment import *

pi = np.pi
e = np.e

def array(obj):
    return Tensor(np.array(obj, dtype=dtype))

def arange(*args):
    return Tensor(np.arange(*args, dtype=dtype))

def zeros(shape):
    return Tensor(np.zeros(shape, dtype=dtype))

def zeros_like(obj):
    return Tensor(np.zeros(obj.shape, dtype=dtype))

def ones(shape):
    return Tensor(np.ones(shape, dtype=dtype))

def ones_like(obj):
    return Tensor(np.ones(obj.shape, dtype=dtype))

def rand(*args):
    return Tensor(np.random.rand(*args).astype(dtype))

def rand_like(obj):
    return Tensor(np.random.rand(*obj.shape).astype(dtype))

def randn(*args):
    return Tensor(np.random.randn(*args).astype(dtype))

def randn_like(obj):
    return Tensor(np.random.randn(*obj.shape).astype(dtype))

def uniform(*args):
    return Tensor(np.random.uniform(*args).astype(dtype))
