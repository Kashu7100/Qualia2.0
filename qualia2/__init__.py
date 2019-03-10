# -*- coding: utf-8 -*- 
from .config import * 
from .core import * 
from .util import * 
from .autograd import *
from .functions import * 
from .nn import * 
from .data import *

pi = np.pi
e = np.e

def array(obj):
    return Tensor(np.array(obj))

def arange(*args):
    return Tensor(np.arange(*args))

def zeros(shape):
    return Tensor(np.zeros(shape))

def zeros_like(obj):
    return Tensor(np.zeros(obj.shape))

def ones(shape):
    return Tensor(np.ones(shape))

def ones_like(obj):
    return Tensor(np.ones(obj.shape))

def rand(*args):
    return Tensor(np.random.rand(*args))

def rand_like(obj):
    return Tensor(np.random.rand(*obj.shape))

def randn(*args):
    return Tensor(np.random.randn(*args))

def randn_like(obj):
    return Tensor(np.random.randn(*obj.shape))
