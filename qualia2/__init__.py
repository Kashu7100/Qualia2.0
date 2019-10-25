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

def seed(seed=None):
    np.random.seed(seed)

def copy(tensor):
    return Tensor(np.copy(tensor.data), tensor.requires_grad, tensor.dtype)

def numel(obj):
    return _mul(*obj.shape)

def array(obj, dtype='float64'):
    return Tensor(obj, dtype=dtype)

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

def uniform(low=0.0, high=1.0, shape=None, dtype='float64'):
    return Tensor(np.random.uniform(low, high, shape), dtype=dtype)

def uniform_like(obj, low=0.0, high=1.0, dtype='float64'):
    return Tensor(np.random.uniform(low, high, *obj.shape), dtype=dtype)

def normal(mean=0, std=1, shape=None, dtype='float64'):
    return Tensor(np.random.normal(mean, std, shape), dtype=dtype)

def normal_like(obj, mean=0, std=1, dtype='float64'):
    return Tensor(np.random.normal(mean, std, *obj.shape), dtype=dtype)
    
def randint(low, high=None, shape=None, dtype='float64'):
    return Tensor(np.random.randint(low, high, shape), dtype=dtype)

def randint_like(obj, low, high=None, dtype='float64'):
    return Tensor(np.random.randint(low, high, *obj.shape), dtype=dtype)

def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype='float64'):
    return Tensor(np.linspace(start, stop, num, endpoint, retstep), dtype=dtype)

def meshgrid(*tensors, **kwargs):
    return Tensor(np.meshgrid(*[t.data for t in tensors], **kwargs))
