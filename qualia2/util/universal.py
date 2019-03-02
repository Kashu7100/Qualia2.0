# -*- coding: utf-8 -*- 
from ..core import *
from ..autograd import Tensor
from functools import reduce

def numerical_grad(fn, tensor, *args, **kwargs):
    delta = 1e-4
    h1 = fn(tensor + delta, *args, **kwargs)
    h2 = fn(tensor - delta, *args, **kwargs)
    return np.divide(np.subtract(h1.data, h2.data), 2*delta)

def check_function(fn, *args, domain=(-1e3,1e3), **kwargs):
    arr = np.random.random_sample((100,100))
    x = Tensor(domain[0]*arr+domain[1]*(1-arr))
    out = fn(x, *args, **kwargs)
    out.backward()
    a_grad = x.grad
    n_grad = numerical_grad(fn, x, *args, **kwargs)
    sse = np.sum(np.power(np.subtract(a_grad, n_grad),2))
    print('[*] measured error: ', sse)
    assert sse < 1e-10

def _single(x):
    assert type(x) is int
    return x

def _pair(x):
    if type(x) is int:
        return (x,x)
    elif type(x) is tuple:
        assert len(x) is 2
        return x
    else:
        raise ValueError

def _triple(x):
    if type(x) is int:
        return (x,x,x)
    elif type(x) is tuple:
        assert len(x) is 3
        return x
    else:
        raise ValueError
    
def _mul(*args):
    return reduce(lambda a, b: a*b, args)