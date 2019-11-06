# -*- coding: utf-8 -*- 
from .helper import *
from .checkpoint import *

from .. import to_cpu
from ..core import *
from ..autograd import Tensor
from functools import reduce
import sys
from logging import getLogger
logger = getLogger('QualiaLogger').getChild('util')

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

def numerical_grad(fn, tensor, *args, **kwargs):
    delta = 1e-4
    h1 = fn(tensor + delta, *args, **kwargs)
    h2 = fn(tensor - delta, *args, **kwargs)
    return np.divide(np.subtract(h1.data, h2.data), 2*delta)

def check_function(fn, *args, x=None, domain=(-1e3,1e3), **kwargs):
    if x is None:
        arr = np.random.random_sample((100,100))
        x = Tensor((domain[1]-domain[0])*arr+domain[0])
    out = fn(x, *args, **kwargs)
    out.backward()
    a_grad = x.grad
    n_grad = numerical_grad(fn, x, *args, **kwargs)
    sse = np.sum(np.power(np.subtract(a_grad, n_grad),2))
    logger.info('[*] measured error: {}'.format(sse))
    return (a_grad, n_grad), sse

def progressbar(progress, process, text_before='', text_after=''):
    bar_length = 40
    block = int(round(bar_length*progress/process))
    sys.stdout.flush()
    text = '\r[*]{}progress: [{:.0f}%] |{}| {}/{} {}'.format(' '+text_before, progress/process*100, '#'*block + "-"*(bar_length-block), progress, process, text_after)
    sys.stdout.write(text)

def download_progress(count, block_size, total_size):
    sys.stdout.flush()
    sys.stdout.write('\r[*] downloading {:.2f}%'.format(float(count * block_size) / float(total_size) * 100.0))