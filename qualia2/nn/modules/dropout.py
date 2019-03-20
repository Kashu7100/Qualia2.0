# -*- coding: utf-8 -*-
from .module import Module
from ...core import * 
from ...functions import dropout
from ...autograd import Tensor 

class Dropout(Module):
    '''Dropout\n
    During training, randomly zeroes some of the elements of the input tensor with probability p using samples from a Bernoulli distribution. 
    Each channel will be zeroed out independently on every forward call.
    
    Args:
        p (float): probability of an element to be zeroed. Default: 0.5

    Shape:
        - Input: Any
        - Output: same as input
    '''
    def __init__(self, p=0.5):
        super().__init__()
        self.num_params = 0
        self.p = p

    def __repr__(self):
        return '{}(p={}) at 0x{:0{}X}'.format(self.__class__.__name__, self.p, id(self), 16)
    
    def forward(self, x): 
        result = dropout(x, self.p, self.training)
        if self.input_shape is None:
            self.input_shape = x.shape
        if self.output_shape is None:
            self.output_shape = result.shape
        return result