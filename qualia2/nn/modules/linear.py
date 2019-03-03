# -*- coding: utf-8 -*-
import math 
from .module import Module
from ...core import * 
from ...functions import linear
from ...autograd import Tensor 

class Linear(Module):
    '''Applies a linear transformation to the incoming data\n 
    Model: 
        y = x*w.T + b 
     
    Args: 
        in_features (int): size of each input sample 
        out_features (int): size of each output sample 
        bias (bool): whether to use bias. Default: True 
     
    Shape: 
        - Input: [N, *, in_features] where '∗' means any number of additional dimensions. 
        - Output: [N, *, out_features] where '∗' means any number of additional dimensions. 
    ''' 
    def __init__(self, in_features, out_features, bias=True): 
        super().__init__() 
        self.in_features = in_features
        self.out_features = out_features
        self.num_params += in_features*out_features
        self.weight = Tensor(np.random.normal(0, math.sqrt(1/in_features),(in_features, out_features))) 
        if bias: 
            self.bias = Tensor(np.zeros(out_features)) 
            self.num_params += out_features
        else: 
            self.bias = None
            
    def __repr__(self):
        return '{}({}, {}, bias={}) at 0x{:0{}X}'.format(self.__class__.__name__, self.in_features, self.out_features, str(self.bias is not None), id(self), 16)
    
    def forward(self, x): 
        result = linear(x, self.weight, self.bias)
        if self.input_shape is None:
            self.input_shape = x.shape
        if self.output_shape is None:
            self.output_shape = result.shape
        return result
