# -*- coding: utf-8 -*-
import math 
from .module import Module
from ...core import * 
from ...util import _single, _pair, _triple
from ...functions import maxpool1d, maxpool2d, maxpool3d
from ...autograd import Tensor 

class MaxPool1d(Module):
    '''Applies a 1D max pooling over an input signal composed of several input planes.\n
    Args:
        kernel_size (int): the size of the window to take a max over
        stride (int): the stride of the window. Default value is kernel_size
        padding (int): implicit zero padding to be added on all three sides
        dilation (int): a parameter that controls the stride of elements in the window
        return_indices (bool): if True, will return the max indices along with the outputs.

    Shape:
        - Input: [N,C,W]
        - Output: [N,C,W_out]
        
        W_out = (W+2*padding-dilation*(kernel_size-1)-1)/stride + 1
    '''
    def __init__(self, kernel_size=2, stride=2, padding=0, dilation=1, return_indices=False): 
        super().__init__()  
        self.kernel_size = _single(kernel_size)
        self.stride = _single(stride)
        self.padding = _single(padding)
        self.dilation = _single(dilation)
        self.num_params = 0
        self.return_indices = return_indices
    
    def __repr__(self):
        return '{}({}, stride={}, padding={}, dilation={}, return_indices={}) at 0x{:0{}X}'.format(self.__class__.__name__, self.kernel_size, self.stride, self.padding, self.dilation, str(self.return_indices), id(self), 16)

    def forward(self, x):
        return maxpool1d(x, self.kernel_size, self.stride, self.padding, self.dilation, self.return_indices)
        
class MaxPool2d(Module):
    '''Applies a 2D max pooling over an input signal composed of several input planes.\n
    Args:
        kernel_size (tuple of int): the size of the window to take a max over
        stride (tuple of int): the stride of the window. Default value is kernel_size
        padding (tuple of int): implicit zero padding to be added on all three sides
        dilation (tuple of int): a parameter that controls the stride of elements in the window
        return_indices (bool): if True, will return the max indices along with the outputs.

    Shape:
        - Input: [N,C,H,W]
        - Output: [N,C,H_out,W_out]

        H_out = (H+2*padding[0]-dilation[0]*(kernel_size[0]-1)-1)/stride[0] + 1
        W_out = (W+2*padding[1]-dilation[1]*(kernel_size[1]-1)-1)/stride[1] + 1
    '''
    def __init__(self, kernel_size=(2,2), stride=(2,2), padding=0, dilation=1, return_indices=False): 
        super().__init__() 
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.num_params = 0
        self.return_indices = return_indices

    def __repr__(self):
        return '{}({}, stride={}, padding={}, dilation={}, return_indices={}) at 0x{:0{}X}'.format(self.__class__.__name__, self.kernel_size, self.stride, self.padding, self.dilation, str(self.return_indices), id(self), 16)

    def forward(self, x):
        return maxpool2d(x, self.kernel_size, self.stride, self.padding, self.dilation, self.return_indices)

class MaxPool3d(Module):
    '''Applies a 3D max pooling over an input signal composed of several input planes.\n
    Args:
        kernel_size (tuple of int): the size of the window to take a max over
        stride (tuple of int): the stride of the window. Default value is kernel_size
        padding (tuple of int): implicit zero padding to be added on all three sides
        dilation (tuple of int): a parameter that controls the stride of elements in the window
        return_indices (bool): if True, will return the max indices along with the outputs.

    Shape:
        - Input: [N,C,H,W,D]
        - Output: [N,C,H_out,W_out,D_out]

        H_out = (H+2*padding[0]-dilation[0]*(kernel_size[0]-1)-1)/stride[0] + 1
        W_out = (W+2*padding[1]-dilation[1]*(kernel_size[1]-1)-1)/stride[1] + 1
        D_out = (D+2*padding[2]-dilation[2]*(kernel_size[2]-1)-1)/stride[2] + 1
    '''
    def __init__(self, kernel_size=(2,2,2), stride=(2,2,2), padding=0, dilation=1, return_indices=False): 
        super().__init__() 
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.padding = _triple(padding)
        self.dilation = _triple(dilation)
        self.num_params = 0
        self.return_indices = return_indices

    def __repr__(self):
        return '{}({}, stride={}, padding={}, dilation={}, return_indices={}) at 0x{:0{}X}'.format(self.__class__.__name__, self.kernel_size, self.stride, self.padding, self.dilation, str(self.return_indices), id(self), 16)

    def forward(self, x):
        return maxpool3d(x, self.kernel_size, self.stride, self.padding, self.dilation, self.return_indices)
