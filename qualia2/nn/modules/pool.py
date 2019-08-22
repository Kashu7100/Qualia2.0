# -*- coding: utf-8 -*-
from .module import Module
from ...core import * 
from ...util import _single, _pair, _triple
from ...functions import maxpool1d, maxpool2d, maxpool3d, avepool1d, avepool2d, avepool3d, globalavepool1d, globlavepool2d, globalavepool3d, maxunpool1d, maxunpool2d, maxunpool3d
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
        self.return_indices = return_indices
    
    def __repr__(self):
        return '{}({}, stride={}, padding={}, dilation={}, return_indices={}) at 0x{:0{}X}'.format(self.__class__.__name__, self.kernel_size, self.stride, self.padding, self.dilation, str(self.return_indices), id(self), 16)

    def forward(self, x):
        if self.return_indices:
            result, idx = maxpool1d(x, self.kernel_size, self.stride, self.padding, self.dilation, self.return_indices)
            if self.input_shape is None:
                self.input_shape = x.shape
            if self.output_shape is None:
                self.output_shape = result.shape
            return result, idx
        else:
            result = maxpool1d(x, self.kernel_size, self.stride, self.padding, self.dilation, self.return_indices)
            if self.input_shape is None:
                self.input_shape = x.shape
            if self.output_shape is None:
                self.output_shape = result.shape
            return result
        
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
        self.return_indices = return_indices

    def __repr__(self):
        return '{}({}, stride={}, padding={}, dilation={}, return_indices={}) at 0x{:0{}X}'.format(self.__class__.__name__, self.kernel_size, self.stride, self.padding, self.dilation, str(self.return_indices), id(self), 16)

    def forward(self, x):
        if self.return_indices:
            result, idx = maxpool2d(x, self.kernel_size, self.stride, self.padding, self.dilation, self.return_indices)
            if self.input_shape is None:
                self.input_shape = x.shape
            if self.output_shape is None:
                self.output_shape = result.shape
            return result, idx
        else:
            result = maxpool2d(x, self.kernel_size, self.stride, self.padding, self.dilation, self.return_indices)
            if self.input_shape is None:
                self.input_shape = x.shape
            if self.output_shape is None:
                self.output_shape = result.shape
            return result

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
        self.return_indices = return_indices

    def __repr__(self):
        return '{}({}, stride={}, padding={}, dilation={}, return_indices={}) at 0x{:0{}X}'.format(self.__class__.__name__, self.kernel_size, self.stride, self.padding, self.dilation, str(self.return_indices), id(self), 16)

    def forward(self, x):
        if self.return_indices:
            result, idx = maxpool3d(x, self.kernel_size, self.stride, self.padding, self.dilation, self.return_indices)
            if self.input_shape is None:
                self.input_shape = x.shape
            if self.output_shape is None:
                self.output_shape = result.shape
            return result, idx
        else:
            result = maxpool3d(x, self.kernel_size, self.stride, self.padding, self.dilation, self.return_indices)
            if self.input_shape is None:
                self.input_shape = x.shape
            if self.output_shape is None:
                self.output_shape = result.shape
            return result

class AvgPool1d(Module):
    '''Applies a 1D average pooling over an input signal composed of several input planes.\n
    Args:
        kernel_size (int): the size of the window to take a max over
        stride (int): the stride of the window. Default value is kernel_size
        padding (int): implicit zero padding to be added on all three sides
        dilation (int): a parameter that controls the stride of elements in the window

    Shape:
        - Input: [N,C,W]
        - Output: [N,C,W_out]
        
        W_out = (W+2*padding-dilation*(kernel_size-1)-1)/stride + 1
    '''
    def __init__(self, kernel_size=2, stride=2, padding=0, dilation=1): 
        super().__init__()  
        self.kernel_size = _single(kernel_size)
        self.stride = _single(stride)
        self.padding = _single(padding)
        self.dilation = _single(dilation)
    
    def __repr__(self):
        return '{}({}, stride={}, padding={}, dilation={}) at 0x{:0{}X}'.format(self.__class__.__name__, self.kernel_size, self.stride, self.padding, self.dilation, id(self), 16)

    def forward(self, x):
        result = avepool1d(x, self.kernel_size, self.stride, self.padding, self.dilation)
        if self.input_shape is None:
            self.input_shape = x.shape
        if self.output_shape is None:
            self.output_shape = result.shape
        return result

class AvgPool2d(Module):
    '''Applies a 2D average pooling over an input signal composed of several input planes.\n
    Args:
        kernel_size (tuple of int): the size of the window to take a max over
        stride (tuple of int): the stride of the window. Default value is kernel_size
        padding (tuple of int): implicit zero padding to be added on all three sides
        dilation (tuple of int): a parameter that controls the stride of elements in the window

    Shape:
        - Input: [N,C,H,W]
        - Output: [N,C,H_out,W_out]

        H_out = (H+2*padding[0]-dilation[0]*(kernel_size[0]-1)-1)/stride[0] + 1
        W_out = (W+2*padding[1]-dilation[1]*(kernel_size[1]-1)-1)/stride[1] + 1
    '''
    def __init__(self, kernel_size=(2,2), stride=(2,2), padding=0, dilation=1): 
        super().__init__() 
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)

    def __repr__(self):
        return '{}({}, stride={}, padding={}, dilation={}) at 0x{:0{}X}'.format(self.__class__.__name__, self.kernel_size, self.stride, self.padding, self.dilation, id(self), 16)

    def forward(self, x):
        result = avepool2d(x, self.kernel_size, self.stride, self.padding, self.dilation)
        if self.input_shape is None:
            self.input_shape = x.shape
        if self.output_shape is None:
            self.output_shape = result.shape
        return result

class AvgPool3d(Module):
    '''Applies a 3D average pooling over an input signal composed of several input planes.\n
    Args:
        kernel_size (tuple of int): the size of the window to take a max over
        stride (tuple of int): the stride of the window. Default value is kernel_size
        padding (tuple of int): implicit zero padding to be added on all three sides
        dilation (tuple of int): a parameter that controls the stride of elements in the window

    Shape:
        - Input: [N,C,H,W,D]
        - Output: [N,C,H_out,W_out,D_out]

        H_out = (H+2*padding[0]-dilation[0]*(kernel_size[0]-1)-1)/stride[0] + 1
        W_out = (W+2*padding[1]-dilation[1]*(kernel_size[1]-1)-1)/stride[1] + 1
        D_out = (D+2*padding[2]-dilation[2]*(kernel_size[2]-1)-1)/stride[2] + 1
    '''
    def __init__(self, kernel_size=(2,2,2), stride=(2,2,2), padding=0, dilation=1): 
        super().__init__() 
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.padding = _triple(padding)
        self.dilation = _triple(dilation)

    def __repr__(self):
        return '{}({}, stride={}, padding={}, dilation={}) at 0x{:0{}X}'.format(self.__class__.__name__, self.kernel_size, self.stride, self.padding, self.dilation, id(self), 16)

    def forward(self, x):
        result = avepool3d(x, self.kernel_size, self.stride, self.padding, self.dilation)
        if self.input_shape is None:
            self.input_shape = x.shape
        if self.output_shape is None:
            self.output_shape = result.shape
        return result
    
class GlobalAvgPool1d(Module):
    ''' Applies a 1D globl average pooling over an input signal composed of several input planes.\n
    
    Shape:
        - Input: [N,C,W]
        - Output: [N,C,1]
    '''
    def __init__(self):
        super().__init__()
    
    def __repr__(self):
        return '{}() at 0x{:0{}X}'.format(self.__class__.__name__, id(self), 16)

    def forward(self, x):
        result = globalavepool1d(x)
        if self.input_shape is None:
            self.input_shape = x.shape
        if self.output_shape is None:
            self.output_shape = result.shape
        return result

class GlobalAvgPool2d(Module):
    ''' Applies a 2D globl average pooling over an input signal composed of several input planes.\n

    Shape:
        - Input: [N,C,H,W]
        - Output: [N,C,1,1]
    '''
    def __init__(self):
        super().__init__()
    
    def __repr__(self):
        return '{}() at 0x{:0{}X}'.format(self.__class__.__name__, id(self), 16)

    def forward(self, x):
        result = globalavepool2d(x)
        if self.input_shape is None:
            self.input_shape = x.shape
        if self.output_shape is None:
            self.output_shape = result.shape
        return result

class GlobalAvgPool3d(Module):
    ''' Applies a 3D globl average pooling over an input signal composed of several input planes.\n

    Shape:
        - Input: [N,C,H,W,D]
        - Output: [N,C,1,1,1]
    '''
    def __init__(self):
        super().__init__()
    
    def __repr__(self):
        return '{}() at 0x{:0{}X}'.format(self.__class__.__name__, id(self), 16)

    def forward(self, x):
        result = globalavepool3d(x)
        if self.input_shape is None:
            self.input_shape = x.shape
        if self.output_shape is None:
            self.output_shape = result.shape
        return result
    
class MaxUnpool1d(Module):
    '''Applies a 1D max unpooling over an input signal composed of several input planes.\n
    Args:
        kernel_size (int): the size of the window to take a max over
        stride (int): the stride of the window. Default value is kernel_size
        padding (int): implicit zero padding to be added on all three sides
        dilation (int): a parameter that controls the stride of elements in the window

    Shape:
        - Input: [N,C,W]
        - Output: [N,C,W_out]
    
        W_out = (W-1)*stride+dilation*(kernel_size-1)+1-2*padding
    '''
    def __init__(self, kernel_size=2, stride=2, padding=0, dilation=1): 
        super().__init__()  
        self.kernel_size = _single(kernel_size)
        self.stride = _single(stride)
        self.padding = _single(padding)
        self.dilation = _single(dilation)
    
    def __repr__(self):
        return '{}({}, stride={}, padding={}, dilation={}) at 0x{:0{}X}'.format(self.__class__.__name__, self.kernel_size, self.stride, self.padding, self.dilation, id(self), 16)

    def forward(self, x, indices):
        result = maxunpool1d(x, indices, self.kernel_size, self.stride, self.padding, self.dilation)
        if self.input_shape is None:
            self.input_shape = x.shape
        if self.output_shape is None:
            self.output_shape = result.shape
        return result

class MaxUnpool2d(Module):
    '''Applies a 2D max unpooling over an input signal composed of several input planes.\n
    Args:
        kernel_size (tuple of int): the size of the window to take a max over
        stride (tuple of int): the stride of the window. Default value is kernel_size
        padding (tuple of int): implicit zero padding to be added on all three sides
        dilation (tuple of int): a parameter that controls the stride of elements in the window

    Shape:
        - Input: [N,C,H,W]
        - Output: [N,C,H_out,W_out]
        
        H_out = (H-1)*stride[0]+dilation[0]*(kernel_size[0]-1)+1-2*padding[0]
        W_out = (W-1)*stride[1]+dilation[1]*(kernel_size[1]-1)+1-2*padding[1]
    '''
    def __init__(self, kernel_size=(2,2), stride=(2,2), padding=0, dilation=1): 
        super().__init__() 
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)

    def __repr__(self):
        return '{}({}, stride={}, padding={}, dilation={}) at 0x{:0{}X}'.format(self.__class__.__name__, self.kernel_size, self.stride, self.padding, self.dilation, id(self), 16)

    def forward(self, x, indices):
        result = maxunpool2d(x, indices, self.kernel_size, self.stride, self.padding, self.dilation)
        if self.input_shape is None:
            self.input_shape = x.shape
        if self.output_shape is None:
            self.output_shape = result.shape
        return result

class MaxUnpool3d(Module):
    '''Applies a 3D max unpooling over an input signal composed of several input planes.\n
    Args:
        kernel_size (tuple of int): the size of the window to take a max over
        stride (tuple of int): the stride of the window. Default value is kernel_size
        padding (tuple of int): implicit zero padding to be added on all three sides
        dilation (tuple of int): a parameter that controls the stride of elements in the window

    Shape:
        - Input: Input: [N,C,H,W,D]
        - Output: [N,C,H_out,W_out,D_out]
        
        H_out = (H-1)*stride[0]+dilation[0]*(kernel_size[0]-1)+1-2*padding[0]
        W_out = (W-1)*stride[1]+dilation[1]*(kernel_size[1]-1)+1-2*padding[1]
        D_out = (D-1)*stride[2]+dilation[2]*(kernel_size[2]-1)+1-2*padding[2]
    '''
    def __init__(self, kernel_size=(2,2,2), stride=(2,2,2), padding=0, dilation=1): 
        super().__init__() 
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.padding = _triple(padding)
        self.dilation = _triple(dilation)

    def __repr__(self):
        return '{}({}, stride={}, padding={}, dilation={}) at 0x{:0{}X}'.format(self.__class__.__name__, self.kernel_size, self.stride, self.padding, self.dilation, id(self), 16)

    def forward(self, x, indices):
        result= maxunpool3d(x, indices, self.kernel_size, self.stride, self.padding, self.dilation)
        if self.input_shape is None:
            self.input_shape = x.shape
        if self.output_shape is None:
            self.output_shape = result.shape
        return result
