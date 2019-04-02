# -*- coding: utf-8 -*-
import math 
from .module import Module
from ...core import * 
from ...util import _single, _pair, _triple, _mul
from ...functions import conv1d, conv2d, conv3d
from ...autograd import Tensor 

class Conv1d(Module):
    ''''Applies a 1D convolution over an input signal composed of several input planes.\n 
    Args: 
        in_channels (int): Number of channels in the input image 
        out_channels (int): Number of channels produced by the convolution 
        kernel_size (int): Size of the convolving kernel 
        stride (int): Stride of the convolution. Default: 1 
        padding (int):  Zero-padding added to both sides of the input. Default: 0 
        dilation (int): Spacing between kernel elements. Default: 1 
        bias (bool):  adds a learnable bias to the output. Default: True 
     
    Shape: 
        - Input: [N, in_channels, W] 
        - Output: [N, out_channels, W_out] 
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, bias=True): 
        super().__init__()  
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _single(kernel_size)
        self.num_params += out_channels*in_channels*self.kernel_size
        self.kernel = Tensor(np.random.uniform(-math.sqrt(out_channels/self.num_params),math.sqrt(out_channels/self.num_params),(out_channels, in_channels, self.kernel_size)))
        #self.kernel = Tensor(0.01*np.random.randn(out_channels, in_channels, self.kernel_size)) 
        if bias: 
            self.bias = Tensor(np.random.uniform(-math.sqrt(out_channels/self.num_params),math.sqrt(out_channels/self.num_params),out_channels))
            #self.bias = Tensor(np.zeros((out_channels))) 
            self.num_params += out_channels
        else: 
            self.bias = None 
        self.stride = _single(stride)
        self.padding = _single(padding) 
        self.dilation = _single(dilation)
    
    def __repr__(self):
        return '{}({}, {}, {}, stride={}, padding={}, dilation={}, bias={}) at 0x{:0{}X}'.format(self.__class__.__name__, self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.dilation, str(self.bias is not None), id(self), 16)

    def forward(self, x):
        result = conv1d(x, self.kernel, self.bias, self.stride, self.padding, self.dilation)
        if self.input_shape is None:
            self.input_shape = x.shape
        if self.output_shape is None:
            self.output_shape = result.shape
        return result

class Conv2d(Module):
    '''Applies a 2D convolution over an input signal composed of several input planes.\n 
    Args: 
        in_channels (int): Number of channels in the input image 
        out_channels (int): Number of channels produced by the convolution 
        kernel_size (tuple of int): Size of the convolving kernel 
        stride (tuple of int): Stride of the convolution. Default: 1 
        padding (tuple of int):  Zero-padding added to both sides of the input. Default: 0 
        dilation (tuple of int): Spacing between kernel elements. Default: 1 
        bias (bool):  adds a learnable bias to the output. Default: True 
     
    Shape: 
        - Input: [N, in_channels, H, W] 
        - Output: [N, out_channels, H_out, W_out] 
 
        H_out = (H+2*padding[0]-dilation[0]*(kernel_size[0]-1)-1)/stride[0]+1 
        W_out = (W+2*padding[1]-dilation[1]*(kernel_size[1]-1)-1)/stride[1]+1 
    ''' 
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, bias=True): 
        super().__init__() 
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.num_params += out_channels*in_channels*_mul(*self.kernel_size)
        self.kernel = Tensor(np.random.uniform(-math.sqrt(out_channels/self.num_params),math.sqrt(out_channels/self.num_params),(out_channels, in_channels, *self.kernel_size))) 
        #self.kernel = Tensor(0.01*np.random.randn(out_channels, in_channels, *self.kernel_size)) 
        if bias: 
            self.bias = Tensor(np.random.uniform(-math.sqrt(out_channels/self.num_params),math.sqrt(out_channels/self.num_params),out_channels))
            #self.bias = Tensor(np.zeros((out_channels))) 
            self.num_params += out_channels
        else: 
            self.bias = None 
        self.stride = _pair(stride)
        self.padding = _pair(padding) 
        self.dilation = _pair(dilation) 
        
    def __repr__(self):
        return '{}({}, {}, {}, stride={}, padding={}, dilation={}, bias={}) at 0x{:0{}X}'.format(self.__class__.__name__, self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.dilation, str(self.bias is not None), id(self), 16)

    def forward(self, x):
        result = conv2d(x, self.kernel, self.bias, self.stride, self.padding, self.dilation)
        if self.input_shape is None:
            self.input_shape = x.shape
        if self.output_shape is None:
            self.output_shape = result.shape
        return result


class Conv3d(Module):
    '''Applies a 3D convolution over an input signal composed of several input planes.\n 
    Args: 
        in_channels (int): Number of channels in the input image 
        out_channels (int): Number of channels produced by the convolution 
        kernel_size (tuple of int): Size of the convolving kernel 
        stride (tuple of int): Stride of the convolution. Default: 1 
        padding (tuple of int):  Zero-padding added to both sides of the input. Default: 0 
        dilation (tuple of int): Spacing between kernel elements. Default: 1 
        bias (bool):  adds a learnable bias to the output. Default: True 
     
    Shape: 
        - Input: [N, in_channels, D, H, W] 
        - Output: [N, out_channels, D_out, H_out, W_out] 
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, bias=True): 
        super().__init__() 
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _triple(kernel_size)
        self.num_params += out_channels*in_channels*_mul(*self.kernel_size)
        self.kernel = Tensor(np.random.uniform(-math.sqrt(out_channels/self.num_params),math.sqrt(out_channels/self.num_params),(out_channels, in_channels, *self.kernel_size))) 
        #self.kernel = Tensor(0.01*np.random.randn(out_channels, in_channels, *self.kernel_size)) 
        if bias: 
            self.bias = Tensor(np.random.uniform(-math.sqrt(out_channels/self.num_params),math.sqrt(out_channels/self.num_params),out_channels))
            #self.bias = Tensor(np.zeros((out_channels))) 
            self.num_params += out_channels
        else: 
            self.bias = None 
        self.stride = _triple(stride)
        self.padding = _triple(padding) 
        self.dilation = _triple(dilation)

    def __repr__(self):
        return '{}({}, {}, {}, stride={}, padding={}, dilation={}, bias={}) at 0x{:0{}X}'.format(self.__class__.__name__, self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.dilation, str(self.bias is not None), id(self), 16)

    def forward(self, x):
        result = conv3d(x, self.kernel, self.bias, self.stride, self.padding, self.dilation)
        if self.input_shape is None:
            self.input_shape = x.shape
        if self.output_shape is None:
            self.output_shape = result.shape
        return result

class ConvTranspose2d(Module):
    '''Applies a 2D transposed convolution over an input signal composed of several input planes.\n 
    Args: 
        in_channels (int): Number of channels in the input image 
        out_channels (int): Number of channels produced by the convolution 
        kernel_size (tuple of int): Size of the convolving kernel 
        stride (tuple of int): Stride of the convolution. Default: 1 
        padding (tuple of int):  Zero-padding added to both sides of the input. Default: 1 
        output_padding (tuple of int): Zero-padding added to both sides of the output. Default: 0
        dilation (tuple of int): Spacing between kernel elements. Default: 1 
        bias (bool):  adds a learnable bias to the output. Default: True 
     
    Shape: 
        - Input: [N, in_channels, H, W] 
        - Output: [N, out_channels, H_out, W_out] 
 
        H_out = (H-1)*stride[0]-2*padding[0]+dilation[0]*(kernel_size[0]-1)+1+output_padding[0]
        W_out = (W-1)*stride[1]-2*padding[1]+dilation[1]*(kernel_size[1]-1)+1+output_padding[1]
    ''' 
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, output_padding=0, dilation=1, bias=True): 
        super().__init__() 
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.num_params += out_channels*in_channels*_mul(*self.kernel_size)
        self.kernel = Tensor(np.random.uniform(-math.sqrt(out_channels/self.num_params),math.sqrt(out_channels/self.num_params),(out_channels, in_channels, *self.kernel_size))) 
        if bias: 
            self.bias = Tensor(np.random.uniform(-math.sqrt(out_channels/self.num_params),math.sqrt(out_channels/self.num_params),out_channels))
            self.num_params += out_channels
        else: 
            self.bias = None 
        self.stride = _pair(stride)
        self.padding = _pair(padding) 
        self.output_padding = _pair(output_padding)
        self.dilation = _pair(dilation)

    def __repr__(self):
        return '{}({}, {}, {}, stride={}, padding={}, output_padding={}, dilation={}, bias={}) at 0x{:0{}X}'.format(self.__class__.__name__, self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.output_padding, self.dilation, str(self.bias is not None), id(self), 16)

    def forward(self, x):
        result = convtranspose2d(x, self.kernel, self.bias, self.stride, self.padding, self.output_padding, self.dilation)
        if self.input_shape is None:
            self.input_shape = x.shape
        if self.output_shape is None:
            self.output_shape = result.shape
        return result    
