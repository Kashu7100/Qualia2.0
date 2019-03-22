# -*- coding: utf-8 -*-
import math 
from .module import Module
from ...core import * 
from ...functions import batch_norm
from ...autograd import Tensor 

class BatchNorm1d(Module):
    '''Applies Batch Normalization over a 2D or 3D input.\n
    Args:
        num_features (int): C from an expected input of size [N,C] or [N,C,L]
        eps (float): a value added to the denominator for numerical stability. Default: 1e-5
        momentum (float): the value used for the running_mean and running_std computation. Default:0.1
        track_running_stats (bool): if True, this module tracks the running mean and std, else this module always uses batch statistics in both training and eval modes.
    Shape:
        - Input: [N,C] or [N,C,L]
        - Output: same as input
    '''
    def __init__(self, num_features, eps=1e-5, momentum=0.1, track_running_stats=True): 
        super().__init__()  
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.track_running_stats = track_running_stats
        self.num_params = 2*num_features
        self.weight = Tensor(np.random.uniform(0,1,(1,num_features)))
        self.bias = Tensor(np.zeros((1,num_features))) 
        if track_running_stats:
            self.mean = Tensor(np.zeros((1,num_features)), requires_grad=False)
            self.std = Tensor(np.zeros((1,num_features)), requires_grad=False)
    
    def __repr__(self):
        return '{}({}, eps={}, momentum={}, track_running_stats={}) at 0x{:0{}X}'.format(self.__class__.__name__, self.num_features, self.eps, self.momentum, self.track_running_stats, id(self), 16)

    def forward(self, x):
        assert x.ndim in [2,3]
        if x.ndim == 3 and self.weight.ndim != 3:
            self.weight = Tensor(np.random.uniform(0,1,(1,self.num_features,1)))
            self.bias = Tensor(np.zeros((1,self.num_features,1)))
            if self.track_running_stats:
                self.mean = Tensor(np.zeros((1,self.num_features,1)), requires_grad=False)
                self.std = Tensor(np.zeros((1,self.num_features,1)), requires_grad=False) 
        if self.training:
            axis = 0 if x.ndim == 2 else (0, 2)
            mean = np.mean(x.data, axis=axis, keepdims=True)
            std = np.std(x.data, axis=axis, keepdims=True)
            if self.track_running_stats:
                self.mean.data = self.momentum*mean + (1 - self.momentum)*self.mean.data
                self.std.data = self.momentum*std + (1 - self.momentum)*self.std.data
            result = batch_norm(x, mean, std, self.weight, self.bias, axis, self.eps)
        else:
            if self.track_running_stats:
                result = batch_norm(x, self.mean.data, self.std.data, self.weight, self.bias, axis, self.eps)
            else:
                axis = 0 if x.ndim == 2 else (0, 2)
                mean = np.mean(x.data, axis=axis, keepdims=True)
                std = np.std(x.data, axis=axis, keepdims=True)
                result = batch_norm(x, mean, std, self.weight, self.bias, axis, self.eps)
        if self.input_shape is None:
            self.input_shape = x.shape
        if self.output_shape is None:
            self.output_shape = result.shape
        return result

class BatchNorm2d(Module):
    '''Applies Batch Normalization over a 4D input\n
    Args:
        num_features (int): C from an expected input of size [N,C,H,W]
        eps (float): a value added to the denominator for numerical stability. Default: 1e-5
        momentum (float): the value used for the running_mean and running_std computation. Default:0.1
        track_running_stats (bool): if True, this module tracks the running mean and std, else this module always uses batch statistics in both training and eval modes.
    Shape:
        - Input: [N,C,H,W]
        - Output: same as input
    '''
    def __init__(self, num_features, eps=1e-5, momentum=0.1, track_running_stats=True): 
        super().__init__()  
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.track_running_stats = track_running_stats
        self.num_params = 2*num_features
        self.weight = Tensor(np.random.uniform(0,1,(1,num_features,1,1)))
        self.bias = Tensor(np.zeros((1,num_features,1,1))) 
        if track_running_stats:
            self.mean = Tensor(np.zeros((1,num_features,1,1)), requires_grad=False)
            self.std = Tensor(np.zeros((1,num_features,1,1)), requires_grad=False)
    
    def __repr__(self):
        return '{}({}, eps={}, momentum={}, track_running_stats={}) at 0x{:0{}X}'.format(self.__class__.__name__, self.num_features, self.eps, self.momentum, self.track_running_stats, id(self), 16)

    def forward(self, x):
        if self.training:
            axis = (0,2,3)
            mean = np.mean(x.data, axis=axis, keepdims=True)
            std = np.std(x.data, axis=axis, keepdims=True)
            if self.track_running_stats:
                self.mean.data = self.momentum*mean + (1 - self.momentum)*self.mean.data
                self.std.data = self.momentum*std + (1 - self.momentum)*self.std.data
            result = batch_norm(x, mean, std, self.weight, self.bias, axis, self.eps)
        else:
            if self.track_running_stats:
                result = batch_norm(x, self.mean.data, self.std.data, self.weight, self.bias, axis, self.eps)
            else:
                axis = (0,2,3)
                mean = np.mean(x.data, axis=axis, keepdims=True)
                std = np.std(x.data, axis=axis, keepdims=True)
                result = batch_norm(x, mean, std, self.weight, self.bias, axis, self.eps)
        if self.input_shape is None:
            self.input_shape = x.shape
        if self.output_shape is None:
            self.output_shape = result.shape
        return result

class BatchNorm3d(Module):
    '''Applies Batch Normalization over a 5D input\n
    Args:
        num_features (int): C from an expected input of size [N,C,D,H,W]
        eps (float): a value added to the denominator for numerical stability. Default: 1e-5
        momentum (float): the value used for the running_mean and running_std computation. Default:0.1
        track_running_stats (bool): if True, this module tracks the running mean and std, else this module always uses batch statistics in both training and eval modes.
    Shape:
        - Input: [N,C,D,H,W]
        - Output: same as input
    '''
    def __init__(self, num_features, eps=1e-5, momentum=0.1, track_running_stats=True): 
        super().__init__()  
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.track_running_stats = track_running_stats
        self.num_params = 2*num_features
        self.weight = Tensor(np.random.uniform(0,1,(1,num_features,1,1,1)))
        self.bias = Tensor(np.zeros((1,num_features,1,1,1))) 
        if track_running_stats:
            self.mean = Tensor(np.zeros((1,num_features,1,1,1)), requires_grad=False)
            self.std = Tensor(np.zeros((1,num_features,1,1,1)), requires_grad=False)
    
    def __repr__(self):
        return '{}({}, eps={}, momentum={}, track_running_stats={}) at 0x{:0{}X}'.format(self.__class__.__name__, self.num_features, self.eps, self.momentum, self.track_running_stats, id(self), 16)

    def forward(self, x):
        if self.training:
            axis = (0,2,3,4)
            mean = np.mean(x.data, axis=axis, keepdims=True)
            std = np.std(x.data, axis=axis, keepdims=True)
            if self.track_running_stats:
                self.mean.data = self.momentum*mean + (1 - self.momentum)*self.mean.data
                self.std.data = self.momentum*std + (1 - self.momentum)*self.std.data
            result = batch_norm(x, mean, std, self.weight, self.bias, axis, self.eps)
        else:
            if self.track_running_stats:
                result = batch_norm(x, self.mean.data, self.std.data, self.weight, self.bias, axis, self.eps)
            else:
                axis = (0,2,3,4)
                mean = np.mean(x.data, axis=axis, keepdims=True)
                std = np.std(x.data, axis=axis, keepdims=True)
                result = batch_norm(x, mean, std, self.weight, self.bias, axis, self.eps)
        if self.input_shape is None:
            self.input_shape = x.shape
        if self.output_shape is None:
            self.output_shape = result.shape
        return result