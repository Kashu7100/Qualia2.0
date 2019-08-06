# -*- coding: utf-8 -*- 
from .module import Module
from ...functions import *

class ReLU(Module):
    def __repr__(self):
        return '{}() at 0x{:0{}X}'.format(self.__class__.__name__, id(self), 16)

    def forward(self, x):
        if self.input_shape is None:
            self.input_shape = x.shape
        if self.output_shape is None:
            self.output_shape = x.shape
        return relu(x)

class LeakyReLU(Module):
    def __repr__(self):
        return '{}() at 0x{:0{}X}'.format(self.__class__.__name__, id(self), 16)

    def forward(self, x):
        if self.input_shape is None:
            self.input_shape = x.shape
        if self.output_shape is None:
            self.output_shape = x.shape
        return leakyrelu(x)

class ELU(Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
    
    def __repr__(self):
        return '{}() at 0x{:0{}X}'.format(self.__class__.__name__, id(self), 16)

    def forward(self, x):
        if self.input_shape is None:
            self.input_shape = x.shape
        if self.output_shape is None:
            self.output_shape = x.shape
        return elu(x, self.alpha)

class Sigmoid(Module):
    def __repr__(self):
        return '{}() at 0x{:0{}X}'.format(self.__class__.__name__, id(self), 16)

    def forward(self, x):
        if self.input_shape is None:
            self.input_shape = x.shape
        if self.output_shape is None:
            self.output_shape = x.shape
        return sigmoid(x)

class Tanh(Module):
    def __repr__(self):
        return '{}() at 0x{:0{}X}'.format(self.__class__.__name__, id(self), 16)

    def forward(self, x):
        if self.input_shape is None:
            self.input_shape = x.shape
        if self.output_shape is None:
            self.output_shape = x.shape
        return tanh(x)

class SoftPuls(Module):
    def __repr__(self):
        return '{}() at 0x{:0{}X}'.format(self.__class__.__name__, id(self), 16)

    def forward(self, x):
        if self.input_shape is None:
            self.input_shape = x.shape
        if self.output_shape is None:
            self.output_shape = x.shape
        return softplus(x)

class SoftSign(Module):
    def __repr__(self):
        return '{}() at 0x{:0{}X}'.format(self.__class__.__name__, id(self), 16)

    def forward(self, x):
        if self.input_shape is None:
            self.input_shape = x.shape
        if self.output_shape is None:
            self.output_shape = x.shape
        return softsign(x)

class SoftMax(Module):
    def __repr__(self):
        return '{}() at 0x{:0{}X}'.format(self.__class__.__name__, id(self), 16)

    def forward(self, x):
        if self.input_shape is None:
            self.input_shape = x.shape
        if self.output_shape is None:
            self.output_shape = x.shape
        return softmax(x)

class Flatten(Module):
    def __repr__(self):
        return '{}() at 0x{:0{}X}'.format(self.__class__.__name__, id(self), 16)
        
    def forward(self, x):
        result = x.reshape(x.shape[0],-1)
        if self.input_shape is None:
            self.input_shape = x.shape
        if self.output_shape is None:
            self.output_shape = result.shape
        return result
