# -*- coding: utf-8 -*- 
from .module import Module
from ...functions import *

class ReLU(Module):
    def forward(self, x):
        if self.input_shape is None:
            self.input_shape = x.shape
        if self.output_shape is None:
            self.output_shape = x.shape
        return relu(x)

class LeakyReLU(Module):
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

    def forward(self, x):
        if self.input_shape is None:
            self.input_shape = x.shape
        if self.output_shape is None:
            self.output_shape = x.shape
        return elu(x, self.alpha)

class Sigmoid(Module):
    def forward(self, x):
        if self.input_shape is None:
            self.input_shape = x.shape
        if self.output_shape is None:
            self.output_shape = x.shape
        return sigmoid(x)

class Tanh(Module):
    def forward(self, x):
        if self.input_shape is None:
            self.input_shape = x.shape
        if self.output_shape is None:
            self.output_shape = x.shape
        return tanh(x)

class SoftPuls(Module):
    def forward(self, x):
        if self.input_shape is None:
            self.input_shape = x.shape
        if self.output_shape is None:
            self.output_shape = x.shape
        return softplus(x)

class SoftSign(Module):
    def forward(self, x):
        if self.input_shape is None:
            self.input_shape = x.shape
        if self.output_shape is None:
            self.output_shape = x.shape
        return softsign(x)

class SoftMax(Module):
    def forward(self, x):
        if self.input_shape is None:
            self.input_shape = x.shape
        if self.output_shape is None:
            self.output_shape = x.shape
        return softmax(x)
    
class Flatten(Module):
    def forward(self, x):
        result = x.reshape(x.shape[0],-1)
        if self.input_shape is None:
            self.input_shape = x.shape
        if self.output_shape is None:
            self.output_shape = result.shape
        return result
