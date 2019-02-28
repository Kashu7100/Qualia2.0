# -*- coding: utf-8 -*- 
from ..core import *
from ..autograd import *

class Reshape(Function):
    @staticmethod
    def forward(a, shape):
        result = Tensor(np.reshape(a.data, shape)) 
        result.set_creator(Reshape.prepare(result.shape, a))
        return result

    def calc_grad(self, dx):
        return np.reshape(dx, self.var[0].shape) 

reshape = Reshape(None)

class Transpose(Function):
    @staticmethod
    def forward(a, axes):
        result = Tensor(np.transpose(a.data, axes)) 
        result.set_creator(Transpose.prepare(result.shape, a, axes=axes))
        return result

    def calc_grad(self, dx):
        return np.transpose(dx, [self.kwargs['axes'].index(i) for i in range(len(self.kwargs['axes']))]) 

transpose = Transpose(None)
