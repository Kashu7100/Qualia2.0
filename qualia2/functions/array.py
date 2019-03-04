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

class ListConcat(Function):
    '''
    Concatenate list of Tensors 
    '''
    @staticmethod
    def forward(list):
        result = Tensor(np.concatenate([np.expand_dims(arr.data, axis=0) for arr in list], axis=0))
        result.set_creator(ListConcat.prepare(result.shape, *list))
        return result
    
    def calc_grad(self, dx):
        result = np.split(dx, len(self.var))
        return [np.squeeze(r, axis=0) for r in result]

listconcat = ListConcat(None)
