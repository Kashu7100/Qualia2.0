# -*- coding: utf-8 -*- 
from ..core import *
from ..autograd import *

class Embedding(Function):
    @staticmethod
    def forward(input, weight, vocab_size):
        '''
        Args:
            input (Tensor): Long Tensor containing indices into the embedding matrix
            weight (Tensor): The embedding matrix with number of rows equal to the maximum possible index + 1, and number of columns equal to the embedding size
            vocab_size (int): 
        '''
        if isinstance(input, Tensor):
            result = Tensor(weight[input.data.astype('int64')].data)
            result.set_creator(Embedding.prepare(result.shape, weight, idx=input.data))
        elif isinstance(input, np.ndarray):
            result = Tensor(weight[input.astype('int64')].data)
            result.set_creator(Embedding.prepare(result.shape, weight, idx=input))
        else:
            raise ValueError
        return result

    def calc_grad(self, dx):
        dw = np.zeros_like(self.var[0].data)
        if gpu:
            np.scatter_add(dw, self.kwargs['idx'].astype('int64'), dx)
        else:
            np.add.at(dw, self.kwargs['idx'].astype('int64'), dx)
        return dw

embedding = Embedding(None)
