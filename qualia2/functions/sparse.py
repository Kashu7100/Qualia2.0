# -*- coding: utf-8 -*- 
from ..core import *
from ..autograd import *
from .linear_algebra import tensordot

class Embedding(Function):
    @staticmethod
    def forward(input, weight, vocab_size):
        '''
        Args:
            input (Tensor): Long Tensor containing indices into the embedding matrix
            weight (Tensor): The embedding matrix with number of rows equal to the maximum possible index + 1, and number of columns equal to the embedding size
        '''
        return tensordot(Embedding.to_one_hot(input, vocab_size), weight)

    @staticmethod
    def to_one_hot(input, vocab_size):
        corpus = input.data
        dim = corpus.shape
        one_hot = np.zeros((vocab_size,*dim))
        for c in range(vocab_size):
            one_hot[c][corpus==c] = 1
        return Tensor(np.swapaxes(one_hot,0,-1), requires_grad=False)

embedding = Embedding(None)
