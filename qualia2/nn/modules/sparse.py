# -*- coding: utf-8 -*-
from .module import Module
from ...core import * 
from ...functions import embedding
from ...autograd import Tensor 
import math

class Embedding(Module):
    '''A simple lookup table that stores embeddings of a fixed dictionary and size.
    Args:
        num_embeddings (int): size of the dictionary of embeddings
        embedding_dim (int): the size of each embedding vector
    '''
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_params += num_embeddings*embedding_dim
        self.weight = Tensor(np.random.normal(0, math.sqrt(1/num_embeddings),(num_embeddings, embedding_dim)))

    def __repr__(self):
        return '{}({}, {}) at 0x{:0{}X}'.format(self.__class__.__name__, self.num_embeddings, self.embedding_dim, id(self), 16)

    def forward(self, x):
        result = embedding(x, self.weight, self.num_embeddings)
        if self.input_shape is None:
            if isinstance(x, int):
                self.input_shape = (1,)
            else:
                self.input_shape = x.shape
        if self.output_shape is None:
            self.output_shape = result.shape
        return result
