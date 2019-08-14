# -*- coding: utf-8 -*- 
from ..core import *
from ..autograd import *

class CosineSimilarity(Function):
    @staticmethod
    def forward(a, b, dim=1, eps=1e-8):
        a_hat = a.data / (np.sqrt(np.sum(a.data**2))+eps)
        b_hat = b.data / (np.sqrt(np.sum(b.data**2))+eps)
        result = Tensor(np.dot(a_hat,b_hat))
        result.set_creator(CosineSimilarity.prepare(result.shape, a, b, a_hat=a_hat, b_hat=b_hat))
        return result
       
    def calc_grad(self, dx):
        da = np.dot(dx, self.kwargs['b_hat'].T)
        db = np.dot(self.kwargs['a_hat'].T, dx)
        return da, db

cosine_similarity = CosineSimilarity(None)