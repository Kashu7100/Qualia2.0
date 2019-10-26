# -*- coding: utf-8 -*- 
from ..core import *
from ..autograd import *

class Dropout(Function):
    @staticmethod
    def forward(x, p=0.5, training=True):
        '''
        During training, randomly zeroes some of the elements of the input tensor with probability p using samples from a Bernoulli distribution. 
        Each channel will be zeroed out independently on every forward call.
        Args:
            x (Tensor): Input tensor with any shepe
            p (float): probability that randomly zeroes some of the elements of the input tensor
            training (bool): True if the model is in training
        '''
        if training:
            np.random.seed()
            mask = (np.random.binomial(1,p,x.shape) == 1)
            tmp = x.data.copy()
            tmp[mask] = 0
            result = Tensor(tmp)
            result.set_creator(Dropout.prepare(result.shape, x, mask=mask))
            x.child.append(id(result.creator))
            return result
        else:
            return x        
    
    def calc_grad(self, dx):
        tmp = dx.copy()
        tmp[self.kwargs['mask']] = 0
        return tmp

dropout = Dropout(None)