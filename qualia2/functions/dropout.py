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
            return x*Tensor(mask, requires_grad=False)
        else:
            return x

dropout = Dropout(None)
