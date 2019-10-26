# -*- coding: utf-8 -*- 
from ..core import *
from ..autograd import *

class Upsample(Function):
    @staticmethod
    def forward(a, scale_factor=2, mode='nearest'):
        result = Tensor(a.data.repeat(scale_factor, axis=2).repeat(scale_factor, axis=3)) 
        result.set_creator(Upsample.prepare(result.shape, a, scale_factor=scale_factor))
        a.child.append(id(result.creator))
        return result

    def calc_grad(self, dx):
        batch, channel, height, width = dx.shape
        sf = self.kwargs['scale_factor']
        th, tw = height//sf, width//sf
        return dx.reshape(*dx.shape[:2],th,sf,tw,sf).mean(axis=(3,5))

upsample = Upsample(None)