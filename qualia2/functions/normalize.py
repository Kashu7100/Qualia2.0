# -*- coding: utf-8 -*- 
from ..core import *
from ..autograd import *

class BatchNorm(Function):
    @staticmethod
    def forward(x, mean, std, weight, bias, axis, eps=1e-5):
        '''Applies Batch Normalization for each channel across a batch of data.\n
        Args:
            x (Tensor): input tensor.
            mean (ndarray): running mean of the input tensor.
            std (ndarray): running std of the input tensor.
            weight (Tensor): weight to apply.
            bias (Tensor): bias to apply. 
            axis (list): axis indicates the all the axis in the input except C dimention 
            eps (float): a value added to the denominator for numerical stability.
        Shape:
            - Input: [N,C,*]
            - Output: [N,C,*]
        '''
        tmp = np.divide(np.subtract(x.data, mean), np.add(std, eps))
        result = Tensor(np.add(np.multiply(tmp, weight.data), bias.data))
        result.set_creator(BatchNorm.prepare(result.shape, x, weight, bias, mean=mean, std=std, eps=eps, tmp=tmp, axis=axis))
        x.child.append(id(result.creator))
        weight.child.append(id(result.creator))
        bias.child.append(id(result.creator))
        return result

    def calc_grad(self, dx):
        db = np.sum(dx, axis=self.kwargs['axis'], keepdims=True)
        dw = np.sum(np.multiply(self.kwargs['tmp'], dx), axis=self.kwargs['axis'], keepdims=True)
        tmp1 = np.multiply(dx, self.var[1].data)
        tmp2 = np.subtract(self.var[0].data, self.kwargs['mean'])
        tmp3 = np.mean(np.multiply(tmp2, tmp1), axis=self.kwargs['axis'], keepdims=True)
        tmp4 = np.add(self.kwargs['std'], self.kwargs['eps'])
        tmp5 = np.divide(np.subtract(tmp1, np.divide(np.multiply(tmp2,tmp3), np.square(tmp4))), tmp4)
        result = np.subtract(tmp5, np.mean(tmp5, axis=self.kwargs['axis'], keepdims=True))
        return result, dw, db

batch_norm = BatchNorm(None)