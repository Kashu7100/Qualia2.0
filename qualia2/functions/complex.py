# -*- coding: utf-8 -*- 
from ..core import *
from ..autograd import *

class CTanh(Function):
    '''
    Elementwise hyperbolic tangent function for complex neural networks
    '''
    @staticmethod
    def forward(a):
        real = np.tanh(a.data.real)
        imag = np.tanh(a.data.imag)
        result = Tensor(real+1j*imag) 
        result.set_creator(CTanh.prepare(result.shape, a, real=real, imag=imag))
        a.child.append(id(result.creator))
        return result

    def calc_grad(self, dx):
        real = dx.real*(1-np.square(self.kwargs['real']))
        imag = dx.imag*(1-np.square(self.kwargs['imag']))
        return real+1j*imag

ctanh = CTanh(None)

class CReLU(Function):
    @staticmethod
    def forward(a):
        mask_real = (a.data.real < 0) 
        tmp_real = a.data.real.copy() 
        tmp_real[mask_real] = 0
        mask_imag = (a.data.imag < 0) 
        tmp_imag = a.data.imag.copy() 
        tmp_imag[mask_imag] = 0
        result = Tensor(tmp_real+1j*tmp_imag) 
        result.set_creator(CReLU.prepare(result.shape, a, mask_real=mask_real, mask_imag=mask_imag))
        a.child.append(id(result.creator))
        return result
    
    def calc_grad(self, dx):
        dx = dx.copy()
        dx.real[self.kwargs['mask_real']] = 0
        dx.imag[self.kwargs['mask_imag']] = 0
        return dx
    
crelu = CReLU(None)

class CMSELoss(Function):
    '''MSE Loss for complex neural networks\n
    Args:
        input (Tensor): output of the network
        target (Tensor): label of the dataset
        reduce (bool): the losses are averaged or summed over observations for each minibatch depending on size_average. 
        size_average (bool): the losses are averaged over each loss element in the batch.

    Model:
        l_n = sum((y_n - x_n)^2/2)
    
    Shape:
        - Input: [N, C] 
        - Target: [N, C] 
        - Output: [1] by default 
                  [N] if not reduce
    '''
    @staticmethod
    def forward(input, target, reduce=True, size_average=True):
        tmp = np.sum(np.divide(np.power(np.abs(np.subtract(input.data, target.data)), 2), 2), axis=1)
        if reduce:
            if size_average:
                result = Tensor(np.mean(tmp,axis=0))
            else:
                result = Tensor(np.sum(tmp,axis=0))
        else:
            result = Tensor(tmp)
        result.set_creator(CMSELoss.prepare(result.shape, input, target))
        input.child.append(id(result.creator))
        target.child.append(id(result.creator))
        return result

    def calc_grad(self, dx):
        return np.subtract(self.var[0].data, self.var[1].data), np.subtract(self.var[1].data, self.var[0].data)

cmse_loss = CMSELoss(None)