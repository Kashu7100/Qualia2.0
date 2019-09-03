# -*- coding: utf-8 -*- 
from ..core import *
from ..autograd import *

class Identity(Function):
    '''
    Identity function
    '''
    @staticmethod
    def forward(a):
        return a

identity = Identity(None)

class ReLU(Function):
    '''
    Rectified linear unit
    '''
    @staticmethod
    def forward(a):
        result = Tensor(np.maximum(0,a.data)) 
        result.set_creator(ReLU.prepare(result.shape, a, mask=(a.data < 0)))
        return result

    def calc_grad(self, dx):
        result = np.copy(dx)
        result[self.kwargs['mask']] = 0
        return result

relu = ReLU(None)

class BReLU(Function):
    ''' 
    Bipolar rectified linear unit 
    '''
    @staticmethod
    def forward(a):
        result = Tensor(np.minimum(0,a.data)) 
        result.set_creator(BReLU.prepare(result.shape, a, mask=(a.data > 0)))
        return result

    def calc_grad(self, dx):
        result = np.copy(dx)
        result[self.kwargs['mask']] = 0
        return result

brelu = BReLU(None)

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
        return result
    
    def calc_grad(self, dx):
        dx = dx.copy()
        dx.real[self.kwargs['mask_real']] = 0
        dx.imag[self.kwargs['mask_imag']] = 0
        return dx
    
crelu = CReLU(None)
    
class LeakyReLU(Function):
    '''
    Leaky rectified linear unit 
    '''
    @staticmethod
    def forward(a):
        mask = (a.data < 0) 
        tmp = a.data.copy() 
        tmp[mask] = np.multiply(0.01,tmp[mask]) 
        result = Tensor(tmp) 
        result.set_creator(LeakyReLU.prepare(result.shape, a, mask=mask))
        return result

    def calc_grad(self, dx):
        dx = dx.copy()
        dx[self.kwargs['mask']] = np.multiply(0.01, dx[self.kwargs['mask']])
        return dx

leakyrelu = LeakyReLU(None)

class ELU(Function):
    '''
    Exponential linear unit
    '''
    @staticmethod
    def forward(a, k):
        mask = (a.data < 0) 
        tmp = a.data.copy() 
        tmp[mask] = np.multiply(k, (np.exp(tmp[mask])-1)) 
        result = Tensor(tmp) 
        result.set_creator(ELU.prepare(result.shape, a, tmp=tmp, mask=mask, const=k))
        return result

    def calc_grad(self, dx):
        dx = dx.copy()
        dx[self.kwargs['mask']] = np.add(np.multiply(dx[self.kwargs['mask']], self.kwargs['tmp'][self.kwargs['mask']]), np.multiply(dx[self.kwargs['mask']], self.kwargs['const']))
        return dx

elu = ELU(None)

class Sigmoid(Function):
    '''
    Sigmoid function
    '''
    @staticmethod
    def forward(a):
        result = Tensor(np.divide(1, np.add(1, np.exp(np.negative(a.data))))) 
        result.set_creator(Sigmoid.prepare(result.shape, a, tmp=result.data))
        return result

    def calc_grad(self, dx):
        return np.multiply(dx, np.multiply(self.kwargs['tmp'], np.subtract(1, self.kwargs['tmp'])))

sigmoid = Sigmoid(None)
logistic = Sigmoid(None)

class SoftPlus(Function):
    '''
    SoftPlus function
    '''
    @staticmethod
    def forward(a):
        result = Tensor(np.log(np.add(1, np.exp(a.data)))) 
        result.set_creator(SoftPlus.prepare(result.shape, a))
        return result

    def calc_grad(self, dx):
        return np.divide(dx, np.add(1, np.exp(np.negative(self.var[0].data))))

softplus = SoftPlus(None)

class SoftSign(Function):
    '''
    SoftSign function
    '''
    @staticmethod
    def forward(a):
        result = Tensor(np.divide(a.data, np.add(1, np.absolute(a.data))))
        result.set_creator(SoftSign.prepare(result.shape, a))
        return result

    def calc_grad(self, dx):
        return np.divide(dx, np.square(np.add(1, np.absolute(self.var[0].data))))

softsign = SoftSign(None)
elliotsig = SoftSign(None)

class SoftMax(Function):
    '''
    SoftMax function
    '''
    @staticmethod
    def forward(a):
        assert a.ndim == 2
        const = np.max(a.data, axis=1, keepdims=True)
        exp = np.exp(np.subtract(a.data, const))
        result = Tensor(np.divide(exp, np.sum(exp, axis=1, keepdims=True)))
        result.set_creator(SoftMax.prepare(result.shape, a, tmp=result.data))
        return result

    def calc_grad(self, dx):
        result = self.kwargs['tmp']
        tmp = dx*np.multiply(self.kwargs['tmp'], np.subtract(1, self.kwargs['tmp']))
        result[dx!=0] = tmp[dx!=0]
        return result

softmax = SoftMax(None)

class Tanhshrink(Function):
    '''
    Elementwise x - Tanh(x) function
    '''
    @staticmethod
    def forward(a):
        tmp = np.tanh(a.data)
        result = Tensor(a.data-tmp) 
        result.set_creator(Tanhshrink.prepare(result.shape, a, tmp=tmp))
        return result

    def calc_grad(self, dx):
        return np.subtract(1, np.multiply(dx, np.subtract(1, np.square(self.kwargs['tmp']))))

tanhshrink = Tanhshrink(None)

class Gaussian(Function):
    '''
    Elementwise Gaussian function
    '''
    @staticmethod
    def forward(a):
        result = Tensor(np.exp(-np.square(a.data)))
        result.set_creator(Gaussian.prepare(result.shape, a, tmp=result.data))
        return result
    
    def calc_grad(self, dx):
        return -2*dx*self.var[0].data*self.kwargs['tmp']

gaussian = Gaussian(None)
