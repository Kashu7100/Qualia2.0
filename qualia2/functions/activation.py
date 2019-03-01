# -*- coding: utf-8 -*- 
from ..core import *
from ..autograd import *

class ReLU(Function):
    @staticmethod
    def forward(a):
        mask = (a.data <= 0) 
        tmp = a.data.copy() 
        tmp[mask] = 0
        result = Tensor(tmp) 
        result.set_creator(ReLU.prepare(result.shape, a, mask=mask))
        return result

    def calc_grad(self, dx):
        dx[self.kwargs['mask']] = 0
        return dx

relu = ReLU(None)

class LeakyReLU(Function):
    @staticmethod
    def forward(a):
        mask = (a.data <= 0) 
        tmp = a.data.copy() 
        tmp[mask] = np.multiply(0.01,tmp[mask]) 
        result = Tensor(tmp) 
        result.set_creator(LeakyReLU.prepare(result.shape, a, mask=mask))
        return result

    def calc_grad(self, dx):
        dx[self.kwargs['mask']] = np.multiply(0.01, dx[self.kwargs['mask']])
        return dx

leakyrelu = LeakyReLU(None)

class ELU(Function):
    @staticmethod
    def forward(a, k):
        mask = (a.data <= 0) 
        tmp = a.data.copy() 
        tmp[mask] = np.multiply(k, (np.exp(tmp[mask])-1)) 
        result = Tensor(tmp) 
        result.set_creator(ELU.prepare(result.shape, a, tmp=tmp, mask=mask, const=k))
        return result

    def calc_grad(self, dx):
        dx[self.kwargs['mask']] = np.add(np.multiply(dx[self.kwargs['mask']], self.kwargs['tmp'][self.kwargs['mask']]), np.multiply(dx[self.kwargs['mask']], self.kwargs['const']))
        return dx

elu = ELU(None)

class Sigmoid(Function):
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
    @staticmethod
    def forward(a):
        result = Tensor(np.log(np.add(1, np.exp(a.data)))) 
        result.set_creator(SoftPlus.prepare(result.shape, a))
        return result

    def calc_grad(self, dx):
        return np.divide(dx, np.add(1, np.exp(np.negative(self.var[0].data))))

softplus = SoftPlus(None)

class SoftSign(Function):
    @staticmethod
    def forward(a):
        result = Tensor(np.divide(a.data, np.add(1, np.absolute(a.data))))
        result.set_creator(SoftSign.prepare(result.shape, a))
        return result

    def calc_grad(self, dx):
        return np.divide(dx, np.square(np.add(1, np.absolute(self.var[0].data))))

softsign = SoftSign(None)
elliotsig = SoftSign(None)