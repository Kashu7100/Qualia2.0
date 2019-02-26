# -*- coding: utf-8 -*- 
from ..core import *
from ..autograd import *

class Sin(Function):
    '''
    Elementwise sine function
    '''
    @staticmethod
    def forward(a):
        result = Tensor(np.sin(a.data)) 
        result.set_creator(Sin.prepare(result.shape, a))
        return result

    def calc_grad(self, dx):
        return np.multiply(dx, np.cos(self.var[0].data))

sin = Sin(None)

class Cos(Function):
    '''
    Elementwise cosine function
    '''
    @staticmethod
    def forward(a):
        result = Tensor(np.cos(a.data)) 
        result.set_creator(Cos.prepare(result.shape, a))
        return result

    def calc_grad(self, dx):
        return np.negative(np.multiply(dx, np.sin(self.var[0].data)))

cos = Cos(None)

class Tan(Function):
    '''
    Elementwise tangent function
    '''
    @staticmethod
    def forward(a):
        result = Tensor(np.tan(a.data)) 
        result.set_creator(Tan.prepare(result.shape, a))
        return result

    def calc_grad(self, dx):
        return np.divide(dx, np.power(np.cos(self.var[0].data),2))

tan = Tan(None)

class Arcsin(Function):
    '''
    Elementwise inverse-sine function
    '''
    @staticmethod
    def forward(a):
        result = Tensor(np.arcsin(a.data)) 
        result.set_creator(Arcsin.prepare(result.shape, a))
        return result

    def calc_grad(self, dx):
        return np.divide(dx, np.sqrt(np.subtract(1,np.power(self.var[0].data,2))))

arcsin = Arcsin(None)


class Arccos(Function):
    '''
    Elementwise inverse-cosine function
    '''
    @staticmethod
    def forward(a):
        result = Tensor(np.arccos(a.data)) 
        result.set_creator(Arccos.prepare(result.shape, a))
        return result

    def calc_grad(self, dx):
        return np.divide(-dx, np.sqrt(np.subtract(1,np.power(self.var[0].data,2))))

arccos = Arccos(None)

class Arctan(Function):
    '''
    Elementwise inverse-tangent function
    '''
    @staticmethod
    def forward(a):
        result = Tensor(np.arctan(a.data)) 
        result.set_creator(Arctan.prepare(result.shape, a))
        return result

    def calc_grad(self, dx):
        return np.divide(dx, np.add(1,np.power(self.var[0].data,2)))

arctan = Arctan(None)


class Sinh(Function):
    '''
    Elementwise hyperbolic sine function
    '''
    @staticmethod
    def forward(a):
        result = Tensor(np.sinh(a.data)) 
        result.set_creator(Sinh.prepare(result.shape, a))
        return result

    def calc_grad(self, dx):
        return np.multiply(dx, np.cosh(self.var[0].data))

sinh = Sinh(None)

class Cosh(Function):
    '''
    Elementwise hyperbolic cosine function
    '''
    @staticmethod
    def forward(a):
        result = Tensor(np.cosh(a.data)) 
        result.set_creator(Cosh.prepare(result.shape, a))
        return result

    def calc_grad(self, dx):
        return np.multiply(dx, np.sinh(self.var[0].data))

cosh = Cosh(None)

class Tanh(Function):
    '''
    Elementwise hyperbolic tangent function
    '''
    @staticmethod
    def forward(a):
        result = Tensor(np.tanh(a.data)) 
        result.set_creator(Tanh.prepare(result.shape, a, tmp=result.data))
        return result

    def calc_grad(self, dx):
        return np.multiply(dx, np.subtract(1, np.power(self.kwargs['tmp'], 2)))

tanh = Tanh(None)

class Arcsinh(Function):
    '''
    Elementwise inverse of hyperbolic sine function
    '''
    @staticmethod
    def forward(a):
        result = Tensor(np.arcsinh(a.data)) 
        result.set_creator(Arcsinh.prepare(result.shape, a))
        return result

    def calc_grad(self, dx):
        return np.divide(dx, np.sqrt(np.add(1, np.power(self.var[0].data,2))))
    
arcsinh = Arcsinh(None)

class Arccosh(Function):
    '''
    Elementwise inverse of hyperbolic cosine function
    '''
    @staticmethod
    def forward(a):
        result = Tensor(np.arccosh(a.data)) 
        result.set_creator(Arccosh.prepare(result.shape, a))
        return result

    def calc_grad(self, dx):
        return np.divide(dx, np.sqrt(np.subtract(np.power(self.var[0].data,2), 1)))

arccosh = Arccosh(None)

class Arctanh(Function):
    '''
    Elementwise inverse of hyperbolic tangent function
    '''
    @staticmethod
    def forward(a):
        result = Tensor(np.arctanh(a.data)) 
        result.set_creator(Arctanh.prepare(result.shape, a))
        return result

    def calc_grad(self, dx):
        return np.divide(dx, np.subtract(1, np.power(self.var[0].data,2)))

arctanh = Arctanh(None)