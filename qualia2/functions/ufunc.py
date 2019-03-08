# -*- coding: utf-8 -*- 
from ..core import *
from ..autograd import *
import functools 

add = Add(None)
subtract = Sub(None)
multiply = Mul(None)
divide = Div(None)
negative = Neg(None)
power = Pow(None)

class Exp(Function):
    @staticmethod
    def forward(a):
        result = Tensor(np.exp(a.data)) 
        result.set_creator(Exp.prepare(result.shape, a))
        return result

    def calc_grad(self, dx):
        return np.exp(np.multiply(self.var[0].data, dx))

exp = Exp(None)

class Log(Function):
    @staticmethod
    def forward(a):
        result = Tensor(np.log(a.data)) 
        result.set_creator(Log.prepare(result.shape, a))
        return result

    def calc_grad(self, dx):
        return np.divide(dx, self.var[0].data)

ln = Log(None)
log = Log(None)

class Log10(Function):
    @staticmethod
    def forward(a):
        result = Tensor(np.log10(a.data)) 
        result.set_creator(Log10.prepare(result.shape, a))
        return result

    def calc_grad(self, dx):
        return np.divide(dx, np.multiply(self.var[0].data, np.log(10)))

log10 = Log10(None)

class Sqrt(Function):
    @staticmethod
    def forward(a):
        result = Tensor(np.sqrt(a.data)) 
        result.set_creator(Sqrt.prepare(result.shape, a))
        return result

    def calc_grad(self, dx):
        return np.divide(dx, np.multiply(2, np.sqrt(self.var[0].data)))

sqrt = Sqrt(None)

class Cbrt(Function):
    @staticmethod
    def forward(a):
        result = Tensor(np.cbrt(a.data)) 
        result.set_creator(Cbrt.prepare(result.shape, a))
        return result
    
    def calc_grad(self, dx):
        return np.divide(dx, np.multiply(3, np.square(np.cbrt(self.var[0].data))))

cbrt = Cbrt(None)

class Mean(Function):
    @staticmethod
    def forward(a, axis=1):
        result = Tensor(np.mean(a.data, axis=axis)) 
        result.set_creator(Mean.prepare(result.shape, a, axis=axis))
        return result

    def calc_grad(self, dx):
        if type(self.kwargs['axis']) is not tuple: 
            self.kwargs['axis'] = (self.kwargs['axis'],)  
        shape = list(self.var[0].shape) 
        for i in self.kwargs['axis']: 
            dx = np.expand_dims(dx, axis=i) 
        reps = [shape[i] if i in self.kwargs['axis'] else 1 for i in range(len(shape))]
        return np.divide(np.tile(dx, reps), functools.reduce(lambda a,b : a+b, [i for i in reps if i>1]))

mean = Mean(None)

class Sum(Function):
    @staticmethod
    def forward(a, axis=1):
        result = Tensor(np.sum(a.data, axis=axis)) 
        result.set_creator(Sum.prepare(result.shape, a, axis=axis))
        return result

    def calc_grad(self, dx):
        if type(self.kwargs['axis']) is not tuple: 
            self.kwargs['axis'] = (self.kwargs['axis'],)  
        shape = list(self.var[0].shape) 
        for i in self.kwargs['axis']: 
            dx = np.expand_dims(dx, axis=i) 
        return np.tile(dx,[shape[i] if i in self.kwargs['axis'] else 1 for i in range(len(shape))])

sum = Sum(None)