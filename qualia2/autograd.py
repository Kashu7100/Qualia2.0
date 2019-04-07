# -*- coding: utf-8 -*- 
from .core import *
from functools import reduce

class Tensor(object):
    '''Wrapper class to execute automatic differentiation\n 
    Args: 
        data (ndarray|int|float): tensor to compute the automatic differentiation 
        requires_grad (bool): Whether to store grads. If False is set, grad of the Tensor will be zeros. 
     
    Attributes: 
        data (ndarray): Stores data of the Tensor 
        grad (ndarray): Stores gradients of the Tensor  
        creator (Function obj): Stores the creator of the Tensor, which will be called at the backpropagation. 
        requires_grad (bool): Whether to store grads. If False is set, grad of the Tensor will be zeros. 
        shape (tuple): Stores the shape of Tensor's data 
        ndim (int): Stores the number of Tensor's data dimentions  
     
    Examples:: 
        The following example will compute the Sum of Squared Error 
        >>> # Create Tensor objects 
        >>> x = qualia2.array([5])
        >>> # Write an equation 
        >>> y = x**2 - 2*x + 1
        >>> print(y)
        >>> # Calclate gradiant 
        >>> y.backward()
        >>> # Print gradient 
        >>> print(x.grad)
    ''' 
    def __init__(self, data, requires_grad=True):
        super().__setattr__('hook', None) 
        if type(data) is not np.ndarray: 
            if type(data) is list:
                self.data = np.array(data)
            else: 
                self.data = np.array([data], dtype=dtype)
        else:
            self.data = data.astype(dtype)
        self.grad = None
        self.creator = None
        self.requires_grad = requires_grad

    def backward(self, *args):
        if not bool(args):
            args = [np.ones_like(self.data, dtype=dtype)]     
        if self.creator is None:
            self.grad = args[0]
        else:
            self.creator.backward(*args) 

    def set_creator(self, obj): 
        self.creator = obj     
    
    def handle_const(self, obj):
        if type(obj) is not Tensor:
            return Tensor(obj, requires_grad=False)
        return obj
    
    def reshape(self, *args):
        result = Tensor(np.reshape(self.data, args)) 
        result.set_creator(Reshape.prepare(result.shape, self))
        return result
    
    def gather(self, dim, idx):
        return Gather.forward(self, dim, idx)
    
    def detach(self):
        '''Returns a new Tensor, detached from the current graph.
        '''
        return Tensor(self.data, requires_grad=False)
    
    def clamp(self, low, high):
        return Clamp.forward(self, low, high)
    
    def register_hook(self, hook):
        self.hook = hook
    
    def __str__(self):
        return f'{self.data} shape={self.shape}'
    
    def __repr__(self):
        return '{}({}, requires_grad={}) at 0x{:0{}X}'.format(self.__class__.__name__, self.data, self.requires_grad, id(self), 16)

    def __setattr__(self, key, value):   
        super().__setattr__(key, value)
        if key == 'data':
            super().__setattr__('shape', self.data.shape)
            super().__setattr__('ndim', self.data.ndim) 
        if self.hook is not None:
            if key == 'grad':
                super().__setattr__('grad', self.hook(value))

    def __getitem__(self, slice):
        return Slice.forward(self, slice)

    def __len__(self): 
        return self.ndim

    def __add__(self, other): 
        other = self.handle_const(other)
        return Add.forward(self, other)
    
    def __radd__(self, other): 
        other = self.handle_const(other)
        return Add.forward(self, other)

    def __sub__(self, other): 
        other = self.handle_const(other)
        return Sub.forward(self, other) 
     
    def __rsub__(self, other): 
        other = self.handle_const(other)
        return Sub.forward(other, self) 
 
    def __mul__(self, other): 
        other = self.handle_const(other)
        return Mul.forward(self, other) 
 
    def __rmul__(self, other): 
        other = self.handle_const(other) 
        return Mul.forward(self, other)  
     
    def __matmul__(self, other):
        return Matmul.forward(self, other) 
     
    def __neg__(self): 
        return Neg.forward(self) 
 
    def __truediv__(self, other): 
        other = self.handle_const(other)
        return Div.forward(self, other)
 
    def __rtruediv__(self, other): 
        other = self.handle_const(other)
        return Div.forward(other, self)
 
    def __pow__(self, other): 
        other = self.handle_const(other)
        return Pow.forward(self, other)
     
    def __rpow__(self, other): 
        raise Exception('__rpow__ is not defined.')

class Function(object):
    '''
    All function should inherit this class. 

    Attributes:
        output_shape (tuple of int): output shape of a function
        var (tuple of Tensor): Tensor(s) that was feeded
    '''
    def __init__(self, output_shape, *args, **kwargs):
        self.output_shape = output_shape
        self.var = args
        self.kwargs = kwargs
        
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    @classmethod
    def prepare(cls, output_shape, *args, **kwargs):
        return cls(output_shape, *args, **kwargs)

    @staticmethod
    def forward(*args, **kwargs):
        raise NotImplementedError
    
    def calc_grad(self, *args):
        raise NotImplementedError

    @staticmethod
    def handle_broadcast(arg, trg):
        if arg.shape != trg.shape:
            if arg.ndim == trg.ndim:
                axis = [i for i in range(arg.ndim) if arg.shape[i] != trg.shape[i]]
                arg = np.sum(arg, axis=tuple(axis))
                return np.reshape(arg, trg.shape)
            elif arg.ndim > trg.ndim:
                assert trg.ndim == 1
                tmp = [1 for _ in range(len(arg.shape))]
                for i, s in enumerate(reversed(arg.shape)):
                    if s == trg.shape[0]:
                        tmp[len(tmp)-1-i] = s
                        break
                axis = [i for i in range(arg.ndim) if tmp[i] != arg.shape[i]]
                arg = np.sum(arg, axis=tuple(axis))
                return np.reshape(arg, trg.shape)
            else:
                raise ValueError
        return arg    

    def backward(self, *args):
        grads = self.calc_grad(*args)
        if type(grads) is list:
            grads = tuple(grads)
        if type(grads) is not tuple:
            grads = (grads,)
        for dx, var in zip(grads, self.var):
            if not var.requires_grad:
                continue
            if var.grad is None:
                var.grad = dx
            else:
                var.grad += dx
        for var in self.var:
            if var.creator is not None:
                var.backward(var.grad)

class Slice(Function):
    @staticmethod
    def forward(a, slice):
        result = Tensor(a.data[slice]) 
        result.set_creator(Slice.prepare(result.shape, a, slice=slice)) 
        return result
    
    def calc_grad(self, dx):
        result = np.zeros_like(self.var[0].data)
        result[self.kwargs['slice']] = dx
        return result

class Reshape(Function):
    @staticmethod
    def forward(a, shape):
        result = Tensor(np.reshape(a.data, shape)) 
        result.set_creator(Reshape.prepare(result.shape, a))
        return result

    def calc_grad(self, dx):
        return np.reshape(dx, self.var[0].shape)

class Gather(Function):
    '''
    Gathers values along an axis specified by dim.
    '''
    @staticmethod
    def forward(a, dim, idx):
        input_valid_dim = a.shape[:dim] + a.shape[dim+1:]
        idx_valid_dim = idx.shape[:dim] + idx.shape[dim+1:]
        if input_valid_dim != idx_valid_dim:
            raise ValueError('[*] All dimensions of index and input should be the same except for dimension dim={}, got: {} and {}.'.format(str(dim), str(a.shape), str(idx.shape)))
        gathered = np.choose(np.swapaxes(idx, 0, dim), np.swapaxes(a.data, 0, dim))
        result = Tensor(np.swapaxes(gathered, 0, dim))
        result.set_creator(Gather.prepare(result.shape, a, dim=dim, idx=idx))
        return result
    
    def calc_grad(self, dx):
        result = np.zeros_like(self.var[0].data)
        def make_slice(arr, dim, i):
            slc = [slice(None)] * arr.ndim
            slc[dim] = i
            return slc
        idx_xsection_shape = self.kwargs['idx'].shape[:self.kwargs['dim']] + self.kwargs['idx'].shape[self.kwargs['dim']+1:]

        idx = [[np.indices(idx_xsection_shape).reshape(self.kwargs['idx'].ndim-1, -1), self.kwargs['idx'][make_slice(self.kwargs['idx'], self.kwargs['dim'], i)].reshape(1, -1)] for i in range(self.kwargs['idx'].shape[self.kwargs['dim']])]
        idx = list(np.concatenate(tuple(idx[0]), axis=0))
        idx.insert(self.kwargs['dim'], idx.pop())

        if not np.isscalar(dx):
            src_xsection_shape = dx.shape[:self.kwargs['dim']] + dx.shape[self.kwargs['dim'] + 1:]
            src_idx = list(idx)
            src_idx.pop(self.kwargs['dim'])
            src_idx.insert(self.kwargs['dim'], np.repeat(np.arange(self.kwargs['idx'].shape[self.kwargs['dim']]), reduce(lambda a, b: a*b, idx_xsection_shape)))
            result[idx] = dx[src_idx]
        else:
            result[idx] = dx
        return result

class Clamp(Function):
    @staticmethod
    def forward(x, low, high):
        result = Tensor(np.clip(x.data, low, high))
        result.set_creator(Clamp.prepare(result.shape, x))
        return result
    
    def calc_grad(self, dx):
        return dx
    
class Neg(Function):
    '''
    Takes numerical negative elementwise.
    '''
    @staticmethod
    def forward(a):
        result = Tensor(np.negative(a.data)) 
        result.set_creator(Neg.prepare(result.shape, a)) 
        return result
    
    def calc_grad(self, dx):
        return np.negative(dx)

class Add(Function):
    '''
    Adds two arrays elementwise.
    '''
    @staticmethod
    def forward(a, b):
        result = Tensor(np.add(a.data, b.data)) 
        result.set_creator(Add.prepare(result.shape, a, b))
        return result
    
    def calc_grad(self, dx):
        return Add.handle_broadcast(dx, self.var[0]), Add.handle_broadcast(dx, self.var[1])
    
class Sub(Function):
    '''
    Subtracts arguments elementwise.
    '''
    @staticmethod
    def forward(a, b):
        result = Tensor(np.subtract(a.data, b.data)) 
        result.set_creator(Sub.prepare(result.shape, a, b))
        return result

    def calc_grad(self, dx):
        return Sub.handle_broadcast(dx, self.var[0]), np.negative(Sub.handle_broadcast(dx, self.var[1]))

class Mul(Function):
    '''
    Multiplies two arrays elementwise.
    '''
    @staticmethod
    def forward(a, b):
        result = Tensor(np.multiply(a.data, b.data)) 
        result.set_creator(Mul.prepare(result.shape, a, b))
        return result

    def calc_grad(self, dx):
        return np.multiply(self.var[1].data, dx), np.multiply(self.var[0].data, dx)
    
class Pow(Function):
    '''
    Computes x1 ** x2 elementwise.
    '''
    @staticmethod
    def forward(a, b):
        result = Tensor(np.power(a.data, b.data)) 
        result.set_creator(Pow.prepare(result.shape, a, b))
        return result

    def calc_grad(self, dx):
        return np.multiply(self.var[1].data, np.multiply(np.power(self.var[0].data, np.subtract(self.var[1].data, np.array([1]))), dx)), None

class Div(Function):
    '''
    Elementwise true division
    '''
    @staticmethod
    def forward(a, b):
        result = Tensor(np.divide(a.data, b.data)) 
        result.set_creator(Div.prepare(result.shape, a, b))
        return result

    def calc_grad(self, dx):
        return np.divide(dx, self.var[1].data), np.negative(np.multiply(dx, np.divide(self.var[0].data, np.power(self.var[1].data, 2))))

class Matmul(Function):
    @staticmethod
    def forward(a, b):
        result = Tensor(np.matmul(a.data, b.data)) 
        result.set_creator(Matmul.prepare(result.shape, a, b))
        return result

    def calc_grad(self, dx):
        return np.matmul(dx, self.var[1].data.T), np.matmul(self.var[0].data.T, dx)
