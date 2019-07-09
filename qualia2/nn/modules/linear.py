# -*- coding: utf-8 -*-
import math 
from .module import Module
from ...core import * 
from ...functions import linear, tensordot
from ...autograd import Tensor 

class Linear(Module):
    '''Applies a linear transformation to the incoming data\n 
    Model: 
        y = x*w.T + b 
     
    Args: 
        in_features (int): size of each input sample 
        out_features (int): size of each output sample 
        bias (bool): whether to use bias. Default: True 
     
    Shape: 
        - Input: [N, *, in_features] where '∗' means any number of additional dimensions. 
        - Output: [N, *, out_features] where '∗' means any number of additional dimensions. 
    ''' 
    def __init__(self, in_features, out_features, bias=True): 
        super().__init__() 
        self.in_features = in_features
        self.out_features = out_features
        self.num_params += in_features*out_features
        self.weight = Tensor(np.random.normal(0, math.sqrt(1/in_features),(in_features, out_features))) 
        if bias: 
            self.bias = Tensor(np.zeros(out_features)) 
            self.num_params += out_features
        else: 
            self.bias = None
            
    def __repr__(self):
        return '{}({}, {}, bias={}) at 0x{:0{}X}'.format(self.__class__.__name__, self.in_features, self.out_features, str(self.bias is not None), id(self), 16)
    
    def forward(self, x): 
        result = tensordot(x, self.weight) + self.bias
        if self.input_shape is None:
            self.input_shape = x.shape
        if self.output_shape is None:
            self.output_shape = result.shape
        return result
    
class CLinear(Module):
    '''Applies a complex linear transformation to the incoming data\n 
    Model: 
        y = x*w.T + b 
     
    Args: 
        in_features (int): size of each input sample 
        out_features (int): size of each output sample 
        bias (bool): whether to use bias. Default: True 
     
    Shape: 
        - Input: [N, *, in_features] where '∗' means any number of additional dimensions. 
        - Output: [N, *, out_features] where '∗' means any number of additional dimensions. 
    ''' 
    def __init__(self, in_features, out_features, bias=True): 
        super().__init__() 
        self.in_features = in_features
        self.out_features = out_features
        self.num_params += in_features*out_features
        self.weight = Tensor(np.random.normal(0, math.sqrt(1/in_features),(in_features, out_features))+1j*np.random.normal(0, math.sqrt(1/in_features),(in_features, out_features)),dtype='complex128') 
        if bias: 
            self.bias = Tensor(np.zeros(out_features)+1j*np.zeros(out_features),dtype='complex128') 
            self.num_params += out_features
        else: 
            self.bias = None
            
    def __repr__(self):
        return '{}({}, {}, bias={}) at 0x{:0{}X}'.format(self.__class__.__name__, self.in_features, self.out_features, str(self.bias is not None), id(self), 16)
    
    def forward(self, x): 
        result = tensordot(x, self.weight) + self.bias
        if self.input_shape is None:
            self.input_shape = x.shape
        if self.output_shape is None:
            self.output_shape = result.shape
        return result

class NoisyLinear(Module):
    '''Applies a linear transformation with parametric noise added to its weights\n
    Model:
        y = (w+sig_w*eps_w)*x+(b+sig_b*eps_b)

    Args:
        in_features (int): size of each input sample 
        out_features (int): size of each output sample 
        std_init (float): std for initializing weights
        factorised_noise (bool): 
    
    Shape: 
        - Input: [N, *, in_features] where '∗' means any number of additional dimensions. 
        - Output: [N, *, out_features] where '∗' means any number of additional dimensions. 
    
    Reference:
        https://arxiv.org/abs/1706.10295
    '''
    def __init__(self, in_features, out_features, std_init=0.4, factorised_noise=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.factorised_noise = factorised_noise
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu = Tensor(np.random.uniform(-mu_range, mu_range, (out_features, in_features)))
        self.weight_sigma = Tensor(np.empty(out_features, in_features))
        self.weight_sigma.fill(self.std_init / math.sqrt(self.in_features))
        self.weight_epsilon = Tensor(np.empty(out_features, in_features), requires_grad=False)
        self.bias_mu = Tensor(np.random.uniform(-mu_range, mu_range, (out_features)))
        self.bias_sigma = Tensor(np.empty(out_features))
        self.bias_sigma.fill(self.std_init / math.sqrt(self.out_features))
        self.bias_epsilon = Tensor(np.empty(out_features), requires_grad=False)
        self.sample_noise()

    def reset_params(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data = np.random.uniform(-mu_range, mu_range, (out_features, in_features))
        self.weight_sigma.fill(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data = np.random.uniform(-mu_range, mu_range, (out_features))
        self.bias_sigma.fill(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = np.random.randn(size)
        return np.sign(x)*(np.sqrt(np.abs(x)))

    def sample_noise(self):
        if self.factorised_noise:
            epsilon_in = self._scale_noise(self.in_features)
            epsilon_out = self._scale_noise(self.out_features)
            self.weight_epsilon.data = np.outer(epsilon_out, epsilon_in)
            self.bias_epsilon.data = epsilon_out
        else:
            self.weight_epsilon.data = np.random.randn(self.out_features, self.in_features)
            self.bias_epsilon.data = np.random.randn(self.out_features)

    def forward(self, inp):
        if self.training:
            return tensordot(inp, self.weight_mu + self.weight_sigma * self.weight_epsilon) + self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            return tensordot(inp, self.weight_mu) + self.bias_mu
