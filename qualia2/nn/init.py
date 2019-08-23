# -*- coding: utf-8 -*- 
from ..core import *
from ..util import _mul
import math

def calculate_gain(nonlinearity, param=None):
    ''' calculate gain\n
    Return the recommended gain value for the given nonlinearity function.
    The values are as follows:
    ================= ====================================================
    nonlinearity      gain
    ================= ====================================================
    Linear / Identity :math:`1`
    Conv{1,2,3}D      :math:`1`
    Sigmoid           :math:`1`
    Tanh              :math:`\frac{5}{3}`
    ReLU              :math:`\sqrt{2}`
    Leaky Relu        :math:`\sqrt{\frac{2}{1 + \text{negative\_slope}^2}}`
    ================= ====================================================

    Args:
        nonlinearity (str): the non-linear function (`functions` name)
        param: optional parameter for the non-linear function

    Examples:
        >>> gain = nn.init.calculate_gain('leaky_relu')
    '''
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        return 1
    elif nonlinearity == 'tanh':
        return 5.0 / 3
    elif nonlinearity == 'relu':
        return math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if param is None:
            negative_slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            negative_slope = param
        else:
            raise ValueError("[*] negative_slope {} not a valid number".format(param))
        return math.sqrt(2.0 / (1 + negative_slope ** 2))
    else:
        raise ValueError("[*] Unsupported nonlinearity: {}".format(nonlinearity))

def uniform_(tensor, a=0, b=1):
    tensor.uniform(low=a, high=b)

def normal_(tensor, mean=0, std=1):
    tensor.normal(mean=mean, std=std)

def constant_(tensor, val):
    tensor.fill(val)

def ones_(tensor):
    tensor.ones()

def zeros_(tensor):
    tensor.zeros()

def _calculate_fan_in_and_fan_out(tensor):
    '''
    Calculates the in features and out features for a Tensor

    Args:
        tensor (Tensor): input tensor
    '''
    dimensions = tensor.ndim
    if dimensions < 2:
        raise ValueError("[*] Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
    if dimensions == 2:
        fan_in = tensor.shape[0]
        fan_out = tensor.shape[1]
    else:
        num_input_fmaps = tensor.shape[1]
        num_output_fmaps = tensor.shape[0]
        receptive_field_size = 1
        if tensor.ndim > 2:
            receptive_field_size = _mul(*tensor.shape[2:])
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size
    return fan_in, fan_out

def xavier_uniform_(tensor, gain=1):
    ''' xavier uniform\n 
    Fills the input `Tensor` with values according to the method
    described in "Understanding the difficulty of training deep feedforward
    neural networks" - Glorot, X. & Bengio, Y. (2010), using a uniform
    distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-a, a)` where
    .. math::
        a = \text{gain} \times \sqrt{\frac{6}{\text{fan\_in} + \text{fan\_out}}}
    Also known as Glorot initialization.

    Args:
        tensor (Tensor): an n-dimensional Tensor
        gain (float): an optional scaling factor

    Examples:
        >>> w = qualia2.empty(3, 5)
        >>> nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('relu'))
    '''
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    a = math.sqrt(3.0) * std
    tensor.uniform(low=-a, high=a)

def xavier_normal_(tensor, gain=1):
    ''' xavier normal\n
    Fills the input `Tensor` with values according to the method
    described in "Understanding the difficulty of training deep feedforward
    neural networks" - Glorot, X. & Bengio, Y. (2010), using a normal
    distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{N}(0, \text{std})` where
    .. math::
        \text{std} = \text{gain} \times \sqrt{\frac{2}{\text{fan\_in} + \text{fan\_out}}}
    Also known as Glorot initialization.
    
    Args:
        tensor (Tensor): an n-dimensional Tensor
        gain (float): an optional scaling factor
    
    Examples:
        >>> w = qualia2.empty(3, 5)
        >>> nn.init.xavier_normal_(w)
    '''
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    tensor.normal(mean=0, std=std)

def kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    '''
    Fills the input `Tensor` with values according to the method
    described in "Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification" - He, K. et al. (2015), using a
    uniform distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-\text{bound}, \text{bound})` where
    .. math::
        \text{bound} = \sqrt{\frac{6}{(1 + a^2) \times \text{fan\_in}}}
    Also known as He initialization.

    Args:
        tensor (Tensor): an n-dimensional `Tensor`
        a (float): the negative slope of the rectifier used after this layer (0 for ReLU
            by default)
        mode (str): either 'fan_in' (default) or 'fan_out'. Choosing `fan_in`
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing `fan_out` preserves the magnitudes in the
            backwards pass.
        nonlinearity (str): the non-linear function (`functions` name),
            recommended to use only with 'relu' or 'leaky_relu' (default).
    
    Examples:
        >>> w = qualia2.empty(3, 5)
        >>> nn.init.kaiming_uniform_(w, mode='fan_in', nonlinearity='relu')
    '''
    assert mode in ['fan_in', 'fan_out']
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    fan = fan_in if mode == 'fan_in' else fan_out
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std
    tensor.uniform(low=-bound, high=bound)

def kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    '''
    Fills the input `Tensor` with values according to the method
    described in "Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification" - He, K. et al. (2015), using a
    normal distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{N}(0, \text{std})` where
    .. math::
        \text{std} = \sqrt{\frac{2}{(1 + a^2) \times \text{fan\_in}}}
    Also known as He initialization.

    Args:
        tensor (Tensor): an n-dimensional `Tensor`
        a (float): the negative slope of the rectifier used after this layer (0 for ReLU
            by default)
        mode (str): either 'fan_in' (default) or 'fan_out'. Choosing `fan_in`
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing `fan_out` preserves the magnitudes in the
            backwards pass.
        nonlinearity (str): the non-linear function (`functions` name),
            recommended to use only with 'relu' or 'leaky_relu' (default).
    
    Examples:
        >>> w = qualia2.empty(3, 5)
        >>> nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='relu')
    '''
    assert mode in ['fan_in', 'fan_out']
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    fan = fan_in if mode == 'fan_in' else fan_out
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    tensor.normal(mean=0, std=std)
