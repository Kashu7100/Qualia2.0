# -*- coding: utf-8 -*- 
from ..core import *
from ..autograd import *

class MSELoss(Function):
    '''
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
        tmp = np.sum(np.divide(np.power(np.subtract(input.data, target.data), 2), 2), axis=1)
        if reduce:
            if size_average:
                result = Tensor(np.mean(tmp,axis=0))
            else:
                result = Tensor(np.sum(tmp,axis=0))
        else:
            result = Tensor(tmp)
        result.set_creator(MSELoss.prepare(result.shape, input, target))
        return result

    def calc_grad(self, dx):
        return np.subtract(self.var[0].data, self.var[1].data), np.subtract(self.var[1].data, self.var[0].data)

mse_loss = MSELoss(None)

class BinaryCrossEntropy(Function):
    '''Creates a criterion that measures the Binary Cross Entropy between the target and the output\n
    Args:
        input (Tensor): output of the network
        target (Tensor): label of the dataset
        reduce (bool): the losses are averaged or summed over observations for each minibatch depending on size_average. 
        size_average (bool): the losses are averaged over each loss element in the batch.

    Model: 
        l_n = y_n*log(x_n)+(1-y_n)*log(1-x_n)

    Shape:
        - Input: [N, 1]
        - Target: [N, 1]
        - Output: [1] by default
                  [N] if not reduce
    '''
    @staticmethod
    def forward(input, target, reduce=True, size_average=True):
        tmp = np.add(np.multiply(target.data, np.log(input.data)), np.multiply((1-target.data), np.log(1-input.data)))
        if reduce:
            if size_average:
                result = Tensor(np.mean(tmp,axis=0))
            else:
                result = Tensor(np.sum(tmp,axis=0))
        else:
            result = Tensor(tmp)
        result.set_creator(BinaryCrossEntropy.prepare(result.shape, input, target))
        return result

    def calc_grad(self, dx):
        dt = np.subtract(np.log(self.var[0].data), np.log(np.subtract(1, self.var[0].data)))
        dx = np.subtract(np.divide(self.var[1].data, self.var[0].data),np.divide(np.subtract(1, self.var[1].data),np.subtract(1, self.var[0].data))) 
        return dx, dt

binary_cross_entropy = BinaryCrossEntropy(None)

class CrossEntropy(Function):
    ''''Creates a criterion that measures the Cross Entropy between the target and the output\n
    Args:
        input (Tensor): output of the network
        target (Tensor): one-hot representation of label for the dataset
        reduce (bool): the losses are averaged or summed over observations for each minibatch depending on size_average. 
        size_average (bool): the losses are averaged over each loss element in the batch.
        
    Model: 
        l_n = -sum_over_classes(y_n*log(x_n))

    Shape:
        - Input: [N, num_class]
        - Target: [N, num_class]
        - Output: [1] by default
                  [N] if not reduce
    '''
    @staticmethod
    def forward(input, target, reduce=True, size_average=True):
        tmp = -np.sum(np.multiply(target.data, np.log(input.data)), axis=1)
        if reduce:
            if size_average:
                result = Tensor(np.mean(tmp,axis=0))
            else:
                result = Tensor(np.sum(tmp,axis=0))
        else:
            result = Tensor(tmp)
        result.set_creator(CrossEntropy.prepare(result.shape, input, target))
        return result

    def calc_grad(self, dx):
        dt = -np.log(self.var[0].data)
        dx = -np.divide(self.var[1].data, self.var[0].data)
        return dx, dt
    
cross_entropy = CrossEntropy(None)
