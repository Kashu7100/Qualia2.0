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

class HuberLoss(Function):
    '''
    Args:
        input (Tensor): output of the network
        target (Tensor): label of the dataset
        reduce (bool): the losses are averaged or summed over observations for each minibatch depending on size_average. 
        size_average (bool): the losses are averaged over each loss element in the batch.

    Model:
        l_n = sum((y_n - x_n)^2/2) for |y_n - x_n| < 1
              sum(|y_n - x_n|) otherwise
    
    Shape:
        - Input: [N, C] 
        - Target: [N, C] 
        - Output: [1] by default 
                  [N] if not reduce
    '''
    @staticmethod
    def forward(input, target, reduce=True, size_average=True):
        mask = (np.absolute(np.subtract(input.data, target.data)) < 1)
        tmp = np.absolute(np.subtract(input.data, target.data))
        tmp[mask] = np.divide(np.power(np.subtract(input.data, target.data), 2), 2)[mask]
        tmp = np.sum(tmp, axis=1)
        if reduce:
            if size_average:
                result = Tensor(np.mean(tmp,axis=0))
            else:
                result = Tensor(np.sum(tmp,axis=0))
        else:
            result = Tensor(tmp)
        result.set_creator(HuberLoss.prepare(result.shape, input, target, mask=mask))
        return result

    def calc_grad(self, dx):
        result = np.divide(np.absolute(np.subtract(self.var[0].data, self.var[1].data)), np.subtract(self.var[0].data, self.var[1].data))
        result[self.kwargs['mask']] = np.subtract(self.var[0].data, self.var[1].data)[self.kwargs['mask']]
        return result, np.negative(result)

smooth_l1_loss = HuberLoss(None)
huber_loss = HuberLoss(None)

class BinaryCrossEntropy(Function):
    '''Creates a criterion that measures the Binary Cross Entropy between the target and the output\n
    Args:
        input (Tensor): output of the network
        target (Tensor): label of the dataset
        reduce (bool): the losses are averaged or summed over observations for each minibatch depending on size_average. 
        size_average (bool): the losses are averaged over each loss element in the batch.

    Model: 
        l_n = -y_n*log(x_n)-(1-y_n)*log(1-x_n)

    Shape:
        - Input: [N, 1]
        - Target: [N, 1]
        - Output: [1] by default
                  [N] if not reduce
    '''
    @staticmethod
    def forward(input, target, reduce=True, size_average=True):
        tmp = -np.add(np.multiply(target.data, np.log(input.data+1e-8)), np.multiply((1-target.data), np.log(1-input.data+1e-8)))
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

class LogisticBinaryCrossEntropy(Function):
    '''Creates a criterion that measures the Binary Cross Entropy between the target and the logistic of output\n
    Args:
        input (Tensor): output of the network
        target (Tensor): label of the dataset
        reduce (bool): the losses are averaged or summed over observations for each minibatch depending on size_average. 
        size_average (bool): the losses are averaged over each loss element in the batch.

    Model: 
        l_n = y_n*log(1+exp(-x_n))+(1-y_n)*log(1+exp(x_n))

    Shape:
        - Input: [N, 1]
        - Target: [N, 1]
        - Output: [1] by default
                  [N] if not reduce
    '''
    @staticmethod
    def forward(input, target, reduce=True, size_average=True):
        sigmoid = np.divide(1, np.add(1, np.exp(np.negative(input.data))))
        tmp = np.add(np.multiply(target.data, np.log(sigmoid+1e-8)), np.multiply((1-target.data), np.log(1-sigmoid+1e-8)))
        if reduce:
            if size_average:
                result = Tensor(np.mean(tmp,axis=0))
            else:
                result = Tensor(np.sum(tmp,axis=0))
        else:
            result = Tensor(tmp)
        result.set_creator(BinaryCrossEntropy.prepare(result.shape, input, target, tmp=sigmoid))
        return result
    
    def calc_grad(self, dx):
        return np.subtract(self.kwargs['tmp'], self.var[1].data), np.subtract(np.log(self.kwargs['tmp']), np.log(np.subtract(1, self.kwargs['tmp'])))

logistic_binary_cross_entropy = LogisticBinaryCrossEntropy(None)

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

class SoftmaxCrossEntropy(Function):
    ''''Creates a criterion that measures the Cross Entropy between the target and the softmax of output\n
    Args:
        input (Tensor): output of the network
        target (Tensor): one-hot representation of label for the dataset
        reduce (bool): the losses are averaged or summed over observations for each minibatch depending on size_average. 
        size_average (bool): the losses are averaged over each loss element in the batch.
        
    Model: 
        l_n = -sum_over_classes(y_n*log(softmax(x_n)))

    Shape:
        - Input: [N, num_class]
        - Target: [N, num_class]
        - Output: [1] by default
                  [N] if not reduce
    '''
    @staticmethod
    def forward(input, target, reduce=True, size_average=True):
        const = np.max(input.data, axis=1, keepdims=True)
        exp = np.exp(np.subtract(input.data, const))
        softmax = np.divide(exp, np.sum(exp, axis=1, keepdims=True))
        tmp = -np.sum(np.multiply(target.data, np.log(softmax)), axis=1)
        if reduce:
            if size_average:
                result = Tensor(np.mean(tmp,axis=0))
            else:
                result = Tensor(np.sum(tmp,axis=0))
        else:
            result = Tensor(tmp)
        result.set_creator(SoftmaxCrossEntropy.prepare(result.shape, input, target, tmp=softmax))
        return result
    
    def calc_grad(self, dx):
        return np.subtract(self.kwargs['tmp'], self.var[1].data), -np.log(self.kwargs['tmp'])

softmax_cross_entropy = SoftmaxCrossEntropy(None)
