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
