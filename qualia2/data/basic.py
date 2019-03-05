# -*- coding: utf-8 -*- 
from .. import to_cpu
from ..core import *
from ..config import gpu
from ..autograd import Tensor
from .dataloader import DataLoader
import matplotlib.pyplot as plt

class Spiral(DataLoader):
    def __init__(self, num_class=3, num_data=100):
        super().__init__()
        self.num_class = num_class
        self.num_data = num_data
        self.data = np.zeros((num_data*num_class, 2))
        self.label = np.zeros((num_data*num_class, num_class))

        for c in range(num_class):
            for i in range(num_data):
                rate = i / num_data
                radius = 1.0*rate
                theta = c*4.0 + 4.0*rate + np.random.randn()*0.2
                self.data[num_data*c+i,0] = radius*np.sin(theta)
                self.data[num_data*c+i,1] = radius*np.cos(theta)
                self.label[num_data*c+i,c] = 1

    def show(self):
        fig, ax = plt.subplots()
        for c in range(self.num_class):
            if gpu:
                ax.scatter(to_cpu(self.data[(self.label[:,c]>0)][:,0]),to_cpu(self.data[(self.label[:,c]>0)][:,1]))
            else:
                ax.scatter(self.data[(self.label[:,c]>0)][:,0],self.data[(self.label[:,c]>0)][:,1])
        plt.xlim(-1,1)
        plt.ylim(-1,1)
        plt.show()
    
    def show_decision_boundary(self, model):
        h = 0.001
        x, y = np.meshgrid(np.arange(-1, 1, h), np.arange(-1, 1, h))
        out = model(Tensor(np.c_[x.ravel(), y.ravel()]))
        pred = np.argmax(out.data, axis=1)
        if gpu:
            plt.contourf(to_cpu(x), to_cpu(y), to_cpu(pred.reshape(x.shape)))
            for c in range(self.num_class):
                plt.scatter(to_cpu(self.data[(self.label[:,c]>0)][:,0]),to_cpu(self.data[(self.label[:,c]>0)][:,1]))
        else:
            plt.contourf(x, y, pred.reshape(x.shape))
            for c in range(self.num_class):
                plt.scatter(self.data[(self.label[:,c]>0)][:,0],self.data[(self.label[:,c]>0)][:,1])
        plt.xlim(-1,1)
        plt.ylim(-1,1)
        plt.show()
