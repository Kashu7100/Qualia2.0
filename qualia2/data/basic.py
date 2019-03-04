# -*- coding: utf-8 -*- 
from .. import to_cpu
from ..core import *
from ..config import gpu
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
                ax.scatter(to_cpu(self.data[c*self.num_data:(c+1)*self.num_data,0]),to_cpu(self.data[c*self.num_data:(c+1)*self.num_data,1]))
            else:
                ax.scatter(self.data[c*self.num_data:(c+1)*self.num_data,0],self.data[c*self.num_data:(c+1)*self.num_data,1])
        plt.show()
