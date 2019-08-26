# -*- coding: utf-8 -*- 
from .. import to_cpu
from ..core import *
from ..autograd import Tensor
from .dataloader import DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Spiral(DataLoader):
    '''Spiral Dataset\n
    Args:
        num_class (int): number of classes
        num_data (int): number of data for each classes

    Shape:
        - data: [num_class*num_data, 2]
        - label: [num_class*num_data, num_class]
    '''
    def __init__(self, num_class=3, num_data=100):
        super().__init__()
        self.num_class = num_class
        self.num_data = num_data
        self.train_data = np.zeros((num_data*num_class, 2))
        self.train_label = np.zeros((num_data*num_class, num_class))

        for c in range(num_class):
            for i in range(num_data):
                rate = i / num_data
                radius = 1.0*rate
                theta = c*4.0 + 4.0*rate + np.random.randn()*0.2
                self.train_data[num_data*c+i,0] = radius*np.sin(theta)
                self.train_data[num_data*c+i,1] = radius*np.cos(theta)
                self.train_label[num_data*c+i,c] = 1

    def show(self, label=None):
        fig, ax = plt.subplots()
        for c in range(self.num_class):
            if gpu:
                ax.scatter(to_cpu(self.train_data[(self.train_label[:,c]>0)][:,0]),to_cpu(self.train_data[(self.train_label[:,c]>0)][:,1]))
            else:
                ax.scatter(self.train_data[(self.train_label[:,c]>0)][:,0],self.train_data[(self.train_label[:,c]>0)][:,1])
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
                plt.scatter(to_cpu(self.train_data[(self.train_label[:,c]>0)][:,0]),to_cpu(self.train_data[(self.train_label[:,c]>0)][:,1]))
        else:
            plt.contourf(x, y, pred.reshape(x.shape))
            for c in range(self.num_class):
                plt.scatter(self.train_data[(self.train_label[:,c]>0)][:,0],self.train_data[(self.train_label[:,c]>0)][:,1])
        plt.xlim(-1,1)
        plt.ylim(-1,1)
        plt.show()

class SwissRoll(DataLoader):
    '''Swiss roll dataset\n
    Args:
        num_class (int): number of classes
        num_data (int): number of data for each classes
    
    Note:
        num_data % num_class == 0 
    '''
    def __init__(self, num_class=5, num_data=2000):
        super().__init__()
        assert num_data % num_class == 0
        self.num_class = num_class
        self.num_data = num_data
        self.train_data = np.zeros((self.num_data, 3))

        theta = 2*np.pi*(1+2*np.random.rand(self.num_data,1))
        x = theta*np.cos(theta)  
        y = 21*np.random.rand(self.num_data,1)
        z = theta * np.sin(theta)
        self.train_data = np.concatenate((x,y,z), axis=1)
        self.train_data += 0.2*np.random.randn(self.num_data,3)
        self.train_label = np.zeros((self.num_data, self.num_class))
        min = np.min(theta)
        i = (np.max(theta) - min)/self.num_class
        for c in range(self.num_class):
            self.train_label[:,c][np.logical_and((min+c*i<theta[:,0]),(theta[:,0]<(min+(c+1)*i)))] = 1
        
    def show(self, label=None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        if gpu:
            for c in range(self.num_class):
                ax.scatter(to_cpu(self.train_data[(self.train_label[:,c] > 0)][:,0]),
                           to_cpu(self.train_data[(self.train_label[:,c] > 0)][:,1]),
                           to_cpu(self.train_data[(self.train_label[:,c] > 0)][:,2]),)
        else:
            for c in range(self.num_class):
                ax.scatter(self.train_data[(self.train_label[:,c] > 0)][:,0],
                           self.train_data[(self.train_label[:,c] > 0)][:,1],
                           self.train_data[(self.train_label[:,c] > 0)][:,2],)
        plt.show()
