# -*- coding: utf-8 -*- 
from .. import to_cpu
from ..core import *
from ..autograd import Tensor
from .dataloader import DataLoader
import matplotlib.pyplot as plt
import os
import tarfile

class CIFAR10(DataLoader):
    def __init__(self):
        super().__init__() 
        path = os.path.dirname(os.path.abspath(__file__)) 
        
        print('[*] preparing data...')
        if not os.path.exists(path + '/download/cifar10/'): 
            print('    this might take few minutes.') 
            os.makedirs(path + '/download/cifar10/') 
            self.download(path+'/download/cifar10/')
        
    def download(self, path): 
        import urllib.request 
        url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz' 
        if not os.path.exists(path+'cifar-10.tar.gz'): 
            urllib.request.urlretrieve(url, path+'cifar-10.tar.gz') 
    
    def _load_data(self, filename):
        with tarfile.open(filename, "r:gz") as file:
            pass
    
    def _unpickle(filename):
        import pickle
        with open(filename, 'rb') as file:
            dict = pickle.load(file, encoding='bytes')
        return dict
                    
    
