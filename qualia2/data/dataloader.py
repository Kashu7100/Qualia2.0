# -*- coding: utf-8 -*- 
from ..core import *
from ..autograd import Tensor
import os
import random

class DataLoader(object):
    ''' DataLoader \n
    provides an iterable over the given dataset.
    '''
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch = batch_size
        self.shuffle = shuffle

    def __repr__(self):
        return '{}({}, batch_size={}, shuffle={})'.format(self.__class__.__name__, str(self.dataset), self.batch, self.shuffle)
        
    def __str__(self):
        return self.__class__.__name__
    
    def __len__(self):
        return len(self.dataset) // self.batch

    def __iter__(self):
        self.idx = 0
        if self.shuffle:
            self.index = np.random.permutation(len(self.dataset))
        else:
            self.index = np.arange(len(self.dataset))
        return self
            
    def __next__(self):
        if len(self) <= self.idx:
            self.idx = 0
            raise StopIteration
        features, label = self.dataset[self.index[self.idx*self.batch:(self.idx+1)*self.batch]]
        self.idx += 1
        return features, label

    def show(self, *args, **kwargs):
        ''' plot the samples of the dataset
        '''
        self.dataset.show(*args, **kwargs)