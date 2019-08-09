# -*- coding: utf-8 -*- 
from .. import to_cpu
from ..core import *
from ..autograd import Tensor
from .dataloader import ImageLoader
import matplotlib.pyplot as plt
import os
import tarfile

class CIFAR10(ImageLoader):
    '''CIFAR10 Dataset\n     
    Args:
        normalize (bool): If true, the intensity value of a specific pixel in a specific image will be rescaled from [0, 255] to [0, 1]. Default: True 
        flatten (bool): If true, data will have a shape of [N, 3*32*32]. Default: False 

    Shape: 
        - data: [N, 3, 32, 32] if flatten [N, 3*32*32]
    '''
    def __init__(self, normalize=True, flatten=False):
        super().__init__() 
        path = os.path.dirname(os.path.abspath(__file__)) 

        print('[*] preparing data...')
        if not os.path.exists(path + '/download/cifar10/'): 
            print('    this might take few minutes.') 
            os.makedirs(path + '/download/cifar10/') 
            self.download(path+'/download/cifar10/')
            self.extract(path+'/download/cifar10/')
        self.train_data = np.empty((50000,3*32*32))
        self.train_label = np.empty((50000,10))
        for i in range(5):
            self.train_data[i*10000:(i+1)*10000] = self._load_data(path + '/download/cifar10/cifar-10-batches-py/data_batch_{}'.format(i+1))
            self.train_label[i*10000:(i+1)*10000] = CIFAR10.to_one_hot(self._load_label(path + '/download/cifar10/cifar-10-batches-py/data_batch_{}'.format(i+1)), 10)

        self.test_data = self._load_data(path + '/download/cifar10/cifar-10-batches-py/test_batch')
        self.test_label = CIFAR10.to_one_hot(self._load_label(path + '/download/cifar10/cifar-10-batches-py/test_batch'), 10)
        print('[*] done.')

        if normalize: 
            self.train_data = np.divide(self.train_data, 255.0)
            self.test_data = np.divide(self.test_data, 255.0)
        if not flatten:
            self.train_data = self.train_data.reshape(-1,3,32,32) 
            self.test_data = self.test_data.reshape(-1,3,32,32) 

    def download(self, path): 
        import urllib.request 
        url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz' 
        if not os.path.exists(path+'cifar-10.tar.gz'): 
            urllib.request.urlretrieve(url, path+'cifar-10.tar.gz') 

    def extract(self, path):
        print('[*] extracting data...')
        os.chdir(path)
        file = tarfile.open('cifar-10.tar.gz')
        file.extractall()
    
    def _unpickle(self, filename):
        import pickle
        with open(filename, 'rb') as file:
            dict = pickle.load(file, encoding='bytes')
        return dict
    
    def _load_data(self, filename):
        if gpu:
            import numpy
            data = np.asarray(self._unpickle(filename)[b'data'])
        else:
            data = self._unpickle(filename)[b'data']
        return data

    def _load_label(self, filename):
        labels = np.array(self._unpickle(filename)[b'labels']) 
        return labels

    def show(self):
        for i in range(10):
            for j in range(10):
                plt.subplot(10,10,i+j*10+1)
                plt.xticks([]) 
                plt.yticks([]) 
                plt.grid(False)
                img = self.train_data[(self.train_label[:,j]>0)][i*10+j].reshape(3,32,32).transpose(1,2,0)
                plt.imshow(to_cpu(img) if gpu else img, interpolation='nearest') 
        plt.show()        

class CIFAR100(ImageLoader):
    '''CIFAR100 Dataset\n     
    Args:
        normalize (bool): If true, the intensity value of a specific pixel in a specific image will be rescaled from [0, 255] to [0, 1]. Default: True 
        flatten (bool): If true, data will have a shape of [N, 3*32*32]. Default: False 

    Shape: 
        - data: [N, 3, 32, 32] if flatten [N, 3*32*32]
    '''
    def __init__(self, normalize=True, flatten=False):
        super().__init__() 
        path = os.path.dirname(os.path.abspath(__file__)) 

        print('[*] preparing data...')
        if not os.path.exists(path + '/download/cifar100/'): 
            print('    this might take few minutes.') 
            os.makedirs(path + '/download/cifar100/') 
            self.download(path+'/download/cifar100/')
            self.extract(path+'/download/cifar100/')
        self.train_data = self._load_data(path + '/download/cifar100/cifar-100-python/train')
        self.train_label = CIFAR100.to_one_hot(self._load_label(path + '/download/cifar100/cifar-100-python/train'), 100)

        self.test_data = self._load_data(path + '/download/cifar100/cifar-100-python/test')
        self.test_label = CIFAR100.to_one_hot(self._load_label(path + '/download/cifar100/cifar-100-python/test'), 100)
        print('[*] done.')

        if normalize: 
            self.train_data = np.divide(self.train_data, 255.0)
            self.test_data = np.divide(self.test_data, 255.0)
        if not flatten:
            self.train_data = self.train_data.reshape(-1,3,32,32) 
            self.test_data = self.test_data.reshape(-1,3,32,32) 

    def download(self, path): 
        import urllib.request 
        url = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz' 
        if not os.path.exists(path+'cifar-100.tar.gz'): 
            urllib.request.urlretrieve(url, path+'cifar-100.tar.gz') 

    def extract(self, path):
        print('[*] extracting data...')
        os.chdir(path)
        file = tarfile.open('cifar-100.tar.gz')
        file.extractall()
    
    def _unpickle(self, filename):
        import pickle
        with open(filename, 'rb') as file:
            dict = pickle.load(file, encoding='bytes')
        return dict
    
    def _load_data(self, filename):
        if gpu:
            import numpy
            data = np.asarray(self._unpickle(filename)[b'data'])
        else:
            data = self._unpickle(filename)[b'data']
        return data

    def _load_label(self, filename):
        labels = np.array(self._unpickle(filename)[b'fine_labels']) 
        return labels

    def show(self):
        for i in range(100):
            plt.subplot(10,10,i+1)
            plt.xticks([]) 
            plt.yticks([]) 
            plt.grid(False)
            img = self.train_data[(self.train_label[:,i]>0)][0].reshape(3,32,32).transpose(1,2,0)
            plt.imshow(to_cpu(img) if gpu else img, interpolation='nearest') 
        plt.show()      
