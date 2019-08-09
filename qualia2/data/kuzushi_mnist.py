# -*- coding: utf-8 -*- 
from .. import to_cpu
from ..core import *
from ..autograd import Tensor
from .dataloader import ImageLoader
import matplotlib.pyplot as plt
import os
import gzip

class KuzushiMNIST(ImageLoader):
    '''KuzushiMNIST Dataset\n     
    Args:
        normalize (bool): If true, the intensity value of a specific pixel in a specific image will be rescaled from [0, 255] to [0, 1]. Default: True 
        flatten (bool): If true, data will have a shape of [N, 28*28]. Default: False 

    Shape: 
        - data: [N, 1, 28, 28] if flatten [N, 28*28]
    '''
    def __init__(self, normalize=True, flatten=False):
        super().__init__() 
        path = os.path.dirname(os.path.abspath(__file__)) 

        print('[*] preparing data...')
        if not os.path.exists(path + '/download/kuzushi_mnist/'): 
            print('    this might take few minutes.') 
            os.makedirs(path + '/download/kuzushi_mnist/') 
            self.download(path+'/download/kuzushi_mnist/')
        self.train_data = self._load_data(path + '/download/kuzushi_mnist/train_data.gz')
        self.train_label = KuzushiMNIST.to_one_hot(self._load_label(path + '/download/kuzushi_mnist/train_labels.gz'), 10)
        self.test_data = self._load_data(path + '/download/kuzushi_mnist/test_data.gz')
        self.test_label = KuzushiMNIST.to_one_hot(self._load_label(path + '/download/kuzushi_mnist/test_labels.gz'), 10)
        print('[*] done.')

        if normalize: 
            self.train_data = np.divide(self.train_data, 255.0)
            self.test_data = np.divide(self.test_data, 255.0)
        if flatten:
            self.train_data = self.train_data.reshape(-1, 28*28) 
            self.test_data = self.test_data.reshape(-1, 28*28) 

    def download(self, path): 
        import urllib.request 
        url = 'http://codh.rois.ac.jp/kmnist/dataset/kmnist/' 
        files = { 
            'train_data.gz':'train-images-idx3-ubyte.gz', 
            'train_labels.gz':'train-labels-idx1-ubyte.gz', 
            'test_data.gz':'t10k-images-idx3-ubyte.gz', 
            'test_labels.gz':'t10k-labels-idx1-ubyte.gz' 
        } 
        for key, value in files.items(): 
            if not os.path.exists(path+key): 
                urllib.request.urlretrieve(url+value, path+key) 
    
    def _load_data(self, filename):
        with gzip.open(filename, 'rb') as file: 
            if gpu:
                import numpy
                data = np.asarray(numpy.frombuffer(file.read(), np.uint8, offset=16))
            else:
                data = np.frombuffer(file.read(), np.uint8, offset=16) 
        return data.reshape(-1,1,28,28) 

    def _load_label(self, filename):
        with gzip.open(filename, 'rb') as file: 
            if gpu:
                import numpy
                labels = np.asarray(numpy.frombuffer(file.read(), np.uint8, offset=8) )
            else:
                labels = np.frombuffer(file.read(), np.uint8, offset=8) 
        return labels

    def show(self):
        for i in range(10):
            for j in range(10):
                plt.subplot(10,10,i+j*10+1)
                plt.xticks([]) 
                plt.yticks([]) 
                plt.grid(False)
                img = self.train_data[(self.train_label[:,j]>0)][i*10+j].reshape(28,28)
                plt.imshow(to_cpu(img) if gpu else img, cmap='gray', interpolation='nearest') 
        plt.show()

class Kuzushi49(ImageLoader):
    '''Kuzushi49 Dataset\n     
    Args:
        normalize (bool): If true, the intensity value of a specific pixel in a specific image will be rescaled from [0, 255] to [0, 1]. Default: True 
        flatten (bool): If true, data will have a shape of [N, 28*28]. Default: False 

    Shape: 
        - data: [N, 1, 28, 28] if flatten [N, 28*28]
    '''
    def __init__(self, normalize=True, flatten=False):
        super().__init__() 
        path = os.path.dirname(os.path.abspath(__file__)) 

        print('[*] preparing data...')
        if not os.path.exists(path + '/download/kuzushi49/'): 
            print('    this might take few minutes.') 
            os.makedirs(path + '/download/kuzushi49/') 
            self.download(path+'/download/kuzushi49/')
        self.train_data = self._load_data(path + '/download/kuzushi49/train_data.npz')
        self.train_label = KuzushiMNIST.to_one_hot(self._load_label(path + '/download/kuzushi49/train_labels.npz'), 49)
        self.test_data = self._load_data(path + '/download/kuzushi49/test_data.npz')
        self.test_label = KuzushiMNIST.to_one_hot(self._load_label(path + '/download/kuzushi49/test_labels.npz'), 49)
        print('[*] done.')

        if normalize: 
            self.train_data = np.divide(self.train_data, 255.0)
            self.test_data = np.divide(self.test_data, 255.0)
        if flatten:
            self.train_data = self.train_data.reshape(-1, 28*28) 
            self.test_data = self.test_data.reshape(-1, 28*28) 
    
    def download(self, path): 
        import urllib.request 
        url = 'http://codh.rois.ac.jp/kmnist/dataset/k49/' 
        files = { 
            'train_data.npz':'k49-train-imgs.npz', 
            'train_labels.npz':'k49-train-labels.npz', 
            'test_data.npz':'k49-test-imgs.npz', 
            'test_labels.npz':'k49-test-labels.npz' 
        } 
        for key, value in files.items(): 
            if not os.path.exists(path+key): 
                urllib.request.urlretrieve(url+value, path+key) 
    
    def _load_data(self, filename):
        if gpu:
            import numpy
            data = np.asarray(numpy.load(filename, 'r')['arr_0'])
        else:
            data = np.load(filename, 'r')['arr_0'] 
        return data.reshape(-1,1,28,28) 

    def _load_label(self, filename):
        if gpu:
            import numpy
            labels = np.asarray(numpy.load(filename, 'r')['arr_0'])
        else:
            labels = np.load(filename, 'r')['arr_0']
        return labels

    def show(self):
        for i in range(49):
            plt.subplot(5,10,i+1)
            plt.xticks([]) 
            plt.yticks([]) 
            plt.grid(False)
            img = self.train_data[(self.train_label[:,i]>0)][0].reshape(28,28)
            plt.imshow(to_cpu(img) if gpu else img, cmap='gray', interpolation='nearest') 
        plt.show()
