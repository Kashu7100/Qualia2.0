# -*- coding: utf-8 -*- 
from .. import to_cpu
from ..core import *
from ..autograd import Tensor
from .dataloader import *
import matplotlib.pyplot as plt
import gzip
import random

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
 
        self.download()
        self.train_data = self._load_data(home_dir + '/data/download/kuzushi_mnist/train_data.gz')
        self.train_label = KuzushiMNIST.to_one_hot(self._load_label(home_dir + '/data/download/kuzushi_mnist/train_labels.gz'), 10)
        self.test_data = self._load_data(home_dir + '/data/download/kuzushi_mnist/test_data.gz')
        self.test_label = KuzushiMNIST.to_one_hot(self._load_label(home_dir + '/data/download/kuzushi_mnist/test_labels.gz'), 10)
        print('[*] done.')

        if normalize: 
            self.train_data = np.divide(self.train_data, 255.0)
            self.test_data = np.divide(self.test_data, 255.0)
        if flatten:
            self.train_data = self.train_data.reshape(-1, 28*28) 
            self.test_data = self.test_data.reshape(-1, 28*28) 

    @property
    def label_dict(self):
        return {
            0: 'お',
            1: 'き',
            2: 'す',
            3: 'つ',
            4: 'な',
            5: 'は',
            6: 'ま',
            7: 'や',
            8: 'れ',
            9: 'を'
        }

    def download(self): 
        url = 'http://codh.rois.ac.jp/kmnist/dataset/kmnist/' 
        files = { 
            'train_data.gz':'train-images-idx3-ubyte.gz', 
            'train_labels.gz':'train-labels-idx1-ubyte.gz', 
            'test_data.gz':'t10k-images-idx3-ubyte.gz', 
            'test_labels.gz':'t10k-labels-idx1-ubyte.gz' 
        } 
        for filename, value in files.items():
            super().download(url+value, filename)
    
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

    def show(self, label=None):
        for i in range(10):
            for j in range(10):
                plt.subplot(10,10,i+j*10+1)
                plt.xticks([]) 
                plt.yticks([]) 
                plt.grid(False)
                if label is None:
                    img = self.train_data[(self.train_label[:,j]>0)][random.randint(0, 100)].reshape(28,28)
                else:
                    img = self.train_data[(self.train_label[:,label]>0)][random.randint(0, 100)].reshape(28,28)
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

        self.download()
        self.train_data = self._load_data(home_dir + '/data/download/kuzushi49/train_data.npz')
        self.train_label = KuzushiMNIST.to_one_hot(self._load_label(home_dir + '/data/download/kuzushi49/train_labels.npz'), 49)
        self.test_data = self._load_data(home_dir + '/data/download/kuzushi49/test_data.npz')
        self.test_label = KuzushiMNIST.to_one_hot(self._load_label(home_dir + '/data/download/kuzushi49/test_labels.npz'), 49)
        print('[*] done.')

        if normalize: 
            self.train_data = np.divide(self.train_data, 255.0)
            self.test_data = np.divide(self.test_data, 255.0)
        if flatten:
            self.train_data = self.train_data.reshape(-1, 28*28) 
            self.test_data = self.test_data.reshape(-1, 28*28) 

    @property
    def label_dict(self):
        return dict(enumerate([
            'あ',
            'い',
            'う',
            'え',
            'お',
            'か',
            'き',
            'く',
            'け',
            'こ',
            'さ',
            'し',
            'す',
            'せ',
            'そ',
            'た',
            'ち',
            'つ',
            'て',
            'と',
            'な',
            'に',
            'ぬ',
            'ね',
            'の',
            'は',
            'ひ',
            'ふ',
            'へ',
            'ほ',
            'ま',
            'み',
            'む',
            'め',
            'も',
            'や',
            'ゆ',
            'よ',
            'ら',
            'り',
            'る',
            'れ',
            'ろ',
            'わ',
            'ゐ',
            'ゑ',
            'を',
            'ん',
            'ゝ'
        ]))
    
    def download(self):
        url = 'http://codh.rois.ac.jp/kmnist/dataset/k49/' 
        files = { 
            'train_data.npz':'k49-train-imgs.npz', 
            'train_labels.npz':'k49-train-labels.npz', 
            'test_data.npz':'k49-test-imgs.npz', 
            'test_labels.npz':'k49-test-labels.npz' 
        } 
        for filename, value in files.items():
            super().download(url+value, filename)
    
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

    def show(self, label=None):
        for i in range(49):
            plt.subplot(5,10,i+1)
            plt.xticks([]) 
            plt.yticks([]) 
            plt.grid(False)
            if label is None:
                img = self.train_data[(self.train_label[:,i]>0)][random.randint(0, 100)].reshape(28,28)
            else:
                img = self.train_data[(self.train_label[:,label]>0)][random.randint(0, 100)].reshape(28,28)
            plt.imshow(to_cpu(img) if gpu else img, cmap='gray', interpolation='nearest') 
        plt.show()
