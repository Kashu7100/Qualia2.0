# -*- coding: utf-8 -*- 
from ..core import *
from ..autograd import Tensor
from .dataloader import *
import matplotlib.pyplot as plt
import gzip
import random

class EMNIST(ImageLoader):
    '''EMNIST Dataset\n     
    Args:
        normalize (bool): If true, the intensity value of a specific pixel in a specific image will be rescaled from [0, 255] to [0, 1]. Default: True 
        flatten (bool): If true, data will have a shape of [N, 28*28]. Default: False 

    Shape: 
        - data: [N, 1, 28, 28] if flatten [N, 28*28]
    '''
    def __init__(self, normalize=True, flatten=False):
        super().__init__() 
        self.download()
        self.train_data = self._load_data(home_dir + '/data/download/emnist/train_data.gz')
        self.train_label = EMNIST.to_one_hot(self._load_label(home_dir + '/data/download/emnist/train_labels.gz'), 61)
        self.test_data = self._load_data(home_dir + '/data/download/emnist/test_data.gz')
        self.test_label = EMNIST.to_one_hot(self._load_label(home_dir + '/data/download/emnist/test_labels.gz'), 61)
        print('[*] done.')

        if normalize: 
            self.train_data = np.divide(self.train_data, 255.0)
            self.test_data = np.divide(self.test_data, 255.0)
        if flatten:
            self.train_data = self.train_data.reshape(-1, 28*28) 
            self.test_data = self.test_data.reshape(-1, 28*28) 
        self.mapping = None

    @property
    def label_dict(self):
        if self.mapping is None:
            self.mapping = {}
            with open(home_dir + '/data/download/emnist/mapping.txt', 'r') as f:
                for line in f:
                    idx, description = line.strip().split(' ', 1)
                    self.mapping[int(idx)] = chr(int(description.strip()))
        return self.mapping

    def download(self):
        files = { 
            'mapping.txt':'https://www.dropbox.com/s/j2pbgp5jrbidgdf/emnist_mapping.txt?dl=1',
            'train_data.gz':'https://www.dropbox.com/s/rz2bpnt59k4zy26/emnist_train_data.gz?dl=1', 
            'train_labels.gz':'https://www.dropbox.com/s/tjnpaz89x1xjwk3/emnist_train_labels.gz?dl=1', 
            'test_data.gz':'https://www.dropbox.com/s/0ngevukoflx8wkr/emnist_test_data.gz?dl=1', 
            'test_labels.gz':'https://www.dropbox.com/s/4zgdxnlcfr3h54x/emnist_test_labels.gz?dl=1' 
        }
        for filename, value in files.items():
            super().download(value, filename)
    
    def _load_data(self, filename):
        with gzip.open(filename, 'rb') as file: 
            if gpu:
                import numpy
                data = np.asarray(numpy.frombuffer(file.read(), np.uint8, offset=16))
            else:
                data = np.frombuffer(file.read(), np.uint8, offset=16) 
        return np.transpose(data.reshape(-1,1,28,28), (0,1,3,2))

    def _load_label(self, filename):
        with gzip.open(filename, 'rb') as file: 
            if gpu:
                import numpy
                labels = np.asarray(numpy.frombuffer(file.read(), np.uint8, offset=8))
            else:
                labels = np.frombuffer(file.read(), np.uint8, offset=8) 
        return labels

    def show(self, label=None):
        for i in range(61):
            plt.subplot(7,10,i+1)
            plt.xticks([]) 
            plt.yticks([]) 
            plt.grid(False)
            if label is None:
                mask = self.test_label[:,i]>0
            else:
                mask = self.test_label[:,label]>0
            img = self.test_data[mask][random.randint(0, 100)].reshape(28,28)
            plt.imshow(to_cpu(img) if gpu else img, cmap='gray', interpolation='nearest') 
        plt.show()        
