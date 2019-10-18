# -*- coding: utf-8 -*- 
from .. import to_cpu
from ..core import *
from ..autograd import Tensor
from .dataloader import *
import matplotlib.pyplot as plt
import os, sys
import tarfile

class STL10(ImageLoader):
    '''STL10 Dataset\n     
    Args:
        normalize (bool): If true, the intensity value of a specific pixel in a specific image will be rescaled from [0, 255] to [0, 1]. Default: True 
        flatten (bool): If true, data will have a shape of [N, 28*28]. Default: False 

    Shape: 
        - data: [N, 3, 96, 96] if flatten [N, 3*96*96]
    '''
    def __init__(self, normalize=True, flatten=False):
        super().__init__() 
        path = os.path.dirname(os.path.abspath(__file__)) 

        print('[*] preparing data...')
        if not os.path.exists(path + '/download/stl10/'): 
            print('    this might take few minutes.') 
            os.makedirs(path + '/download/stl10/') 
            self.download(path+'/download/stl10/')
        self.train_data = self._load_data(path + '/download/stl10/stl10_binary/train_X.bin')
        self.train_label = STL10.to_one_hot(self._load_label(path + '/download/stl10/stl10_binary/train_y.bin'), 10)
        self.test_data = self._load_data(path + '/download/stl10/stl10_binary/test_X.bin')
        self.test_label = STL10.to_one_hot(self._load_label(path + '/download/stl10/stl10_binary/test_y.bin'), 10)
        print('[*] done.')

        if normalize: 
            self.train_data = np.divide(self.train_data, 255.0)
            self.test_data = np.divide(self.test_data, 255.0)
        if flatten:
            self.train_data = self.train_data.reshape(-1, 3*96*96) 
            self.test_data = self.test_data.reshape(-1, 3*96*96) 

    @property
    def label_dict(self):
        return {
            0: 'airplane',
            1: 'bird',
            2: 'car',
            3: 'cat',
            4: 'deer',
            5: 'dog',
            6: 'horse',
            7: 'monkey',
            8: 'ship',
            9: 'truck',
        }

    def download(self, path): 
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r[*] downloading %.2f%%' % (float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
    
        import urllib.request 
        url = 'http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz' 
        if not os.path.exists(path+'stl10.tar.gz'): 
            urllib.request.urlretrieve(url, path+'stl10.tar.gz', reporthook=_progress) 
            tarfile.open(path+'stl10.tar.gz', 'r:gz').extractall(path)
    
    def _load_data(self, filename):
        with open(filename, 'rb') as file: 
            if gpu:
                import numpy
                data = np.asarray(numpy.fromfile(file, numpy.uint8))
            else:
                data = np.fromfile(file, np.uint8) 
        return data.reshape(-1, 3, 96, 96).transpose(0,1,3,2) 

    def _load_label(self, filename):
        with open(filename, 'rb') as file: 
            if gpu:
                import numpy
                labels = np.asarray(numpy.fromfile(file, numpy.uint8))
            else:
                labels = np.fromfile(file, np.uint8)
        return labels-1

    def show(self):
        for i in range(10):
            for j in range(10):
                plt.subplot(10,10,i+j*10+1)
                plt.xticks([]) 
                plt.yticks([]) 
                plt.grid(False)
                img = self.train_data[(self.train_label[:,j]>0)][i*10+j].reshape(3,96,96).transpose(2,1,0)
                plt.imshow(to_cpu(img) if gpu else img, interpolation='nearest') 
        plt.show()        
