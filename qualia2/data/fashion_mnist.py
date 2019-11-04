# -*- coding: utf-8 -*- 
from .. import to_cpu
from ..core import *
from .dataset import *
from .transforms import Compose, ToTensor, Normalize
import matplotlib.pyplot as plt
import gzip

class FashionMNIST(Dataset):
    '''FashionMNIST Dataset\n     
    Args:
        normalize (bool): If true, the intensity value of a specific pixel in a specific image will be rescaled from [0, 255] to [0, 1]. Default: True 
        flatten (bool): If true, data will have a shape of [N, 28*28]. Default: False 

    Shape: 
        - data: [N, 1, 28, 28]
    '''
    def __init__(self, train=True, 
                transforms=Compose([ToTensor(), Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])]), 
                target_transforms=None):
        super().__init__(train, transforms, target_transforms)

    def __len__(self):
        if self.train:
            return 60000
        else:
            return 10000
        
    def state_dict(self):
        return {
            0: 'T-shirt/top',
            1: 'Trouser',
            2: 'Pullover',
            3: 'Dress',
            4: 'Coat',
            5: 'Sandal',
            6: 'Shirt',
            7: 'Sneaker',
            8: 'Bag',
            9: 'Ankle boot'
        }

    def prepare(self):
        url = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/' 
        files = { 
            'train_data.gz':'train-images-idx3-ubyte.gz', 
            'train_labels.gz':'train-labels-idx1-ubyte.gz', 
            'test_data.gz':'t10k-images-idx3-ubyte.gz', 
            'test_labels.gz':'t10k-labels-idx1-ubyte.gz' 
        } 
        for filename, value in files.items():
            self._download(url+value, filename)
        if self.train:
            self.data = self._load_data(self.root+'/train_data.gz')
            self.label = FashionMNIST.to_one_hot(self._load_label(self.root+'/train_labels.gz'), 10)
        else:
            self.data = self._load_data(self.root+'/test_data.gz')
            self.label = FashionMNIST.to_one_hot(self._load_label(self.root+'/test_labels.gz'), 10)
    
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
    
    def show(self, row=10, col=10):
        H, W = 28, 28
        img = np.zeros((H*row, W*col))
        for r in range(row):
            for c in range(col):
                img[r*H:(r+1)*H, c*W:(c+1)*W] = self.data[random.randint(0, len(self.data)-1)].reshape(H,W)
        plt.imshow(to_cpu(img) if gpu else img, cmap='gray', interpolation='nearest') 
        plt.axis('off')
        plt.show() 