# -*- coding: utf-8 -*- 
from .. import to_cpu
from ..core import *
from .dataset import *
from .transforms import Compose, ToTensor, Normalize
import matplotlib.pyplot as plt
import tarfile

class STL10(Dataset):
    '''STL10 Dataset\n     
    Args:
        normalize (bool): If true, the intensity value of a specific pixel in a specific image will be rescaled from [0, 255] to [0, 1]. Default: True 
        flatten (bool): If true, data will have a shape of [N, 28*28]. Default: False 

    Shape: 
        - data: [N, 3, 96, 96] if flatten [N, 3*96*96]
    '''
    def __init__(self, train=True, 
                transforms=Compose([ToTensor(), Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])]), 
                target_transforms=None):
        super().__init__(train, transforms, target_transforms)
    
    def __len__(self):
        if self.train:
            return 5000
        else:
            return 8000

    def state_dict(self):
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

    def prepare(self): 
        url = 'http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz' 
        self._download(url, 'stl10.tar.gz') 
        tarfile.open(self.root+'/stl10.tar.gz', 'r:gz').extractall(self.root)
        if self.train:
            self.data = self._load_data(self.root+'/stl10_binary/train_X.bin')
            self.label = STL10.to_one_hot(self._load_label(self.root+'/stl10_binary/train_y.bin'), 10)
        else:
            self.data = self._load_data(self.root+'/stl10_binary/test_X.bin')
            self.label = STL10.to_one_hot(self._load_label(self.root+'/stl10_binary/test_y.bin'), 10)
    
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

    def show(self, row=5, col=5):
        H, W = 96, 96
        img = np.zeros((H*row, W*col, 3))
        for r in range(row):
            for c in range(col):
                img[r*H:(r+1)*H, c*W:(c+1)*W] = self.data[random.randint(0, len(self.data)-1)].reshape(3,H,W).transpose(1,2,0)/255
        plt.imshow(to_cpu(img), interpolation='nearest') 
        plt.axis('off')
        plt.show()         