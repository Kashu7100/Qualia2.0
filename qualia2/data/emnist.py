# -*- coding: utf-8 -*- 
from .. import to_cpu
from ..core import *
from .dataset import *
from .transforms import Compose, ToTensor, Normalize
import matplotlib.pyplot as plt
import gzip

class EMNIST(Dataset):
    '''EMNIST Dataset\n     
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
        self.mapping = None
    
    def __len__(self):
        if self.train:
            return 697932 
        else:
            return 116323

    def state_dict(self):
        if self.mapping is None:
            self.mapping = {}
            with open(home_dir + '/data/download/emnist/mapping.txt', 'r') as f:
                for line in f:
                    idx, description = line.strip().split(' ', 1)
                    self.mapping[int(idx)] = chr(int(description.strip()))
        return self.mapping

    def prepare(self):
        files = { 
            'mapping.txt':'https://www.dropbox.com/s/j2pbgp5jrbidgdf/emnist_mapping.txt?dl=1',
            'train_data.gz':'https://www.dropbox.com/s/rz2bpnt59k4zy26/emnist_train_data.gz?dl=1', 
            'train_labels.gz':'https://www.dropbox.com/s/tjnpaz89x1xjwk3/emnist_train_labels.gz?dl=1', 
            'test_data.gz':'https://www.dropbox.com/s/0ngevukoflx8wkr/emnist_test_data.gz?dl=1', 
            'test_labels.gz':'https://www.dropbox.com/s/4zgdxnlcfr3h54x/emnist_test_labels.gz?dl=1' 
        }
        for filename, value in files.items():
            self._download(value, filename)
        if self.train:
            self.data = self._load_data(self.root+'/train_data.gz')
            self.label = EMNIST.to_one_hot(self._load_label(self.root+'/train_labels.gz'), 61)
        else:
            self.data = self._load_data(self.root+'/test_data.gz')
            self.label = EMNIST.to_one_hot(self._load_label(self.root+'/test_labels.gz'), 61)
    
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

    def show(self, row=10, col=10):
        H, W = 28, 28
        img = np.zeros((H*row, W*col))
        for r in range(row):
            for c in range(col):
                img[r*H:(r+1)*H, c*W:(c+1)*W] = self.data[random.randint(0, len(self.data)-1)].reshape(H,W)
        plt.imshow(to_cpu(img) if gpu else img, cmap='gray', interpolation='nearest') 
        plt.axis('off')
        plt.show()  