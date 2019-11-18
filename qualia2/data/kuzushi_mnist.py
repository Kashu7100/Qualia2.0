# -*- coding: utf-8 -*- 
from .. import to_cpu
from ..core import *
from .dataset import *
from .transforms import Compose, ToTensor, Normalize
import matplotlib.pyplot as plt
import gzip

class KuzushiMNIST(Dataset):
    '''KuzushiMNIST Dataset\n     
    
    Args:
        train (bool): if True, load training dataset
        transforms (transforms): transforms to apply on the features
        target_transforms (transforms): transforms to apply on the labels

    Shape: 
        - data: [N, 1, 28, 28] if flatten [N, 28*28]
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
            'label_map': kuzushi_mnist_labels
        }

    def prepare(self): 
        url = 'http://codh.rois.ac.jp/kmnist/dataset/kmnist/' 
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
            self.label = KuzushiMNIST.to_one_hot(self._load_label(self.root+'/train_labels.gz'), 10)
        else:
            self.data = self._load_data(self.root+'/test_data.gz')
            self.label = KuzushiMNIST.to_one_hot(self._load_label(self.root+'/test_labels.gz'), 10)
    
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

class Kuzushi49(Dataset):
    '''Kuzushi49 Dataset\n     
    
    Args:
        train (bool): if True, load training dataset
        transforms (transforms): transforms to apply on the features
        target_transforms (transforms): transforms to apply on the labels

    Shape: 
        - data: [N, 1, 28, 28]
    '''
    def __init__(self, train=True, 
                transforms=Compose([ToTensor(), Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])]), 
                target_transforms=None):
        super().__init__(train, transforms, target_transforms)

    def __len__(self):
        if self.train:
            return 232365
        else:
            return 38547

    def state_dict(self):
        return {
            'label_map': kuzushi49_labels
        }
    
    def prepare(self):
        url = 'http://codh.rois.ac.jp/kmnist/dataset/k49/' 
        files = { 
            'train_data.npz':'k49-train-imgs.npz', 
            'train_labels.npz':'k49-train-labels.npz', 
            'test_data.npz':'k49-test-imgs.npz', 
            'test_labels.npz':'k49-test-labels.npz' 
        } 
        for filename, value in files.items():
            self._download(url+value, filename)
        if self.train:
            self.data = self._load_data(self.root+'/train_data.npz')
            self.label = KuzushiMNIST.to_one_hot(self._load_label(self.root+'/train_labels.npz'), 49)
        else:
            self.data = self._load_data(self.root+'/test_data.npz')
            self.label = KuzushiMNIST.to_one_hot(self._load_label(self.root+'/test_labels.npz'), 49)
    
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

    def show(self, row=10, col=10):
        H, W = 28, 28
        img = np.zeros((H*row, W*col))
        for r in range(row):
            for c in range(col):
                img[r*H:(r+1)*H, c*W:(c+1)*W] = self.data[random.randint(0, len(self.data)-1)].reshape(H,W)
        plt.imshow(to_cpu(img) if gpu else img, cmap='gray', interpolation='nearest') 
        plt.axis('off')
        plt.show() 

kuzushi_mnist_labels = {
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

kuzushi49_labels = dict(enumerate([
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