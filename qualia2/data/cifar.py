# -*- coding: utf-8 -*- 
from .. import to_cpu
from ..core import *
from ..autograd import Tensor
from .dataloader import ImageLoader
import matplotlib.pyplot as plt
import tarfile
import pickle
import random

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
        self.download()
        self.train_data = np.empty((50000,3*32*32))
        self.train_label = np.empty((50000,10))
        for i in range(5):
            self.train_data[i*10000:(i+1)*10000] = self._load_data(home_dir + '/data/download/cifar10/cifar-10.tar.gz', i+1, 'train')
            self.train_label[i*10000:(i+1)*10000] = CIFAR10.to_one_hot(self._load_label(home_dir + '/data/download/cifar10/cifar-10.tar.gz', i+1, 'train'), 10)

        self.test_data = self._load_data(home_dir + '/data/download/cifar10/cifar-10.tar.gz', i+1, 'test')
        self.test_label = CIFAR10.to_one_hot(self._load_label(home_dir + '/data/download/cifar10/cifar-10.tar.gz', i+1, 'test'), 10)
        print('[*] done.')

        if normalize: 
            self.train_data = np.divide(self.train_data, 255.0)
            self.test_data = np.divide(self.test_data, 255.0)
        if not flatten:
            self.train_data = self.train_data.reshape(-1,3,32,32) 
            self.test_data = self.test_data.reshape(-1,3,32,32) 

    @property
    def label_dict(self):
        return {
            0: 'airplane', 
            1: 'automobile', 
            2: 'bird', 
            3: 'cat', 
            4: 'deer', 
            5: 'dog', 
            6: 'frog', 
            7: 'horse', 
            8: 'ship', 
            9: 'truck'
        }

    def download(self): 
        url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz' 
        super().download(url, 'cifar-10.tar.gz')
    
    def _load_data(self, filename, idx, data_type='train'):
        assert data_type in ['train', 'test']
        with tarfile.open(filename, 'r:gz') as file:
            for item in file.getmembers():
                if ('data_batch_{}'.format(idx) in item.name and data_type == 'train') or ('test_batch' in item.name and data_type == 'test'):
                    data_dict = pickle.load(file.extractfile(item), encoding='bytes')
                    if gpu:
                        import numpy
                        data = np.asarray(data_dict[b'data'])
                    else:
                        data = data_dict[b'data']
                    return data

    def _load_label(self, filename, idx, data_type='train'):
        assert data_type in ['train', 'test']
        with tarfile.open(filename, 'r:gz') as file:
            for item in file.getmembers():
                if ('data_batch_{}'.format(idx) in item.name and data_type == 'train') or ('test_batch' in item.name and data_type == 'test'):
                    data_dict = pickle.load(file.extractfile(item), encoding='bytes')
                    return np.array(data_dict[b'labels'])

    def show(self, label=None):
        for i in range(10):
            for j in range(10):
                plt.subplot(10,10,i+j*10+1)
                plt.xticks([]) 
                plt.yticks([]) 
                plt.grid(False)
                if label is None:
                    img = self.train_data[(self.train_label[:,j]>0)][random.randint(0, self.train_data.shape[0]//len(self.label_dict)-1)].reshape(3,32,32).transpose(1,2,0)
                else:
                    img = self.train_data[(self.train_label[:,label]>0)][random.randint(0, self.train_data.shape[0]//len(self.label_dict)-1)].reshape(3,32,32).transpose(1,2,0)
                plt.imshow(to_cpu(img) if gpu else img, interpolation='nearest') 
        plt.show()        

class CIFAR100(ImageLoader):
    '''CIFAR100 Dataset\n     
    Args:
        normalize (bool): If true, the intensity value of a specific pixel in a specific image will be rescaled from [0, 255] to [0, 1]. Default: True 
        flatten (bool): If true, data will have a shape of [N, 3*32*32]. Default: False 
        label_type (str): "fine" label (the class to which it belongs) or "coarse" label (the superclass to which it belongs)
    Shape: 
        - data: [N, 3, 32, 32] if flatten [N, 3*32*32]
    '''
    def __init__(self, normalize=True, flatten=False, label_type='coarse'):
        super().__init__()         
        assert label_type in ['fine', 'coarse']
        self.label_type = label_type
        self.download()        
        self.train_data = self._load_data(home_dir + '/data/download/cifar100/cifar-100.tar.gz', 'train')
        self.train_label = CIFAR100.to_one_hot(self._load_label(home_dir + '/data/download/cifar100/cifar-100.tar.gz', 'train'), 100)

        self.test_data = self._load_data(home_dir + '/data/download/cifar100/cifar-100.tar.gz', 'test')
        self.test_label = CIFAR100.to_one_hot(self._load_label(home_dir + '/data/download/cifar100/cifar-100.tar.gz', 'test'), 100)
        print('[*] done.')

        if normalize: 
            self.train_data = np.divide(self.train_data, 255.0)
            self.test_data = np.divide(self.test_data, 255.0)
        if not flatten:
            self.train_data = self.train_data.reshape(-1,3,32,32) 
            self.test_data = self.test_data.reshape(-1,3,32,32) 
    
    @property
    def label_dict(self):
        if self.label_type == 'fine':
            return dict(enumerate([
                'apple',
                'aquarium_fish',
                'baby',
                'bear',
                'beaver',
                'bed',
                'bee',
                'beetle',
                'bicycle',
                'bottle',
                'bowl',
                'boy',
                'bridge',
                'bus',
                'butterfly',
                'camel',
                'can',
                'castle',
                'caterpillar',
                'cattle',
                'chair',
                'chimpanzee',
                'clock',
                'cloud',
                'cockroach',
                'couch',
                'crab',
                'crocodile',
                'cup',
                'dinosaur',
                'dolphin',
                'elephant',
                'flatfish',
                'forest',
                'fox',
                'girl',
                'hamster',
                'house',
                'kangaroo',
                'computer_keyboard',
                'lamp',
                'lawn_mower',
                'leopard',
                'lion',
                'lizard',
                'lobster',
                'man',
                'maple_tree',
                'motorcycle',
                'mountain',
                'mouse',
                'mushroom',
                'oak_tree',
                'orange',
                'orchid',
                'otter',
                'palm_tree',
                'pear',
                'pickup_truck',
                'pine_tree',
                'plain',
                'plate',
                'poppy',
                'porcupine',
                'possum',
                'rabbit',
                'raccoon',
                'ray',
                'road',
                'rocket',
                'rose',
                'sea',
                'seal',
                'shark',
                'shrew',
                'skunk',
                'skyscraper',
                'snail',
                'snake',
                'spider',
                'squirrel',
                'streetcar',
                'sunflower',
                'sweet_pepper',
                'table',
                'tank',
                'telephone',
                'television',
                'tiger',
                'tractor',
                'train',
                'trout',
                'tulip',
                'turtle',
                'wardrobe',
                'whale',
                'willow_tree',
                'wolf',
                'woman',
                'worm',
            ]))
        elif self.label_type == 'coarse':
            return dict(enumerate([
                'aquatic mammals',
                'fish',
                'flowers',
                'food containers',
                'fruit and vegetables',
                'household electrical device',
                'household furniture',
                'insects',
                'large carnivores',
                'large man-made outdoor things',
                'large natural outdoor scenes',
                'large omnivores and herbivores',
                'medium-sized mammals',
                'non-insect invertebrates',
                'people',
                'reptiles',
                'small mammals',
                'trees',
                'vehicles 1',
                'vehicles 2',
            ]))
            
    def download(self): 
        url = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz' 
        super().download(url, 'cifar-100.tar.gz')
        
    def _load_data(self, filename, data_type='train'):
        assert data_type in ['train', 'test']
        with tarfile.open(filename, 'r:gz') as file:
            for item in file.getmembers():
                if data_type in item.name:
                    data_dict = pickle.load(file.extractfile(item), encoding='bytes')
                    if gpu:
                        import numpy
                        data = np.asarray(data_dict[b'data'])
                    else:
                        data = data_dict[b'data']
                    return data

    def _load_label(self, filename, data_type='train'):
        assert data_type in ['train', 'test']
        with tarfile.open(filename, 'r:gz') as file:
            for item in file.getmembers():
                if data_type in item.name:
                    data_dict = pickle.load(file.extractfile(item), encoding='bytes')
                    if self.label_type == 'fine':
                        return np.array(data_dict[b'fine_labels'])
                    elif self.label_type == 'coarse':
                        return np.array(data_dict[b'coarse_labels'])

    def show(self, label=None):
        for i in range(len(self.label_dict)):
            plt.subplot(len(self.label_dict)//10,10,i+1)
            plt.xticks([]) 
            plt.yticks([]) 
            plt.grid(False)
            if label is None:
                img = self.train_data[(self.train_label[:,i]>0)][random.randint(0, self.train_data.shape[0]//len(self.label_dict)-1)].reshape(3,32,32).transpose(1,2,0)
            else:
                img = self.train_data[(self.train_label[:,label]>0)][random.randint(0, self.train_data.shape[0]//len(self.label_dict)-1)].reshape(3,32,32).transpose(1,2,0)
            plt.imshow(to_cpu(img) if gpu else img, interpolation='nearest') 
        plt.show()
