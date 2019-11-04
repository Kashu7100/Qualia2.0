# -*- coding: utf-8 -*- 
from ..core import *
from ..util import download_progress
from ..autograd import Tensor
from .transforms import Compose
import os
import random
import sys
from logging import getLogger
logger = getLogger('QualiaLogger').getChild('data')

class Dataset(object):
    ''' Dataset \n
    '''
    def __init__(self, train=True, transforms=None, target_transforms=None):
        logger.info('[*] preparing data...')
        logger.info('    this might take few minutes.') 
        self.train = train
        self.transforms = transforms if transforms is not None else Compose()
        self.target_transforms = target_transforms if target_transforms is not None else Compose()
        self.root = home_dir + '/data/download/{}'.format(self.__class__.__name__.lower())
        self.data = None
        self.label = None
        self.prepare()
        logger.info('[*] done.')

    def __repr__(self):
        return '{}(train={}, transforms={}, target_transforms={})'.format(self.__class__.__name__, self.train, self.transforms, self.target_transforms)
        
    def __str__(self):
        return self.__class__.__name__
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        if self.label is None:
            return self.transforms(self.data[key]), None
        else:
            return self.transforms(self.data[key]), self.target_transforms(self.label[key])

    def show(self):
        raise NotImplementedError

    def state_dict(self):
        raise NotImplementedError

    def prepare(self):
        raise NotImplementedError

    def _download(self, url, filename=None):
        ''' downloads data from the url
        Args:
            url (str): url of the data
            filename (str): filename to save the data
        '''
        if not os.path.exists(self.root+'/'):  
            os.makedirs(self.root+'/') 
        data_dir = self.root
        if filename is None:
            from urllib.parse import urlparse
            parts = urlparse(url)
            filename = os.path.basename(parts.path)
        cache = os.path.join(data_dir, filename)
        if not os.path.exists(cache): 
            from urllib.request import urlretrieve
            urlretrieve(url, cache, reporthook=download_progress)
        sys.stdout.flush()
        sys.stdout.write('\r[*] downloading 100.00%')
        print('\r')
            
    @staticmethod
    def to_one_hot(label, num_class):
        ''' convert the label to one hot representation
        Args:
            label (ndarray | Tensor): label data
            num_class (int): number of the class in a dataset
        '''
        if isinstance(label, Tensor):
            one_hot = np.zeros((len(label.data), num_class), dtype=np.int32)    
        else:
            one_hot = np.zeros((len(label), num_class), dtype=np.int32)
        for c in range(num_class):
            one_hot[:,c][label==c] = 1
        return one_hot

    @staticmethod
    def to_vector(label):
        ''' convert the label to vector representation
        Args:
            label (ndarray | Tensor): label data
        '''
        if isinstance(label, Tensor):
            return np.argmax(label.data, axis=1)
        else:
            return np.argmax(label, axis=1)