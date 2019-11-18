# -*- coding: utf-8 -*- 
from .. import to_cpu
from ..core import *
from .dataset import *
from .imagenet_labels import imagenet_labels
from .transforms import Compose, ToTensor, Normalize
import matplotlib.pyplot as plt
import os
import glob

class ImageNet(Dataset):
    ''' ImageNet 1k Dataset\n

    Args:
        data_dir (str): the location of downloaded dataset 
        train (bool): if True, load training dataset
        transforms (transforms): transforms to apply on the features
        target_transforms (transforms): transforms to apply on the labels
    '''
    def __init__(self, data_dir, 
                train=True, 
                transforms=Compose([ToTensor(), Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])]), 
                target_transforms=None):
        self.data_dir = data_dir
        super().__init__(train, transforms, target_transforms)

    def state_dict(self):
        return {
            'label_map': imagenet_labels
        } 
    
    def prepare(self):
        if len(glob.glob(self.root+'/*')) == 0:
            pass