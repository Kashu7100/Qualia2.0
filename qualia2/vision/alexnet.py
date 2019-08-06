# -*- coding: utf-8 -*-
from ..nn.modules.module import Sequential, Module
from ..nn.modules import Linear, Conv2d, MaxPool2d, SoftMax, ReLU, Flatten, Dropout
import os

path = os.path.dirname(os.path.abspath(__file__))

class Alexnet(Module):
    ''' Alexnet \n
    Args:
        pretrained (bool): if true, load a pretrained weights
    '''
    def __init__(self, pretrained=False):
        super().__init__()
        self.features = Sequential(
            Conv2d(3, 64, 11, stride=4, padding=2),
            ReLU(),
            MaxPool2d(kernel_size=3, stride=2),
            Conv2d(64, 192, 5, padding=2),
            ReLU(),
            MaxPool2d(kernel_size=3, stride=2),
            Conv2d(192, 384, 3),
            ReLU(),
            Conv2d(384, 256, 3),
            ReLU(),
            Conv2d(256, 256, 3),
            ReLU(),
            MaxPool2d(kernel_size=3, stride=2)
        )
        self.classifier = Sequential(
            Dropout(0.5),
            Linear(6*6*256, 4096),
            ReLU(),
            Dropout(0.5),
            Linear(4096, 4096),
            ReLU(),
            Linear(4096, 1000),
            SoftMax()
        )

        if pretrained:
            if not os.path.exists(path+'/weights/'):
                os.makedirs(path+'/weights/')
            if not os.path.exists(path+'/weights/alexnet.hdf5'):
                print('[*] downloading weights...')
                self.download(path+'/weights/')
            self.load(path+'/weights/alexnet')
    
    def download(self, path): 
        import urllib.request 
        url = 'https://www.dropbox.com/s/3ipdx7y73o31ht3/alexnet.hdf5?dl=1'
        with urllib.request.urlopen(url) as u:
            data = u.read()
        with open(path+'alexnet.hdf5', 'wb') as file:
            file.write(data)
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.reshape(-1,6*6*256))
        return x
