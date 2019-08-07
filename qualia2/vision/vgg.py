# -*- coding: utf-8 -*-
from ..nn.modules.module import Module, Sequential
from ..nn.modules import Linear, Conv2d, MaxPool2d, AvgPool2d, ReLU, Dropout, BatchNorm2d
from ..functions import reshape
import os

path = os.path.dirname(os.path.abspath(__file__))

cfg = {
    '11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    '13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    '16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    '19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(Module):
    ''' Base class of VGG\n
    Args:
        features (Module): feanture Module
        cfg (int): model config
        pretrained (bool): if true, load a pretrained weights
    '''
    def __init__(self, features, cfg, pretrained=False, batch_norm=False):
        super().__init__()
        self.features = features
        #self.avgpool = AvgPool2d((7, 7))
        self.classifier = Sequential(
            Linear(512*7*7, 4096),
            ReLU(),
            Dropout(),
            Linear(4096, 4096),
            ReLU(),
            Dropout(),
            Linear(4096, 1000)
        )

        if pretrained:
            if not os.path.exists(path+'/weights/'):
                os.makedirs(path+'/weights/')
            if not os.path.exists(path+'/weights/vgg{}.hdf5'.format(cfg if not batch_norm else str(cfg)+'_bn')):
                print('[*] downloading weights...')
                self.download(path+'/weights/', cfg, batch_norm)
                self.unzip(path+'/weights/', cfg, batch_norm)
            self.load(path+'/weights/vgg{}'.format(cfg if not batch_norm else str(cfg)+'_bn'))

    def download(self, path, cfg, batch_norm): 
        import urllib.request
        url = {
            'vgg11': 'https://www.dropbox.com/s/ea93ty84gos9eau/vgg11.zip?dl=1',
            'vgg13': 'https://www.dropbox.com/s/ex4f98rlwp9tra4/vgg13.zip?dl=1',
        }
        with urllib.request.urlopen(url['vgg{}'.format(cfg if not batch_norm else str(cfg)+'_bn')]) as u:
            data = u.read()
        with open(path+'vgg{}.zip'.format(cfg if not batch_norm else str(cfg)+'_bn'), 'wb') as file:
            file.write(data)
        
    def unzip(self, path, cfg, batch_norm):
        from zipfile import ZipFile
        with ZipFile(path+'vgg{}.zip'.format(cfg if not batch_norm else str(cfg)+'_bn'), 'r') as zip:
            zip.extractall(path)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.reshape(-1, 512*7*7))
        return x

    @staticmethod
    def create_layers(cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers.append(MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(Conv2d(in_channels, v, kernel_size=3, padding=1))
                if batch_norm:
                    layers.append(BatchNorm2d(v))
                layers.append(ReLU())
                in_channels = v
        return Sequential(*layers)

    @classmethod
    def vgg11(cls, pretrained=False):
        return cls(VGG.create_layers(cfg['11'], False), 11, pretrained)

    @classmethod
    def vgg13(cls, pretrained=False):
        return cls(VGG.create_layers(cfg['13'], False), 13, pretrained)

    @classmethod
    def vgg16(cls, pretrained=False):
        return cls(VGG.create_layers(cfg['16'], False), 16, pretrained)

    @classmethod
    def vgg19(cls, pretrained=False):
        return cls(VGG.create_layers(cfg['19'], False), 19, pretrained)
    
    @classmethod
    def vgg11_bn(cls, pretrained=False):
        return cls(VGG.create_layers(cfg['11'], True), 11, pretrained, True)

    @classmethod
    def vgg13_bn(cls, pretrained=False):
        return cls(VGG.create_layers(cfg['13'], True), 13, pretrained, True)

    @classmethod
    def vgg16_bn(cls, pretrained=False):
        return cls(VGG.create_layers(cfg['16'], True), 16, pretrained, True)

    @classmethod
    def vgg19_bn(cls, pretrained=False):
        return cls(VGG.create_layers(cfg['19'], True), 19, pretrained, True)

VGG11 = VGG.vgg11
VGG11_bn = VGG.vgg11_bn
VGG13 = VGG.vgg13
VGG13_bn = VGG.vgg13_bn
VGG16 = VGG.vgg16
VGG16_bn = VGG.vgg16_bn
VGG19 = VGG.vgg19
VGG19_bn = VGG.vgg19_bn