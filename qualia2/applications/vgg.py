# -*- coding: utf-8 -*-
import sys
sys.path.append('..')
from nn.modules.module import Module, Sequential
from nn.modules import Linear, Conv2d, MaxPool2d, AvgPool2d, ReLU, Dropout, BatchNorm2d
from functions import reshape

cfg = {
    '11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    '13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    '16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    '19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(Module):
    def __init__(self, features, num_classes=1000):
        super().__init__()
        self.features = features
        self.avgpool = AvgPool2d((7, 7))
        self.classifier = Sequential(
            Linear(512 * 7 * 7, 4096),
            ReLU(),
            Dropout(),
            Linear(4096, 4096),
            ReLU(),
            Dropout(),
            Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = reshape(x, (x.shape[0], -1))
        x = self.classifier(x)
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
    def vgg11(cls, num_classes, batch_norm=False):
        return cls(VGG.create_layers(cfg['11'], batch_norm), num_classes)

    @classmethod
    def vgg13(cls, num_classes, batch_norm=False):
        return cls(VGG.create_layers(cfg['13'], batch_norm), num_classes)

    @classmethod
    def vgg16(cls, num_classes, batch_norm=False):
        return cls(VGG.create_layers(cfg['16'], batch_norm), num_classes)

    @classmethod
    def vgg19(cls, num_classes, batch_norm=False):
        return cls(VGG.create_layers(cfg['19'], batch_norm), num_classes)
