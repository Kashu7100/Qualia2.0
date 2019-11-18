# -*- coding: utf-8 -*-
from ..nn.modules.module import Module, Sequential
from ..nn.modules import Linear, Conv2d, MaxPool2d, ReLU, Dropout, BatchNorm2d
from ..functions import reshape

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
    def __init__(self, ver, pretrained=False, batch_norm=False):
        super().__init__()
        self.features = VGG.create_layers(ver, batch_norm)
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
            url = {
                'vgg11': 'https://www.dropbox.com/s/zax0up21ks8c16i/vgg11.qla?dl=1',
                'vgg13': 'https://www.dropbox.com/s/vabk0hatr4zjogl/vgg13.qla?dl=1',
                'vgg16': 'https://www.dropbox.com/s/7zy4cnv7shwdvnw/vgg16.qla?dl=1',
                'vgg19': 'https://www.dropbox.com/s/5b8lu6uiqu1xl96/vgg19.qla?dl=1',
                'vgg11_bn': '',
                'vgg13_bn': '',
                'vgg16_bn': '',
                'vgg19_bn': '',
            }
            if not batch_norm:
                self.load_state_dict_from_url(url['vgg{}'.format(ver)], version=1)
            else:
                raise FileNotFoundError

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.reshape(-1, 512*7*7))
        return x

    @staticmethod
    def create_layers(ver, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg[ver]:
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
        return cls(11, pretrained)

    @classmethod
    def vgg13(cls, pretrained=False):
        return cls(13, pretrained)

    @classmethod
    def vgg16(cls, pretrained=False):
        return cls(16, pretrained)

    @classmethod
    def vgg19(cls, pretrained=False):
        return cls(19, pretrained)
    
    @classmethod
    def vgg11_bn(cls, pretrained=False):
        return cls(11, pretrained, True)

    @classmethod
    def vgg13_bn(cls, pretrained=False):
        return cls(13, pretrained, True)

    @classmethod
    def vgg16_bn(cls, pretrained=False):
        return cls(16, pretrained, True)

    @classmethod
    def vgg19_bn(cls, pretrained=False):
        return cls(19, pretrained, True)

VGG11 = VGG.vgg11
VGG11_bn = VGG.vgg11_bn
VGG13 = VGG.vgg13
VGG13_bn = VGG.vgg13_bn
VGG16 = VGG.vgg16
VGG16_bn = VGG.vgg16_bn
VGG19 = VGG.vgg19
VGG19_bn = VGG.vgg19_bn
