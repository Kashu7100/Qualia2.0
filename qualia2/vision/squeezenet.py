# -*- coding: utf-8 -*-
from ..nn.modules.module import Module, Sequential
from ..nn.modules import Conv2d, MaxPool2d, GlobalAvgPool2d, ReLU, Dropout
from ..nn import init
from ..functions import reshape, relu, concat
import os

path = os.path.dirname(os.path.abspath(__file__))

class Fire(Module):
    def __init__(self, inplanes, squeeze_planes, expand1x1_planes, expand3x3_planes):
        super().__init__()
        self.inplanes = inplanes
        self.squeeze = Conv2d(inplanes, squeeze_planes, kernel_size=1, padding=0)
        self.expand1x1 = Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1, padding=0)
        self.expand3x3 = Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)

    def forward(self, x):
        x = relu(self.squeeze(x))
        return concat(relu(self.expand1x1(x)), relu(self.expand3x3(x)), axis=1)

class SqueezeNet(Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.features = Sequential(
            Conv2d(3, 64, kernel_size=3, stride=2, padding=0),
            ReLU(),
            MaxPool2d(kernel_size=3, stride=2, padding=0),
            Fire(64, 16, 64, 64),
            Fire(128, 16, 64, 64),
            MaxPool2d(kernel_size=3, stride=2, padding=0),
            Fire(128, 32, 128, 128),
            Fire(256, 32, 128, 128),
            MaxPool2d(kernel_size=3, stride=2, padding=0),
            Fire(256, 48, 192, 192),
            Fire(384, 48, 192, 192),
            Fire(384, 64, 256, 256),
            Fire(512, 64, 256, 256),
        )
                
        self.classifier = Sequential(
            Dropout(0.5),
            Conv2d(512, 1000, kernel_size=1, padding=0),
            ReLU(),
            GlobalAvgPool2d()
        )

        if pretrained:
            pass
        else:
            for m in self._modules['features']._modules.values():
                if isinstance(m, Conv2d):
                    init.kaiming_uniform_(m.kernel)
                    if m.bias is not None:
                        init.constant_(m.bias, 0)
            for m in self._modules['classifier']._modules.values():
                if isinstance(m, Conv2d):
                    init.normal_(m.kernel, mean=0.0, std=0.01)
                    if m.bias is not None:
                        init.constant_(m.bias, 0)
                
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.reshape(-1, 1000)