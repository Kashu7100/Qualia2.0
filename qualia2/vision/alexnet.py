# -*- coding: utf-8 -*-
from ..nn.modules.module import Sequential, Module
from ..nn.modules import Linear, Conv2d, MaxPool2d, SoftMax, ReLU, Dropout

class AlexNet(Module):
    ''' AlexNet \n
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
            self.load_state_dict_from_url('https://www.dropbox.com/s/2lgr0q2h6wyxkjg/alexnet.qla?dl=1', version=1)
            
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.reshape(-1,6*6*256))
        return x