# -*- coding: utf-8 -*-
from ..nn.modules.module import Sequential
from ..nn.modules import Linear, Conv2d, MaxPool2d, SoftMax, ReLU, Flatten, BatchNorm2d

# input image is (3,224,224)
AlexNet = Sequential()
AlexNet.append(Conv2d(3, 96, 11, stride=4))
AlexNet.append(BatchNorm2d((96)))
AlexNet.append(ReLU())

AlexNet.append(Conv2d(96, 256, 5, padding=2))
AlexNet.append(BatchNorm2d((256)))
AlexNet.append(ReLU())
AlexNet.append(MaxPool2d((2,2)))

AlexNet.append(Conv2d(256, 384, 3))
AlexNet.append(BatchNorm2d((384)))
AlexNet.append(ReLU())
AlexNet.append(MaxPool2d((2,2)))

AlexNet.append(Conv2d(384, 384, 3))
AlexNet.append(BatchNorm2d((384)))
AlexNet.append(ReLU())

AlexNet.append(Conv2d(384, 256, 3))
AlexNet.append(BatchNorm2d((256)))
AlexNet.append(ReLU())
AlexNet.append(MaxPool2d((2,2)))

AlexNet.append(Flatten())

AlexNet.append(Linear(6*6*256, 4096))
AlexNet.append(ReLU())
AlexNet.append(Linear(4096, 4096))
AlexNet.append(ReLU())
AlexNet.append(Linear(4096, 1000))
AlexNet.append(SoftMax())