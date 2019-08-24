# -*- coding: utf-8 -*-
from ..nn.modules.module import Module, Sequential
from ..nn.modules import Linear, Conv2d, MaxPool2d, GlobalAvgPool2d, Dropout, BatchNorm2d, Flatten, ReLU
from ..functions import reshape, relu
from ..nn  import init
import os

path = os.path.dirname(os.path.abspath(__file__))

def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    ''' 3x3 convolution with padding
    '''
    return Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    ''' 1x1 convolution
    '''
    return Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)

class Basic(Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, base_width=64, dilation=1, norm_layer=BatchNorm2d):
        super().__init__()
        assert base_width == 64
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = relu(out+identity)
        return out

class Bottleneck(Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, base_width=64, dilation=1, norm_layer=BatchNorm2d):
        super().__init__()
        
        width = int(planes * (base_width / 64.))
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = relu(self.bn1(self.conv1(x)))
        out = relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = relu(out+identity)
        return out

class ResNet(Module):
    r''' ResNet
    Args:
        block (Module): Basic Block to create layers
        layers (list of int): config of layers
        num_classes (int): size of output classes 
        zero_init_residual (bool): Zero-initialize the last BN in each residual branch, so that the residual branch starts with zeros, and each residual block behaves like an identity. 
                                   This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        replace_stride_with_dilation (list of bool): each element in the list indicates if we should replace the 2x2 stride with a dilated convolution
    '''
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, replace_stride_with_dilation=[False, False, False], norm_layer=BatchNorm2d, pretrained=False):
        super().__init__()
        assert len(replace_stride_with_dilation) == 3
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        self.base_width = 64
        self.features = Sequential(
            conv1 = Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
            bn1 = norm_layer(self.inplanes),
            relu1 = ReLU(),
            maxpool = MaxPool2d(kernel_size=3, stride=2, padding=1),
            layer1 = self._make_layer(block, 64, layers[0]),
            layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]),
            layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]),
            layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]),
            avgpool = GlobalAvgPool2d(),
            flatten = Flatten()
        )

        self.classifier = Linear(512 * block.expansion, num_classes)

        if pretrained:
            pass
        else:
            for m in self.modules():
                if isinstance(m, Conv2d):
                    init.kaiming_normal_(m.kernel, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, BatchNorm2d):
                    init.constant_(m.weight, 1)
                    init.constant_(m.bias, 0)

            if zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        init.constant_(m.bn3.weight, 0)
                    elif isinstance(m, Basic):
                        init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation*=stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride), 
                norm_layer(planes * block.expansion),
            )

        layers = Sequential()
        layers.append(block(self.inplanes, planes, stride, downsample, self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))
        return layers

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    @classmethod
    def resnet18(cls, pretrained=False):
        return cls(Basic, [2, 2, 2, 2], pretrained=pretrained)

    @classmethod
    def resnet34(cls, pretrained=False):
        return cls(Basic, [3, 4, 6, 3], pretrained=pretrained)

    @classmethod
    def resnet50(cls, pretrained=False):
        return cls(Bottleneck, [3, 4, 6, 3], pretrained=pretrained)

    @classmethod
    def resnet101(cls, pretrained=False):
        return cls(Bottleneck, [3, 4, 23, 3], pretrained=pretrained)
    
    @classmethod
    def resnet152(cls, pretrained=False):
        return cls(Bottleneck, [3, 8, 36, 3], pretrained=pretrained)

ResNet18 = ResNet.resnet18
ResNet34 = ResNet.resnet34
ResNet50 = ResNet.resnet50
ResNet101 = ResNet.resnet101
ResNet152 = ResNet.resnet152