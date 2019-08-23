# Great thanks to torchvision implementation!
from ..functions import concat, dropout, relu
from ..nn.modules import Sequential, Module, Conv2d, ReLU, BatchNorm2d, MaxPool2d, AvgPool2d, GlobalAvgPool2d, Flatten, Linear
from ..nn import init

class DenseLayer(Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super().__init__()
        self.norm1 = BatchNorm2d(num_input_features)
        self.conv1 = Conv2d(num_input_features, bn_size*growth_rate, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm2 = BatchNorm2d(bn_size*growth_rate)
        self.conv2 = Conv2d(bn_size*growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_rate = drop_rate

    def forward(self, *prev_features):
        concated_features = concat(*prev_features, axis=1)
        bottleneck_output = self.conv1(relu(self.norm1(concated_features)))
        new_features = self.conv2(relu(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features

class DenseBlock(Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super().__init__()
        for i in range(num_layers):
            layer = DenseLayer(
                num_input_features + i*growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate
            )
            self.add_module('denselayer{}'.format(i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for layer in self._modules.values():
            features.append(layer(*features))
        return concat(*features, axis=1)

def transition(num_input_features, num_output_features):
    return Sequential(
            norm = BatchNorm2d(num_input_features),
            relu = ReLU(),
            conv = Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, padding=0, bias=False),
            pool = AvgPool2d(kernel_size=2, stride=2)
        )

class DenseNet(Module):
    r'''Densely Connected Convolutional Networks
    Args:
        growth_rate (int): how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints): how many layers in each pooling block
        num_init_features (int): the number of filters to learn in the first convolution layer
        bn_size (int): multiplicative factor for number of bottle neck layers  (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float): dropout rate after each dense layer
        num_classes (int): number of classification classes
    '''
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000, pretrained=False):
        super().__init__()
        self.features = Sequential(
            conv0 = Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            norm0 = BatchNorm2d(num_init_features),
            relu0 = ReLU(),
            pool0 = MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate
            )
            self.features.add_module('denseblock{}'.format(i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = transition(num_input_features=num_features, num_output_features=num_features//2)
                self.features.add_module('transition{}'.format(i + 1), trans)
                num_features = num_features // 2
        # final batch norm
        self.features.add_module('norm5', BatchNorm2d(num_features))
        self.features.add_module('relu_final', ReLU())
        self.features.add_module('global_pool', GlobalAvgPool2d())
        self.features.add_module('flatten', Flatten())

        self.classifier = Linear(num_features, num_classes)

        if pretrained:
            pass
        else:
            for m in self.modules():
                if isinstance(m, Conv2d):
                    init.kaiming_normal_(m.kernel)
                elif isinstance(m, BatchNorm2d):
                    init.constant_(m.weight, 1)
                    init.constant_(m.bias, 0)
                elif isinstance(m, Linear):
                    init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.features(x)
        out = self.classifier(out)
        return out

    @classmethod
    def densenet121(cls, pretrained=False):
        return cls(32, (6, 12, 24, 16), 64, pretrained=pretrained)

    @classmethod
    def densenet161(cls, pretrained=False):
        return cls(48, (6, 12, 36, 24), 96, pretrained=pretrained)

    @classmethod
    def densenet169(cls, pretrained=False):
        return cls(32, (6, 12, 32, 32), 64, pretrained=pretrained)

    @classmethod
    def densenet201(cls, pretrained=False):
        return cls(32, (6, 12, 48, 32), 64, pretrained=pretrained)

DenseNet121 = DenseNet.densenet121
DenseNet161 = DenseNet.densenet161
DenseNet169 = DenseNet.densenet169
DenseNet201 = DenseNet.densenet201