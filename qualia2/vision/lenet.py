from ..nn.modules.module import Sequential, Module
from ..nn.modules import Linear, Conv2d, AvgPool2d, SoftMax, Tanh

class LeNet5(Module):
    ''' LeNet-5 \n
    Args:
        pretrained (bool): if true, load a pretrained weights

    Input:
        Shape: (*,1,32,32)
    '''
    def __init__(self, pretrained=False):
        super().__init__()
        self.features = Sequential(
            Conv2d(3, 6, 5, stride=1, padding=0),
            Tanh(),
            AvgPool2d(kernel_size=2, stride=2),
            Conv2d(6, 16, 5, stride=1, padding=0),
            Tanh(),
            AvgPool2d(kernel_size=2, stride=2),
        )
        self.classifier = Sequential(
            Linear(5*5*16, 120),
            Tanh(),
            Linear(120, 84),
            Tanh(),
            Linear(84, 10),
            SoftMax()
        )

        if pretrained:
            pass
            
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.reshape(-1,5*5*16))
        return x