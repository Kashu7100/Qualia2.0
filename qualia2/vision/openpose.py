# -*- coding: utf-8 -*-
from ..nn.modules.module import Module, Sequential
from ..nn.modules import Conv2d, ReLU
from ..functions import reshape, concat
from .vgg import VGG19
import os

path = os.path.dirname(os.path.abspath(__file__))

class OpenPoseBody(Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.model0 = VGG19().features[0:23]
        self.model0.append(
            Conv2d(512, 256, kernel_size=3, padding=1),
            ReLU(),
            Conv2d(256, 128, kernel_size=3, padding=1),
            ReLU()
        )

        # PAFs
        # stage 1
        self.model1_1 = OpenPoseBody.create_block([
            [128, 128, 3, 1, 1],
            [128, 128, 3, 1, 1],
            [128, 128, 3, 1, 1],
            [128, 512, 1, 1, 0],
            [512, 38, 1, 1, 0]
        ])
        # stage 2-6
        block1 = [
            [185, 128, 7, 1, 3],
            [128, 128, 7, 1, 3],
            [128, 128, 7, 1, 3],
            [128, 128, 7, 1, 3],
            [128, 128, 7, 1, 3],
            [128, 128, 1, 1, 0],
            [128, 38, 1, 1, 0]
        ]
        self.model2_1 = OpenPoseBody.create_block(block1)
        self.model3_1 = OpenPoseBody.create_block(block1)
        self.model4_1 = OpenPoseBody.create_block(block1)
        self.model5_1 = OpenPoseBody.create_block(block1)
        self.model6_1 = OpenPoseBody.create_block(block1)
        
        # heatmaps
        # stage 1
        self.model1_2 = OpenPoseBody.create_block([
            [128, 128, 3, 1, 1],
            [128, 128, 3, 1, 1],
            [128, 128, 3, 1, 1],
            [128, 512, 1, 1, 0],
            [512, 19, 1, 1, 0]
        ])
        # stage 2-6
        block2 = [
            [185, 128, 7, 1, 3],
            [128, 128, 7, 1, 3],
            [128, 128, 7, 1, 3],
            [128, 128, 7, 1, 3],
            [128, 128, 7, 1, 3],
            [128, 128, 1, 1, 0],
            [128, 19, 1, 1, 0]
        ]
        self.model2_2 = OpenPoseBody.create_block(block2)
        self.model3_2 = OpenPoseBody.create_block(block2)
        self.model4_2 = OpenPoseBody.create_block(block2)
        self.model5_2 = OpenPoseBody.create_block(block2)
        self.model6_2 = OpenPoseBody.create_block(block2)

        if pretrained:
            self.load_state_dict_from_url('https://www.dropbox.com/s/mo1namsapx5ifns/openpose_body.qla?dl=1')
    
    @staticmethod
    def create_block(block):
        layers = []
        for v in block:
            layers.append(Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride=v[3], padding=v[4]))
            if v[1] in [19, 38]:
                break
            layers.append(ReLU())
        return Sequential(*layers)

    def forward(self, x):
        out1 = self.model0(x)

        out1_1 = self.model1_1(out1)
        out1_2 = self.model1_2(out1)
        out2 = concat(out1_1, out1_2, out1, axis=1)

        out2_1 = self.model2_1(out2)
        out2_2 = self.model2_2(out2)
        out3 = concat(out2_1, out2_2, out1, axis=1)

        out3_1 = self.model3_1(out3)
        out3_2 = self.model3_2(out3)
        out4 = concat(out3_1, out3_2, out1, axis=1)

        out4_1 = self.model4_1(out4)
        out4_2 = self.model4_2(out4)
        out5 = concat(out4_1, out4_2, out1, axis=1)

        out5_1 = self.model5_1(out5)
        out5_2 = self.model5_2(out5)
        out6 = concat(out5_1, out5_2, out1, axis=1)

        out6_1 = self.model6_1(out6)
        out6_2 = self.model6_2(out6)

        return out6_1, out6_2
    
class OpenPoseHand(Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.model1_0 = VGG19().features[0:23]
        self.model1_0.append(
            Conv2d(512, 512, kernel_size=3, padding=1),
            ReLU(),
            Conv2d(512, 512, kernel_size=3, padding=1),
            ReLU(),
            Conv2d(512, 512, kernel_size=3, padding=1),
            ReLU(),
            Conv2d(512, 512, kernel_size=3, padding=1),
            ReLU(),
            Conv2d(512, 128, kernel_size=3, padding=1),
            ReLU()
        )

        self.model1_1 = OpenPoseHand.create_block([
            [128, 512, 1, 1, 0],
            [512, 22, 1, 1, 0]
        ])

        block = [
            [150, 128, 7, 1, 3],
            [128, 128, 7, 1, 3],
            [128, 128, 7, 1, 3],
            [128, 128, 7, 1, 3],
            [128, 128, 7, 1, 3],
            [128, 128, 1, 1, 0],
            [128, 22, 1, 1, 0]
        ]
        self.model2 = OpenPoseHand.create_block(block)
        self.model3 = OpenPoseHand.create_block(block)
        self.model4 = OpenPoseHand.create_block(block)
        self.model5 = OpenPoseHand.create_block(block)
        self.model6 = OpenPoseHand.create_block(block)  

        if pretrained:
            self.load_state_dict_from_url('https://www.dropbox.com/s/kugt485exy21ta0/openpose_hand.qla?dl=1')
  
    @staticmethod
    def create_block(block):
        layers = []
        for v in block:
            layers.append(Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride=v[3], padding=v[4]))
            if v[1] in [22]:
                break
            layers.append(ReLU())
        return Sequential(*layers)
    
    def forward(self, x):
        out1_0 = self.model1_0(x)
        out1_1 = self.model1_1(out1_0)
        concat_stage2 = concat(out1_1, out1_0, axis=1)
        
        out_stage2 = self.model2(concat_stage2)
        concat_stage3 = concat(out_stage2, out1_0, axis=1)
        
        out_stage3 = self.model3(concat_stage3)
        concat_stage4 = concat(out_stage3, out1_0, axis=1)
        
        out_stage4 = self.model4(concat_stage4)
        concat_stage5 = concat(out_stage4, out1_0, axis=1)
            
        out_stage5 = self.model5(concat_stage5)
        concat_stage6 = concat(out_stage5, out1_0, axis=1)
            
        out_stage6 = self.model6(concat_stage6)
        return out_stage6
