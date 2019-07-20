# -*- coding: utf-8 -*- 
from ..core import Env, Tensor, np
from PIL import Image

class AtariBase(Env):
    @staticmethod
    def to_gray(image):
        return Image.fromarray(image).convert('L')
    
    @staticmethod
    def resize(image, width, height):
        return np.array(image.resize((width, height))).astype('float64')
    
    @staticmethod
    def normalize(image):
        return image / 255.0

class BreakOut(AtariBase):
    ''' BreakOut \n
    Maximize your score in the Atari 2600 game Breakout.

    Observation:
        Type: Box(210, 160, 3) 
        RGB image
    
    Actions:
        Discrete(4)
        Num	Action
        0 	no operation
        1 	fire
        2 	move right
        3   move left 
    
    '''
    def __init__(self, width=84, height=84):
        super().__init__('Breakout-v0')
        self.width = width
        self.height = height

    def state_transformer(self, state):
        # Convert RGB to BW
        image = BreakOut.to_gray(state)
        image = BreakOut.resize(image, self.width, self.height)
        image = BreakOut.normalize(image)
        return Tensor(image)