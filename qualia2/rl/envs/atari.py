# -*- coding: utf-8 -*- 
from ..core import Env, Tensor, np
from PIL import Image

class AtariBase(Env):
    @property
    def actions(self):
        return self.env.unwrapped.get_action_meanings()
        
    @staticmethod
    def state_to_image(state):
        return Image.fromarray(state)
    
    @staticmethod
    def to_gray(image):
        return image.convert('L')
    
    @staticmethod
    def resize(image, width, height):
        result = np.array(image.resize((width, height))).astype('float64')
        if result.ndim == 2:
            result = np.expand_dims(result, axis=0)
        elif result.ndim == 3:
            result = np.swapaxes(result, 0, 2)
        else:
            raise ValueError('[*] Unexpected dimension was given.')
        return result

    @staticmethod
    def normalize(image):
        return image / 255.0

class BreakOut(AtariBase):
    ''' BreakOut \n
    Maximize your score in the Atari 2600 game Breakout.

    Observation:
    Gym Default:
        Type: Box(210, 160, 3) 
        RGB image
    Transformed:
        (1, 84, 84)
        BW image
    
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
        image = BreakOut.state_to_image(state)
        image = BreakOut.to_gray(image)
        image = BreakOut.resize(image, self.width, self.height)
        image = BreakOut.normalize(image)
        return Tensor(image)

class BreakOutRAM(AtariBase):
    ''' BreakOutRAM \n
    Maximize your score in the Atari 2600 game Breakout.

    Observation:
        Box(128,)
        the RAM of the Atari machine
    
    Actions:
        Discrete(4)
        Num	Action
        0 	no operation
        1 	fire
        2 	move right
        3   move left 
    '''
    def __init__(self):
        super().__init__('Breakout-ram-v0')

class Pong(AtariBase):
    ''' Pong \n
    Maximize your score in the Atari 2600 game Pong.

    Observation:
    Gym Default:
        Type: Box(210, 160, 3) 
        RGB image
    Transformed:
        (1, 84, 84)
        BW image
    
    Actions:
        Discrete(6)
        Num	Action
        0 	no operation
        1 	fire
        2 	move right
        3   move left
        4   RIGHTFIRE
        5   RIGHTFIRE
    '''
    def __init__(self, width=84, height=84):
        super().__init__('Pong-v0')
        self.width = width
        self.height = height

    def state_transformer(self, state):
        image = Pong.state_to_image(state)
        image = Pong.to_gray(image)
        image = Pong.resize(image, self.width, self.height)
        image = Pong.normalize(image)
        return Tensor(image)

class PongRAM(AtariBase):
    ''' Pong \n
    Maximize your score in the Atari 2600 game Pong.

    Observation:
        Box(128,)
        the RAM of the Atari machine
    
    Actions:
        Discrete(6)
        Num	Action
        0 	no operation
        1 	fire
        2 	move right
        3   move left
        4   RIGHTFIRE
        5   RIGHTFIRE
    '''
    def __init__(self):
        super().__init__('Pong-ram-v0')
