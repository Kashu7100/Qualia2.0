# -*- coding: utf-8 -*- 
from ..core import Env, Tensor

class LunarLander(Env):
    ''' LunarLander \n
    Observation:
        Type: Box(8)

    Actions:
        Type: Discrete(4)
    '''
    def __init__(self):
        super().__init__('LunarLander-v2')

class LunarLanderContinuous(Env):
    ''' LunarLanderContinuous \n
    Observation:
        Type: Box(8)

    Actions:
        Type: Box(2)
    '''
    def __init__(self):
        super().__init__('LunarLanderContinuous-v2')