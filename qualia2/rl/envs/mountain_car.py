# -*- coding: utf-8 -*- 
from ..core import Env, Tensor

class MountainCar(Env):
    ''' MountainCar\n
    Get an under powered car to the top of a hill (top = 0.5 position)
    
    Observation: 
        Type: Box(2)
        Num	Observation    Min      Max
        0 	position 	  -1.2  	0.6
        1 	velocity      -0.07 	0.07
        
    Actions:
        Type: Discrete(3)
        Num	Action
        0 	push left
        1 	no push
        2 	push right
    
    Reward:
        -1 for each step
    Reference:
        https://github.com/openai/gym/wiki/MountainCar-v0
    '''
    def __init__(self):
        super().__init__('MountainCar-v0')

class MountainCarContinuous(Env):
    ''' MountainCar\n
    Get an under powered car to the top of a hill (top = 0.5 position)
    
    Observation: 
        Type: Box(2)
        Num	Observation    Min      Max
        0 	position 	  -1.2  	0.6
        1 	velocity      -0.07 	0.07
        
    Actions:
        Type: Box(1)
        Num	Action
        0 	Push car to the left (negative value) or to the right (positive value)
    
    Reward:
        Reward is 100 for reaching the target of the hill on the right hand side, minus the squared sum of actions from start to goal.
    Reference:
        https://github.com/openai/gym/wiki/MountainCarContinuous-v0
    '''
    def __init__(self):
        super().__init__('MountainCarContinuous-v0')