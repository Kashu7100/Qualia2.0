# -*- coding: utf-8 -*- 
from ..core import Env, Tensor

class Pendulum(Env):
    ''' Pendulum\n
    Try to keep a frictionless pendulum standing up.

    Observation: 
        Type: Box(3)
        Num	Observation               Min           Max
        0	cos(theta)               -1.0       	1.0
        1	sin(theta)               -1.0       	1.0
        2	theta dot                -8.0 	        8.0
        
    Actions:
        Type: Box(1)
        Num	Action                    Min           Max
        0	Joint effort             -2.0       	2.0

    Reward:
        The precise equation for reward: -(theta^2 + 0.1*theta_dt^2 + 0.001*action^2)
        Theta is normalized between -pi and pi. Therefore, the lowest cost is -(pi^2 + 0.1*8^2 + 0.001*2^2) = -16.2736044, and the highest cost is 0. 
        In essence, the goal is to remain at zero angle (vertical), with the least rotational velocity, and the least effort.

    Starting State:
        Random angle from -pi to pi, and random velocity between -1 and 1
    '''
    def __init__(self):
        super().__init__('Pendulum-v0')