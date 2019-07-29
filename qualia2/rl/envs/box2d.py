# -*- coding: utf-8 -*- 
from ..core import Env, Tensor

class BipedalWalker(Env):
    '''BipedalWalker \n
    Get a 2D biped walker to walk through rough terrain.
    Observation:
        Type: Box(24)
        Num 	Observation 	            Min 	Max 	Mean
        0 	    hull_angle 	                0 	    2*pi 	0.5
        1 	    hull_angularVelocity 	    -inf 	+inf 	-
        2 	    vel_x 	                    -1 	    +1 	    -
        3 	    vel_y 	                    -1 	    +1 	    -
        4   	hip_joint_1_angle 	        -inf 	+inf 	-
        5   	hip_joint_1_speed 	        -inf 	+inf 	-
        6   	knee_joint_1_angle 	        -inf 	+inf 	-
        7   	knee_joint_1_speed 	        -inf 	+inf 	-
        8   	leg_1_ground_contact_flag 	0 	    1 	    -
        9   	hip_joint_2_angle 	        -inf 	+inf 	-
        10  	hip_joint_2_speed 	        -inf 	+inf 	-
        11  	knee_joint_2_angle 	        -inf 	+inf 	-
        12     	knee_joint_2_speed 	        -inf 	+inf 	-
        13 	    leg_2_ground_contact_flag 	0 	    1 	    -
        14-23 	10 lidar readings 	        -inf 	+inf 	-    
    Actions:
        Type: Box(4) - Torque control(default)
        Num 	Name     	                Min 	Max
        0 	    Hip_1 (Torque / Velocity) 	-1 	    +1
        1 	    Knee_1 (Torque / Velocity) 	-1 	    +1
        2 	    Hip_2 (Torque / Velocity) 	-1 	    +1
        3 	    Knee_2 (Torque / Velocity) 	-1 	    +1
    Rewards:
        Reward is given for moving forward, total 300+ points up to the far end. If the robot falls, it gets -100. 
        Applying motor torque costs a small amount of points, more optimal agent will get better score. 
    Reference:
        https://github.com/openai/gym/wiki/BipedalWalker-v2
    '''
    def __init__(self):
        super().__init__('BipedalWalker-v2')

class BipedalWalkerHardcore(Env):
    '''BipedalWalker \n
    Get a 2D biped walker to walk through rough terrain.
    Observation:
        Type: Box(24)
        Num 	Observation 	            Min 	Max 	Mean
        0 	    hull_angle 	                0 	    2*pi 	0.5
        1 	    hull_angularVelocity 	    -inf 	+inf 	-
        2 	    vel_x 	                    -1 	    +1 	    -
        3 	    vel_y 	                    -1 	    +1 	    -
        4   	hip_joint_1_angle 	        -inf 	+inf 	-
        5   	hip_joint_1_speed 	        -inf 	+inf 	-
        6   	knee_joint_1_angle 	        -inf 	+inf 	-
        7   	knee_joint_1_speed 	        -inf 	+inf 	-
        8   	leg_1_ground_contact_flag 	0 	    1 	    -
        9   	hip_joint_2_angle 	        -inf 	+inf 	-
        10  	hip_joint_2_speed 	        -inf 	+inf 	-
        11  	knee_joint_2_angle 	        -inf 	+inf 	-
        12     	knee_joint_2_speed 	        -inf 	+inf 	-
        13 	    leg_2_ground_contact_flag 	0 	    1 	    -
        14-23 	10 lidar readings 	        -inf 	+inf 	-    
    Actions:
        Type: Box(4) - Torque control(default)
        Num 	Name     	                Min 	Max
        0 	    Hip_1 (Torque / Velocity) 	-1 	    +1
        1 	    Knee_1 (Torque / Velocity) 	-1 	    +1
        2 	    Hip_2 (Torque / Velocity) 	-1 	    +1
        3 	    Knee_2 (Torque / Velocity) 	-1 	    +1
    Rewards:
        Reward is given for moving forward, total 300+ points up to the far end. If the robot falls, it gets -100. 
        Applying motor torque costs a small amount of points, more optimal agent will get better score. 
    Reference:
        https://github.com/openai/gym/wiki/BipedalWalker-v2
    '''
    def __init__(self):
        super().__init__('BipedalWalkerHardcore-v2')

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

class CarRacing(Env):
    ''' CarRacing \n
    Observation:
        Type: Box(96,96,3)
    '''
    def __init__(self):
        super().__init__('CarRacing-v0')