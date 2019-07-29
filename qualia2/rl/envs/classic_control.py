# -*- coding: utf-8 -*- 
from ..core import Env, Tensor

class CartPole(Env):
    ''' CartPole\n
    A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. 
    The pendulum starts upright, and the goal is to prevent it from falling over by increasing and reducing the cart's velocity.
    
    Observation: 
        Type: Box(4)
        Num	Observation               Min           Max
        0	Cart Position             -4.8          4.8
        1	Cart Velocity             -Inf          Inf
        2	Pole Angle                -24 deg       24 deg
        3	Pole Velocity At Tip      -Inf          Inf
        
    Actions:
        Type: Discrete(2)
        Num	Action
        0	Push cart to the left
        1	Push cart to the right
    Reward:
        0 for each step
        -1 if terminate condition meet before max_steps-5
        1 if terminate condition meet after max_steps-5
        (Note: original reward with the gym environment is not used)
    
    Reference:
        https://github.com/openai/gym/wiki/CartPole-v0
    '''
    def __init__(self):
        super().__init__('CartPole-v0')

    def step(self, action):
        self.steps += 1
        next_state, reward, done, info = self.env.step(action)
        return self.state_transformer(next_state), self.reward_transformer(reward, done), done, info
    
    def reward_transformer(self, reward, done):
        # clip rewards
        if done:
            if self.steps < self.max_steps-5:
                return Tensor(-1.0)
            else:
                return Tensor(1.0)
        else:
            return Tensor(0.0)

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
    
    def step(self, action):
        self.steps += 1
        next_state, reward, done, info = self.env.step(action)
        return self.state_transformer(next_state), self.reward_transformer(reward, done), done, info

    def reward_transformer(self, reward, done):
        if done:
            return Tensor(1 if self.steps < self.max_steps else -1)
        else:
            return Tensor(0)

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
    
    def step(self, action):
        self.steps += 1
        next_state, reward, done, info = self.env.step(action)
        return self.state_transformer(next_state), self.reward_transformer(reward, done), done, info

    def reward_transformer(self, reward, done):
        if done:
            return Tensor(1 if self.steps < self.max_steps else -1)
        else:
            return Tensor(0)

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
    
class Acrobot(Env):
    ''' Acrobot \n
    The acrobot system includes two joints and two links, where the joint between the two links is actuated.
    '''
    def __init__(self):
        super().__init__('Acrobot-v1')