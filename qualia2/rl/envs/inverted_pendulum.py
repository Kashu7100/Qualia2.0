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
        -1 if terminate condition meet before max_steps-10
        1 if terminate condition meet after max_steps-10
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
            if self.steps < self.max_steps-10:
                return Tensor(-1.0)
            else:
                return Tensor(1.0)
        else:
            return Tensor(0.0)