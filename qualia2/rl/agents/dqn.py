# -*- coding: utf-8 -*- 
from ..core import ValueAgent
from ..util import Trainer

class DQN(ValueAgent):
    '''DQN 2013 implementation\n
    This implementation uses single network for learning. 
    DQN class incopolates the model (Module) and the optim (Optimizer).
    The model learns with experience replay, which is implemented in update() method.
    '''
    def __init__(self, eps, actions):
        super().__init__(eps, actions)

class DQNTrainer(Trainer):
    ''' DQNTrainer \n
    Args:
        memory (deque): replay memory object
        capacity (int): capacity of the memory
        batch (int): batch size for training
        gamma (int): gamma value
    '''
    def __init__(self, memory, batch, capacity, gamma=0.99):
        super().__init__(memory, batch, capacity, gamma)    

    def train(self, env, agent, episodes=200, render=False, filename=None):
        self.before_train(env, agent)
        self.train_routine(env, agent, episodes=episodes, render=render, filename=filename)
        self.after_train()
        return agent
