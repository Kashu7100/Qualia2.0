# -*- coding: utf-8 -*- 
from ..core import Agent
from ..util import Trainer

class DQN(Agent):
    '''DQN 2013 implementation\n
    This implementation uses single network for learning. 
    DQN class incopolates the model (Module) and the optim (Optimizer).
    The model learns with experience replay, which is implemented in update() method.
    '''
    def __init__(self, eps, actions):
        super().__init__(eps, actions)

class DQNTrainer(Trainer):
    def __init__(self, memory, batch=64, capacity=2048, gamma=0.9):
        super().__init__(memory, batch, capacity, gamma)    

    def train(self, env, model, optim, episodes=150, render=False, filename=None):
        agent = DQN.reload(env, model)
        agent.set_optim(optim)
        self.before_train(env, agent)
        self.train_routine(env, agent, episodes=episodes, render=render, filename=filename)
        return agent
