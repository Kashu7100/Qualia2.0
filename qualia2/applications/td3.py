# -*- coding: utf-8 -*- 
from ..core import *
from ..autograd import Tensor
from ..functions import mse_loss
from ..util import ReplayMemory
import random
from numpy import ndarray

class TD3(object):
    '''Twin Delayed DDPG implementation\n
    
    '''
    def __init__(self, actor, critic, optim, batch, gamma=0.99, **kwargs):
        self.main_actor = actor()
        self.target_actor = actor()
        self.target_actor.load_state_dict(self.main_actor.state_dict())
        self.actor_optim = optim(self.main_actor.params)

        self.main_critic1 = critic()
        self.target_critic1 = critic()
        self.target_critic1.load_state_dict(self.main_critic1.state_dict())
        self.critic1_optim = optim(self.main_critic1.params)

        self.main_critic2 = critic()
        self.target_critic2 = critic()
        self.target_critic2.load_state_dict(self.main_critic2.state_dict())
        self.critic2_optim = optim(self.main_critic2.params)

        self.batch = batch
        self.capacity = capacity
        self.memory = ReplayMemory(self.capacity)
        self.memory.batch = batch
        self.gamma = gamma
