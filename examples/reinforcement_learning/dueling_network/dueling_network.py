# -*- coding: utf-8 -*- 
from qualia2.environment.mountaincar import MountainCar
from qualia2.applications.dqn import DDQN
from qualia2.nn.modules import Module, Linear
from qualia2.functions import tanh, mean
from qualia2.nn.optim import Adadelta
import os
path = os.path.dirname(os.path.abspath(__file__))

class Model(Module):
    def __init__(self):
        super().__init__()
        self.linear1 = Linear(2, 64)
        self.linear2 = Linear(64, 64)
        self.linear_adv = Linear(64, 3)
        self.linear_val = Linear(64, 1)

    def forward(self, x):
        x = tanh(self.linear1(x))
        x = tanh(self.linear2(x))
        adv = self.linear_adv(x)
        val = self.linear_val(x)
        result = val + adv - mean(adv, axis=1).reshape(-1,1) 
        return result

agent = DDQN(Model, Adadelta, 10000, 80)
env = MountainCar(agent, 200, 1000)
env.run()
env.animate(path+'/dueling_network')
