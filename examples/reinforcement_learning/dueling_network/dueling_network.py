# -*- coding: utf-8 -*- 
from qualia2.environment.mountaincar import MountainCar
from qualia2.applications.dqn import DDQN
from qualia2.nn.modules import Module, Linear
from qualia2.functions import tanh, mean
from qualia2.nn.optim import Adadelta
import argparse
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dueling network example with Qualia2.0')
    parser.add_argument('mode', metavar='str', type=str, choices=['train', 'test'], help='select mode to run the model : train or test.')
    parser.add_argument('-i', '--itr', type=int, default=1000, help=' Number of iterations for the training. Default: 1000')
    parser.add_argument('-c', '--capacity', type=int, default=10000, help='Capacity of the replay memory. Default: 10000')
    parser.add_argument('-b', '--batch', type=int, default=80, help='Batch size to train the model. Default: 80')
    parser.add_argument('-s', '--save', type=bool, default=False, help='Save mp4 video of the result. Default: False')
    parser.add_argument('-p', '--plot', type=bool, default=False, help='Plot rewards over the training. Default: False')

    args = parser.parse_args()
    if args.mode == 'train':
        agent = DDQN(Model, Adadelta, args.capacity, args.batch)
        env = MountainCar(agent, 200, args.itr)
        env.run()
        if args.save:
            env.animate(path+'/dueling_network_example')
        if args.plot:
            env.plot_rewards()
    if args.mode == 'test':
        agent = DDQN(Model, Adadelta, args.capacity, args.batch)
        agent.load(path+'/qualia2/environment/tmp/MountrainCar1000/MountainCar-ep1000-sc111')
        env = MountainCar(agent, 200, args.itr)
        env.simulate()
        if args.save:
            env.animate(path+'/dueling_network_example')
