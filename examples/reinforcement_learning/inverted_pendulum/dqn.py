# -*- coding: utf-8 -*- 
from qualia2.rl.envs import CartPole
from qualia2.rl import ReplayMemory
from qualia2.rl.agents import DQNTrainer, DQN
from qualia2.nn.modules import Module, Linear
from qualia2.functions import tanh
from qualia2.nn.optim import Adadelta
import argparse
import os
path = os.path.dirname(os.path.abspath(__file__))

class Model(Module):
    def __init__(self):
        super().__init__()
        self.linear1 = Linear(4, 32)
        self.linear2 = Linear(32, 32)
        self.linear3 = Linear(32, 2)

    def forward(self, x):
        x = tanh(self.linear1(x))
        x = tanh(self.linear2(x))
        x = tanh(self.linear3(x))
        return x

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DQN example with Qualia2.0')
    parser.add_argument('mode', metavar='str', type=str, choices=['train', 'test'], help='select mode to run the model : train or test.')
    parser.add_argument('-i', '--itr', type=int, default=100, help=' Number of iterations for the training. Default: 100')
    parser.add_argument('-c', '--capacity', type=int, default=10000, help='Capacity of the replay memory. Default: 10000')
    parser.add_argument('-b', '--batch', type=int, default=80, help='Batch size to train the model. Default: 80')
    parser.add_argument('-s', '--save', type=bool, default=False, help='Save mp4 video of the result. Default: False')
    parser.add_argument('-p', '--plot', type=bool, default=False, help='Plot rewards over the training. Default: False')
    args = parser.parse_args()
    
    env = CartPole()
    
    if args.mode == 'train':
        trainer = DQNTrainer(ReplayMemory,args.batch,args.capacity)
        agent = trainer.train(env, Model, Adadelta, episodes=args.itr, filename=path+'/dqn_example')
        if args.plot:
            trainer.plot()
            
    if args.mode == 'test':
        agent = DQN.reload(env, Model)
        agent.load(path+'/dqn_example')
        
    if args.save:
        agent.play(env, path+'/dqn_cartpole')
    else:
        agent.play(env)
