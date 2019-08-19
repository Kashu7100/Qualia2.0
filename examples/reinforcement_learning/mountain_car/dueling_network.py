# -*- coding: utf-8 -*- 
from qualia2.rl.envs import MountainCar
from qualia2.rl import ReplayMemory
from qualia2.rl.agents import DDQNTrainer, DDQN
from qualia2.nn.modules import Module, Linear
from qualia2.functions import mean, relu
from qualia2.nn.optim import Adam
import argparse
import os
path = os.path.dirname(os.path.abspath(__file__))

class DuelingNet(Module):
    def __init__(self):
        super().__init__()
        self.linear1 = Linear(2, 64)
        self.linear2 = Linear(64, 64)
        self.linear_adv = Linear(64, 3)
        self.linear_val = Linear(64, 1)

    def forward(self, x):
        x = relu(self.linear1(x))
        x = relu(self.linear2(x))
        adv = self.linear_adv(x)
        val = self.linear_val(x)
        result = val + adv - mean(adv, axis=1).reshape(-1,1)
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dueling Net example with Qualia2.0')
    parser.add_argument('mode', metavar='str', type=str, choices=['train', 'test'], help='select mode to run the model : train or test.')
    parser.add_argument('-i', '--itr', type=int, default=200, help=' Number of iterations for the training. Default: 500')
    parser.add_argument('-c', '--capacity', type=int, default=10000, help='Capacity of the replay memory. Default: 10000')
    parser.add_argument('-b', '--batch', type=int, default=80, help='Batch size to train the model. Default: 32')
    parser.add_argument('-s', '--save', type=bool, default=False, help='Save mp4 video of the result. Default: True')
    parser.add_argument('-p', '--plot', type=bool, default=True, help='Plot rewards over the training. Default: True')
    
    args = parser.parse_args()
    
    env = MountainCar()
    agent = DDQN.init(env, DuelingNet())
    agent.set_optim(Adam, lr=0.0001)
    
    if args.mode == 'train':
        trainer = DDQNTrainer(ReplayMemory,args.batch,args.capacity)
        agent = trainer.train(env, agent, episodes=args.itr, filename=path+'/weights/dueling_net_example')
        if args.plot:
            trainer.plot()
    if args.mode == 'test':
        agent = DDQN.init(env, DuelingNet())
        agent.load(path+'/dueling_net_example')
        
    if args.save:
        agent.play(env, filename=path+'/weights/dueling_net_example')
    else:
        agent.play(env)
