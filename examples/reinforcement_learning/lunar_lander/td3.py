# -*- coding: utf-8 -*- 
from qualia2.rl.envs import LunarLanderContinuous
from qualia2.rl import ReplayMemory
from qualia2.rl.agents import TD3, TD3Trainer
from qualia2.nn.modules import Module, Linear
from qualia2.functions import relu, tanh, concatenate
from qualia2.nn.optim import Adam
import argparse
import os
path = os.path.dirname(os.path.abspath(__file__))

env = LunarLanderContinuous()
max_action = float(env.action_space.high[0])

class Actor(Module):
    def __init__(self):
        super().__init__()
        self.l1 = Linear(8, 400)
        self.l2 = Linear(400, 300)
        self.l3 = Linear(300, 2)

        self.max_action = max_action
        
    def forward(self, state):
        a = relu(self.l1(state))
        a = relu(self.l2(a))
        a = tanh(self.l3(a)) * self.max_action
        return a 

class Critic(Module):
    def __init__(self):
        super().__init__()
        self.l1 = Linear(8+2, 400)
        self.l2 = Linear(400, 300)
        self.l3 = Linear(300, 1)
        
    def forward(self, state, action):
        state_action = concatenate(state, action, axis=1)
        q = relu(self.l1(state_action))
        q = relu(self.l2(q))
        q = self.l3(q)
        return q

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TD3 example with Qualia2.0')
    parser.add_argument('mode', metavar='str', type=str, choices=['train', 'test'], help='select mode to run the model : train or test.')
    parser.add_argument('-i', '--itr', type=int, default=1200, help=' Number of iterations for the training. Default: 1200')
    parser.add_argument('-c', '--capacity', type=int, default=50000, help='Capacity of the replay memory. Default: 50000')
    parser.add_argument('-b', '--batch', type=int, default=80, help='Batch size to train the model. Default: 80')
    parser.add_argument('-s', '--save', type=bool, default=False, help='Save mp4 video of the result. Default: True')
    parser.add_argument('-p', '--plot', type=bool, default=True, help='Plot rewards over the training. Default: True')
    
    args = parser.parse_args()
    agent = TD3.init(env, Actor(), Critic())
    agent.set_actor_optim(Adam, lr = 0.0005)
    agent.set_critic_optim(Adam, lr = 0.0005)

    if args.mode == 'train':
        trainer = TD3Trainer(ReplayMemory,args.batch,args.capacity)
        agent = trainer.train(env, agent, episodes=args.itr, filename=path+'/td3_example')
        trainer.plot()
    if args.mode == 'test':
        agent.load(path+'/td3')
    agent.play(env)
