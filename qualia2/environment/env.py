# -*- coding: utf-8 -*- 
from ..util import progressbar
import gym
import matplotlib.pyplot as plt
from matplotlib import animation 

class Environment(object):
    '''Environment\n
    Wrapper class of gym.env for reinforcement learning.
    '''
    def __init__(self, env, agent, max_step, max_episodes):
        self.env = gym.make(env)
        self.env.reset()
        self.num_states = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n
        self.agent = agent
        self.max_steps = max_step
        self.max_episodes = max_episodes
        self.frames = []
    
    def run(self):
        raise NotImplementedError
    
    def show(self):
        self.frames = []
        self.env.reset()
        for _ in range(200):
            self.env.render()
            self.env.step(self.env.action_space.sample())
            self.frames.append(self.env.render(mode='rgb_array'))
    
    def animate(self, filename):
        plt.figure(figsize=(self.frames[0].shape[1]/72.0, self.frames[0].shape[0]/72.0), dpi=72)
        result = plt.imshow(self.frames[0])
        plt.axis('off')
        video = animation.FuncAnimation(plt.gcf(), lambda t: result.set_data(self.frames[t]), frames=len(self.frames), interval=50)
        video.save(filename+'.mp4')
