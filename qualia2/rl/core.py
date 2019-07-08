# -*- coding: utf-8 -*- 
from .. import zeros, to_cpu
from ..core import *
from ..autograd import Tensor
from ..functions import huber_loss, amax
import random
import numpy
import gym
import matplotlib.pyplot as plt
from matplotlib import animation 
from logging import getLogger
logger = getLogger('QualiaLogger').getChild('rl')

class Agent(object):
    ''' Agent \n
    Base class for all implemented agents. Some methods needs to be over ridden.
    Args:
        actions (list): list of actions
        eps (float): epsilon value for the policy 
    '''
    def __init__(self, actions, eps):
        self.actions = actions
        self.eps = eps
        self.model = None
        self.target = None
        self.optim = None
        self.episode_count = 0
    
    def __str__(self):
        return str(self.__class__.__name__)

    def __call__(self, observation, *args):
        return self.policy(observation, *args)

    @classmethod
    def reload(cls, env, model, *args, eps=None):
        actions = list(range(env.action_space.n))
        agent = cls(actions, eps)
        agent.model = model(*args)
        agent.target = model(*args)
        agent.target.load_state_dict(agent.model.state_dict())
        return agent

    def set_optim(self, optim, **kwargs):
        self.optim = optim(self.model.params, **kwargs)
        
    def policy(self, observation, *args):
        # returns action as numpy array
        if self.eps is None:
            eps = max(0.5*(1/(self.episode_count+1)), 0.001)
        else:
            eps = self.eps
        if random.uniform(0,1) < eps:
            return numpy.random.choice(self.actions)
        else:
            return numpy.argmax(self.model(observation.reshape(1,-1), *args).asnumpy())

    def save(self, filename):
        self.model.save(filename)

    def load(self, filename):
        self.model.load(filename)
        self.target.load_state_dict(self.model.state_dict())

    def play(self, env, render=True, filename=None):
        self.eps = 0.001
        frames = []
        state = env.reset()
        done = False
        steps = 0
        episode_reward = 0
        while not done:
            if render:
                env.render()
            action = self.policy(state)
            next_state, reward, done, _ = env.step(action)
            frames.append(env.render(mode='rgb_array'))
            episode_reward += reward.data[0]
            state = next_state
            steps += 1
        logger.info("[*] Episode end - steps: {} reward: {}".format(steps, episode_reward))
        if render:
            env.close()
        if filename is not None:
            env.animate(frames, filename)

    def get_train_signal(self, experience, gamma):
        self.model.train()
        state, next_state, reward, action, done = experience
        # get state action value
        action_value = self.model(state).gather(1, action) 
        action_next = amax(self.model(next_state), axis=1)
        action_next[np.all(next_state.data==0, axis=1)] = 0
        target_action_value = reward + gamma*action_next
        return action_value, target_action_value.detach()

    def update(self, action_value, target_action_value, loss_func=huber_loss):
        loss = loss_func(action_value, target_action_value)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return to_cpu(loss.data) if gpu else loss.data
    
class Env(object):
    ''' Env \n
    Wrapper class of gym.env for reinforcement learning.
    Args:
        env (str): task name 
    '''
    def __init__(self, env):
        self.env = gym.make(env)
        self.steps = 0
    
    def __str__(self):
        return str(self.__class__.__name__)
        
    @property
    def max_steps(self):
        return self.env._max_episode_steps 

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space
    
    def reset(self):
        self.steps = 0
        return self.state_transformer(self.env.reset())

    def render(self, **kwargs):
        return self.env.render(**kwargs)

    def close(self):
        self.env.close()

    def step(self, action):
        self.steps += 1
        next_state, reward, done, info = self.env.step(action)
        return self.state_transformer(next_state), self.reward_transformer(reward), done, info

    def state_transformer(self, state):
        if state is None:
            return zeros((self.observation_space.shape[0]))
        return Tensor(state)

    def reward_transformer(self, reward):
        return Tensor(reward)

    def show(self, filename=None):
        frames = []
        self.env.reset()
        for _ in range(200):
            self.env.render()
            self.env.step(self.env.action_space.sample())
            frames.append(self.env.render(mode='rgb_array'))
        self.env.close()
        if filename is not None:
            self.animate(frames, filename)

    def animate(self, frames, filename):
        plt.clf()
        plt.figure(figsize=(frames[0].shape[1]/72.0, frames[0].shape[0]/72.0), dpi=72)
        result = plt.imshow(frames[0])
        plt.axis('off')
        video = animation.FuncAnimation(plt.gcf(), lambda t: result.set_data(frames[t]), frames=len(frames), interval=50)
        video.save(filename+'.mp4')
        plt.close()  
