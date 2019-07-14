# -*- coding: utf-8 -*- 
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

class BaseAgent(object):
    ''' BaseAgent \n
    Base class for value based agents. Some methods needs to be over ridden.
    Args:
        actions (list): list of actions
        eps (float): epsilon value for the policy 
    '''
    def __init__(self, actions, model):
        self.actions = actions
        self.eps = 1
        self.model = model
        self.target = model
        self.update_target_model()
        self.optim = None
        self.episode_count = 0

    @classmethod
    def init(cls, env, model):
        actions = env.action_space.n
        return cls(actions, model)

    def set_optim(self, optim, **kwargs):
        self.optim = optim(self.model.params, **kwargs)
    
    def __str__(self):
        return str('{}'.format(self.__class__.__name__))

    def __call__(self, observation, *args):
        return self.policy(observation, *args)
    
    def policy(self, observation, *args):
        raise NotImplementedError
        
    def save(self, filename):
        self.model.save(filename)

    def load(self, filename):
        self.model.load(filename)
        self.target.load_state_dict(self.model.state_dict())
    
    def play(self, env, render=True, filename=None):
        frames = []
        state = env.reset()
        done = False
        steps = 0
        episode_reward = []
        while not done:
            if render:
                frames.append(env.render(mode='rgb_array'))
            action = self.policy(state, eps=0.001)
            next, reward, done, _ = env.step(action)
            episode_reward.append(reward.data[0])                
            state = next
            steps += 1
        logger.info("[*] Episode end - steps: {} reward: {}".format(steps, sum(episode_reward)))
        if render:
            env.close()
        if filename is not None:
            env.animate(frames, filename)

    def get_train_signal(self, experience, gamma=0.9):
        self.model.eval()
        state, next_state, reward, action, done = experience
        # get state action value
        state_action_value = self.model(state).gather(1, action) 
        next_state_action_value = amax(self.model(next_state), axis=1)
        next_state_action_value[done] = 0
        target_action_value = reward + gamma * next_state_action_value
        return state_action_value, target_action_value.detach()

    def update(self, state_action_value, target_action_value, loss_func=huber_loss):
        self.model.train()
        loss = loss_func(state_action_value, target_action_value)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return to_cpu(loss.data) if gpu else loss.data
    
    def update_target_model(self):
        self.target.load_state_dict(self.model.state_dict())

class ValueAgent(BaseAgent):
    def policy(self, observation, *args, eps=None):
        # returns action as numpy array
        if eps is None:
            eps = max(0.5*(1/(self.episode_count+1)), 0.05)
        if random.random() < eps:
            return numpy.random.choice(self.actions)
        else:
            self.model.eval()
            return numpy.argmax(self.model(observation.reshape(1,-1), *args).asnumpy())
    
class PolicyAgent(BaseAgent):
    def policy(self, observation, *args, eps=None):
        # returns action as numpy array
        if eps is None:
            eps = max(0.5*(1/(self.episode_count+1)), 0.05)
        if random.random() < eps:
            return numpy.random.choice(self.actions)
        else:
            self.model.eval()
            return numpy.random.choice(self.actions, p=self.model(observation.reshape(1,-1), *args).asnumpy())

class ActorCriticAgent(BaseAgent):
    def __init__(self, actions, actor, critic):
        self.actions = actions
        self.eps = 1
        self.actor = actor
        self.actor_target = actor
        self.critic = critic
        self.critic_target = critic
        self.actor_optim = None
        self.critic_optim = None
        self.episode_count = 0

    @classmethod
    def init(cls, env, actor, critic):
        actions = env.action_space.n
        return cls(actions, actor, critic)

    def policy(self, observation, *args, eps=None):
        return self.actor(observation).asnumpy()

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
            return Tensor(np.zeros((self.observation_space.shape[0])))
        return Tensor(state)

    def reward_transformer(self, reward):
        return Tensor(reward)

    def show(self, filename=None):
        frames = []
        self.env.reset()
        for _ in range(self.max_steps):
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
