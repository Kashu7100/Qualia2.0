# -*- coding: utf-8 -*- 
from ..core import *
from ..autograd import Tensor
from ..functions import huber_loss
from ..util import ReplayMemory
import random
from numpy import ndarray

class DQN(object):
    '''DQN 2013 implementation\n
    This implementation uses single network for learning. 
    DQN class incopolates the model (Module) and the optim (Optimizer).
    The model learns with experience replay, which is implemented in replay() method.
    '''
    def __init__(self, model, optim, capacity, batch, gamma=0.99):
        self.model = model
        self.optim = optim
        self.batch = batch
        self.capacity = capacity
        self.memory = ReplayMemory(self.capacity)
        self.memory.batch = batch
        self.gamma = gamma
    
    def __call__(self, state, episode, num_actions):
        '''decide an action to take based on Ɛ-greedy policy
        Args:
            state (ndarray): current state of the environment
            episode (int): current number of episode
        Returns:
            (ndarray): action to take in env
        '''
        epsilon = 0.5*(1/(episode+1))
        if epsilon <= random.uniform(0,1):
            self.model.eval()
            if state.ndim == 1:
                state = state.reshape(1,-1)
            if gpu:
                state = to_gpu(state)
            tmp = self.model(Tensor(state, requires_grad=False))
            action = np.argmax(tmp.data, axis=1)
        else:
            action = np.random.choice(num_actions, 1)
        if gpu:
            return to_cpu(action.reshape(-1))    
        else:
            return action

    def memorize(self, state, action, nextstate, reward):
        if gpu:
            state = to_gpu(state) if isinstance(state, ndarray) else state
            action = to_gpu(action) if isinstance(action, ndarray) else action
            nextstate = to_gpu(nextstate) if isinstance(nextstate, ndarray) else nextstate
            reward = to_gpu(reward) if isinstance(reward, ndarray) else reward
        self.memory.append(state, action, nextstate, reward)

    def experience_replay(self):
        if len(self.memory) <= self.batch:
            return
        state_action_value, teacher_state_action_value = self.get_teaching_signal()
        self.update_model(state_action_value, teacher_state_action_value)
    
    def get_teaching_signal(self):
        self.model.eval()
        state, action, next, reward = self.memory.sample()
        # get state action value
        state_action_value = self.model(state).gather(1, action) 
        state_action_next = np.max(self.model(next).data, axis=1).reshape(-1,1)
        state_action_next[np.all(next.data==0, axis=1)] = 0
        teacher_state_action_value = Tensor(np.add(reward, np.multiply(self.gamma, state_action_next)), requires_grad=False)
        return state_action_value, teacher_state_action_value

    def update_model(self, input, target):
        self.model.train()
        loss = huber_loss(input, target)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def save(self, filename):
        self.model.save(filename)
    
    def load(self, filename):
        self.model.load(filename)

class DDQN(object):
    pass

class DeulingNetwork(object):
    pass

class A2C(object):
    pass