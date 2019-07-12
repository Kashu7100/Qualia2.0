# -*- coding: utf-8 -*- 
from ..autograd import Tensor
from ..core import *
from ..functions import listconcat, absolute
from collections import deque
from collections import namedtuple
import random

Experience = namedtuple('Experience', ['state','next','reward','action','done'])

class ReplayMemory(deque):
    def __init__(self, maxlen):
        super().__init__(maxlen=maxlen)
    
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return ReplayMemory(list(self)[idx])
        else:
            return super().__getitem__(idx)
        
    def sample(self, batch_size, steps=1):
        assert steps > 0
        if steps > 1:
            idx = random.sample(range(len(self)-steps),batch_size)
            result = []
            for step in range(steps):
                tmp = [*zip(*[self[i+step] for i in idx])]
                result.append(Experience(*[listconcat(tmp[i]).detach() for i in range(3)],np.concatenate(tuple([np.asarray(i).reshape(1,-1) for i in tmp[3]]),axis=0),np.array(tmp[4])))
            return result, None, None
        else:
            tmp = [*zip(*random.sample(self, batch_size))]
            return Experience(*[listconcat(tmp[i]).detach() for i in range(3)],np.concatenate(tuple([np.asarray(i).reshape(1,-1) for i in tmp[3]]),axis=0),np.array(tmp[4])), None, None   

class PrioritizedMemory(deque):
    def __init__(self, maxlen, alpha=0.6, beta_start=0.4, beta_frames=100000):
        super().__init__(maxlen=maxlen)
        self.priorities = np.zeros((maxlen))
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.pos = 0
        self.frame_idx = 1
    
    def beta(self):
        return min(1.0, self.beta_start + self.frame_idx * (1.0 - self.beta_start) / self.beta_frames)
    
    def append(self, experience):
        max_priority = np.max(self.priorities) if len(self)>0 else 1.0
        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.maxlen
        super().append(experience)
 
    def update_priorities(self, idx, priorities):
        self.priorities[idx] = priorities

    def sample(self, batch_size, steps=1):
        assert steps > 0
        if steps > 1:
            prob = self.priorities[:len(self)-steps]**self.alpha
            prob /= np.sum(prob)
            idx = np.random.choice(len(self)-steps, batch_size, p=prob)
            weights = (len(self)*prob[idx])**(-self.beta())
            weights /= np.max(weights)
            self.frame_idx += 1
            result = []
            for step in range(steps):
                tmp = [*zip(*[self[i+step] for i in to_cpu(idx)])]
                result.append(Experience(*[listconcat(tmp[i]).detach() for i in range(3)],np.concatenate(tuple([np.asarray(i).reshape(1,-1) for i in tmp[3]]),axis=0),np.array(tmp[4])))
            return result, idx, weights
        else:
            prob = self.priorities[:len(self)]**self.alpha
            prob /= np.sum(prob)
            idx = np.random.choice(len(self), batch_size, p=prob)
            tmp = [*zip(*[self[i] for i in to_cpu(idx)])]
            weights = (len(self)*prob[idx])**(-self.beta())
            weights /= np.max(weights)
            self.frame_idx += 1
            return Experience(*[listconcat(tmp[i]).detach() for i in range(3)],np.concatenate(tuple([np.asarray(i).reshape(1,-1) for i in tmp[3]]),axis=0),np.array(tmp[4])), idx, weights
