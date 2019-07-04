# -*- coding: utf-8 -*- 
from .util import Experience
from ..core import *
from ..functions import listconcat
from collections import deque
import random

class ReplayMemory(deque):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
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
            return result
        else:
            tmp = [*zip(*random.sample(self, batch_size))]
            return Experience(*[listconcat(tmp[i]).detach() for i in range(3)],np.concatenate(tuple([np.asarray(i).reshape(1,-1) for i in tmp[3]]),axis=0),np.array(tmp[4])) 
