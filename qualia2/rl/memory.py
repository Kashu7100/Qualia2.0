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
    
    def sample(self, batch_size, aslist=False):
        if aslist:
            return random.sample(self, batch_size)
        tmp = [*zip(*random.sample(self, batch_size))]
        return Experience(*[listconcat(tmp[i]).detach() for i in range(3)],np.concatenate(tuple([np.asarray(i).reshape(1,-1) for i in tmp[3]]),axis=0),np.array(tmp[4]))
        
