# -*- coding: utf-8 -*- 
from . import to_cpu
from .config import gpu
from .core import *
from .autograd import Tensor
from functools import reduce
import sys
import random
import time
from datetime import timedelta
import matplotlib.pyplot as plt

def numerical_grad(fn, tensor, *args, **kwargs):
    delta = 1e-4
    h1 = fn(tensor + delta, *args, **kwargs)
    h2 = fn(tensor - delta, *args, **kwargs)
    return np.divide(np.subtract(h1.data, h2.data), 2*delta)

def check_function(fn, *args, domain=(-1e3,1e3), **kwargs):
    arr = np.random.random_sample((100,100))
    x = Tensor(domain[0]*arr+domain[1]*(1-arr))
    out = fn(x, *args, **kwargs)
    out.backward()
    a_grad = x.grad
    n_grad = numerical_grad(fn, x, *args, **kwargs)
    sse = np.sum(np.power(np.subtract(a_grad, n_grad),2))
    print('[*] measured error: ', sse)
    assert sse < 1e-10

def _single(x):
    assert type(x) is int
    return x

def _pair(x):
    if type(x) is int:
        return (x,x)
    elif type(x) is tuple:
        assert len(x) is 2
        return x
    else:
        raise ValueError

def _triple(x):
    if type(x) is int:
        return (x,x,x)
    elif type(x) is tuple:
        assert len(x) is 3
        return x
    else:
        raise ValueError
    
def _mul(*args):
    return reduce(lambda a, b: a*b, args)

def progressbar(progress, process, text_before='', text_after=''):
    bar_length = 40
    block = int(round(bar_length*progress/process))
    sys.stdout.flush()
    text = '\r[*]{}progress: [{:.0f}%] |{}| {}/{} {}'.format(' '+text_before, progress/process*100, '#'*block + "-"*(bar_length-block), progress, process, text_after)
    sys.stdout.write(text)
    
def trainer(model, criterion, optimizer, dataloader, epochs, minibatch, save_filename, load_filename=None): 
    ''' trainer helps the training process of supervised learning
    Args: 
        model (Module): model to train 
        criterion (Function): loss function to use 
        optimizer (Optimizer): optimizer to use 
        dataloader (DataLoader): dataloader to use 
        epochs (int): number of epochs 
        minibatch (int): number of batch to use for training 
        save_filename (string): specify the filename as well as the saving path without the file extension. (ex) path/to/filename 
        load_filename (string): specify the filename as well as the loading path without the file extension. (ex) path/to/filename
    ''' 
    if load_filename is not None:
        if os.path.exists(load_filename+'.hdf5'):
            model.load(load_filename)
            print('[*] weights loaded.') 
        else:
            raise Exception('[*] File not found: weights cannot be loaded.')
    dataloader.batch = minibatch
    dataloader.training = True
    model.training = True
    losses = []
    start = time.time()
    print('[*] training started.')
    for epoch in range(epochs):
        for i, (data, label) in enumerate(dataloader): 
            output = model(data) 
            loss = criterion(output, label)
            model.zero_grad()
            loss.backward()
            optimizer.step()
            progressbar(i, len(dataloader), 'epoch: {}/{} test loss:{:.4f} '.format(epoch+1, epochs, to_cpu(loss.data) if gpu else loss.data), '(time: {})'.format(str(timedelta(seconds=time.time()-start))))
            losses.append(to_cpu(loss.data) if gpu else loss.data)
        model.save(save_filename) 
    plt.plot([i for i in range(len(losses))], losses)
    plt.show()
    print('\n[*] training completed.')

def tester(model, dataloader, minibatch, filename):
    ''' tester helps the testing process of supervised learning
    Args: 
        model (Module): model to train 
        dataloader (DataLoader): dataloader to use 
        minibatch (int): number of batch to use for testing 
        filename (string): specify the filename as well as the loading path without the file extension. (ex) path/to/filename
    ''' 
    if os.path.exists(filename+'.hdf5'):
        model.load(filename)
        print('[*] weights loaded for testing.')
    else:
        raise Exception('[*] File not found: weights cannot be loaded.') 
    dataloader.batch = minibatch
    dataloader.training = False
    model.training = False
    print('[*] testing started.')
    acc = 0
    for i, (data, label) in enumerate(dataloader): 
        output = model(data) 
        out = np.argmax(output.data, axis=1) 
        ans = np.argmax(label.data, axis=1)
        acc += sum(out == ans)/label.shape[0]
        progressbar(i, len(dataloader))
    print('\n[*] test acc: {:.2f}%'.format(float(acc/len(dataloader)*100)))

class ReplayMemory(object):
    '''Replay Memory\n
    Memory class for experience replay. 
    Args:
        capacity (int): capacity of the memory
    '''
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.transitions = None
        self.idx = 0
        self.batch = 1
        self.is_full = False

    def __len__(self):
        return self.capacity if self.is_full else self.idx
    
    def sample(self):
        if self.transitions is None:
            raise Exception('[*] Cannot sample from empty memory.')
        if not self.is_full:
            assert self.idx > self.batch
        i = random.sample(list(range(self.capacity)), self.batch) if self.is_full else random.sample(list(range(self.idx)), self.batch)
        state, action, next, reward = (self.transitions[i,:self.states],
                                       self.transitions[i,self.states:self.states+self.actions],
                                       self.transitions[i,self.states+self.actions:2*self.states+self.actions],
                                       self.transitions[i,2*self.states+self.actions:])
        return Tensor(state.reshape(self.batch,-1), requires_grad=False), action.reshape(self.batch,-1).astype(np.int32), Tensor(next.reshape(self.batch,-1), requires_grad=False), reward.reshape(self.batch,-1)

    def shuffle(self):
        if not self.is_full:
            raise Exception('[*] shuffle of non-full memory is prohibited.')
        i = np.random.permutation(self.capacity)
        self.transitions = self.transitions[i]

    @staticmethod
    def get_len(obj):
        if type(obj) is int or type(obj) is float:
            return 1
        else:
            return len(obj)

    def append(self, state, action, nextstate, reward):
        '''
        Args:
            state (ndarray|int|float): current state
            action (ndarray|int|float): action decided
            nextstate (ndarray|int|float): next state
            reward (ndarray|int|float): reward of the action took
        '''
        if self.transitions is None:
            self.states = ReplayMemory.get_len(state)
            self.actions = ReplayMemory.get_len(action)
            self.rewards = ReplayMemory.get_len(reward)
            self.transitions = np.zeros((self.capacity, 2*self.states+self.actions+self.rewards))
        if len(self) < self.capacity // self.batch:
            if (self.idx+1)//self.capacity > 0:
                self.is_full = True
        self.transitions[self.idx,:self.states] = state
        self.transitions[self.idx,self.states:self.states+self.actions] = action
        self.transitions[self.idx,self.states+self.actions:2*self.states+self.actions] = nextstate
        self.transitions[self.idx,2*self.states+self.actions:] = reward
        self.idx = (self.idx+1)%self.capacity
