# -*- coding: utf-8 -*- 
from . import to_cpu
from .config import gpu
from .core import *
from .autograd import Tensor
from functools import reduce
import sys
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
    text = '\r[*]{}progress: [{:.0f}%] |{}| {}/{} {}'.format(' '+text_before, progress/process*100, '#'*block + "-"*(bar_length-block), progress, process, text_after)
    sys.stdout.write(text)
    sys.stdout.flush()

def trainer(model, criterion, optimizer, dataloader, epochs, minibatch, save_filename, load_filename=None): 
    '''trainer helps the training process of supervised learning
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
        print('[*] weights loaded.') 
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
    print('[*] test acc: {:.2f}%'.format(float(acc/len(dataloader)*100)))
