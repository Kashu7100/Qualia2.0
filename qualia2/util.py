# -*- coding: utf-8 -*- 
from . import to_cpu
from .core import *
from .autograd import Tensor
from functools import reduce
import sys
import random
import time
from datetime import timedelta
import matplotlib.pyplot as plt
from logging import getLogger
logger = getLogger('QualiaLogger').getChild('util')

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
    
class Trainer(object):
    ''' Trainer base class\n
    '''
    def __init__(self, batch, path):
        self.batch = batch
        self.path = path
        self.losses = []
        self.data_transformer = lambda x:x
        self.label_transformer = lambda x:x
    
    def __repr__(self):
        print('{}'.format(self.__class__.__name__))
    
    def train(self, model, dataloader, optim, criterion, epochs=200, filename=None):
        ''' trainer helps the training process of supervised learning
        Args: 
            model (Module): model to train 
            dataloader (DataLoader): dataloader to use 
            optim (Optimizer): optimizer to use 
            criterion (Function): loss function to use 
            epochs (int): number of epochs
            filename (string): specify the filename as well as the loading path without the file extension. (ex) path/to/filename
        ''' 
        self.before_train(dataloader, model, filename)
        self.train_routine(model, dataloader, optim, criterion, epochs)
        self.after_train()
        
    def before_train(self, dataloader, model, filename):
        self.data_name = str(dataloader)
        self.model_name = str(model)
        dataloader.batch = self.batch
        dataloader.training = True
        model.train()

        if filename is not None:
            if os.path.exists(filename+'.hdf5'):
                model.load(filename)
                logger.info('[*] weights loaded.') 
            else:
                logger.error('[*] File not found: weights cannot be loaded.')
                raise FileNotFoundError
        logger.info('[*] training started.')

    def before_episode(self, dataloader, model):
        pass

    def set_data_transformer(self, fn):
        self.data_transformer = fn

    def set_label_transformer(self, fn):
        self.label_transformer = fn

    def train_routine(self, model, dataloader, optim, criterion, epochs=200):
        start = time.time()
        for epoch in range(epochs):
            self.before_episode(dataloader, model)
            tmp_loss = []
            for i, (data, label) in enumerate(dataloader): 
                output = model(self.data_transformer(data)) 
                loss = criterion(output, self.label_transformer(label))
                optim.zero_grad()
                loss.backward()
                optim.step()
                progressbar(i, len(dataloader), 'epoch: {}/{} train loss:{:.4f} '.format(epoch+1, epochs, to_cpu(loss.data) if gpu else loss.data), '(time: {})'.format(str(timedelta(seconds=time.time()-start))))
                tmp_loss.append(to_cpu(loss.data) if gpu else loss.data)
            self.after_episode(epoch+1, model, tmp_loss)

    def after_episode(self, epoch, model, loss):
            if len(loss) > 0:
                self.losses.append(sum(loss)/len(loss))
            
            if not os.path.exists(self.path + '/train/{}/{}/'.format(self.data_name,self.model_name)):
                os.makedirs(self.path + '/train/{}/{}/'.format(self.data_name,self.model_name)) 
            model.save(self.path+'/train/{}/{}/{}'.format(self.data_name,self.model_name,epoch)) 
                
    def after_train(self):
        logger.info('[*] training completed.')
        self.plot(self.path + '/train/{}/{}/train_curve.png'.format(self.data_name,self.model_name))

    def test(self, model, dataloader, batch, filename):
        dataloader.training = False
        dataloader.batch = batch
        model.eval()

        if os.path.exists(filename+'.hdf5'):
            model.load(filename)
            logger.info('[*] weights loaded for testing.')
        else:
            logger.error('[*] File not found: weights cannot be loaded.') 
            raise FileNotFoundError
    
        logger.info('[*] testing started.')
        acc = 0
        for i, (data, label) in enumerate(dataloader): 
            output = model(self.data_transformer(data)) 
            out = np.argmax(output.data, axis=1) 
            ans = np.argmax(label.data, axis=1)
            acc += sum(out == ans)/label.shape[0]
            progressbar(i, len(dataloader))
        logger.info('\n[*] test acc: {:.2f}%'.format(float(acc/len(dataloader)*100)))
    
    def plot(self, filename=None):
        plt.plot([i for i in range(len(self.losses))], self.losses)
        plt.title('training losses of {} in {}'.format(self.model_name, self.data_name))
        plt.ylabel('training loss')
        plt.xlabel('epochs')
        if filename is not None:
            plt.savefig(filename)
        else:
            plt.show()
    
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
            logger.info('[*] weights loaded.') 
        else:
            logger.error('[*] File not found: weights cannot be loaded.')
            raise Exception
    dataloader.batch = minibatch
    dataloader.training = True
    model.training = True
    losses = []
    start = time.time()
    logger.info('[*] training started.')
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
    logger.info('[*] training completed.')

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
        logger.info('[*] weights loaded for testing.')
    else:
        logger.error('[*] File not found: weights cannot be loaded.') 
        raise Exception
    dataloader.batch = minibatch
    dataloader.training = False
    model.training = False
    logger.info('[*] testing started.')
    acc = 0
    for i, (data, label) in enumerate(dataloader): 
        output = model(data) 
        out = np.argmax(output.data, axis=1) 
        ans = np.argmax(label.data, axis=1)
        acc += sum(out == ans)/label.shape[0]
        progressbar(i, len(dataloader))
    logger.info('\n[*] test acc: {:.2f}%'.format(float(acc/len(dataloader)*100)))
